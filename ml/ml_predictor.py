"""
ml_predictor.py — 机器学习修正层（研究校准版）

工作原理:
  规则引擎打出 0-100 分 → ML 在此基础上做 ±15 分修正
  随推送结果积累自动重训练，冷启动前(< MIN_SAMPLES)输出0不介入

研究依据:
  - 校准(calibration)比准确率(accuracy)更重要: Kelly只在校准良好时可靠
  - ML不应"复制赔率"，而应学习"与市场预测的残差"(去相关)
  - 用CalibratedClassifierCV确保概率校准
  - line_moved从布尔升级为连续值(盘口变化幅度)

模型进化路径:
  < 30条  → 不介入
  30-100  → 逻辑回归（强正则化 + 概率校准）
  100+    → LightGBM（限深防过拟合 + 概率校准）

依赖: pip install scikit-learn lightgbm numpy
"""

import json
import logging
import pickle
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

MIN_SAMPLES = 30
MODEL_PATH = Path("ml_model.pkl")
SCALER_PATH = Path("ml_scaler.pkl")


# ── 特征定义 ──────────────────────────────────────────────────────
# 比赛盘特征（8维）
# 研究: 特征应与市场预测"去相关"，学习残差而非复制赔率
GAME_FEATURES = [
    "edge",              # Pinnacle公平概率 - Polymarket有效买入价（扣除成本后的真实edge）
    "poly_price",        # Polymarket客队价格本身
    "injury_delta",      # 主队伤病Elo - 客队伤病Elo（正值=主队伤更重）
    "home_b2b",          # 主队背靠背(0/1)
    "away_b2b",          # 客队背靠背(0/1)
    "line_moved",        # 盘口变化幅度（连续值，非布尔；研究:连续量化更有信息量）
    "volume_ratio",      # 成交量/基准值
    "source_quality",    # 数据源质量分(Pinnacle=1.0, DK=0.7, FD=0.7, AN=0.6)
]

# 期货盘特征（8维）
FUTURES_FEATURES = [
    "price",             # 当前价格
    "fundamental_edge",  # 基本面隐含概率 - 市场价格
    "daily_change",      # 24h价格变动（有方向性）
    "weekly_change",     # 7d价格变动
    "volume_ratio",      # 成交量/基准值
    "depth_score",       # 市场深度得分(0-15)
    "days_to_resolution",# 结算天数
    "market_type_enc",   # 期货类型编码(champion=0,mvp=1,division=2,playoff=3)
]

MARKET_TYPE_ENC = {"champion": 0, "mvp": 1, "division": 2, "playoff": 3, "roy": 4, "other": 5}
# 数据源质量分 — 4级瀑布回退
SOURCE_QUALITY = {
    "pinnacle_via_odds_api": 1.0,
    "pinnacle": 1.0,
    "draftkings": 0.85,
    "fanduel": 0.85,
    "consensus": 0.7,
    "unknown": 0.5,
}


class MLPredictor:

    def __init__(self, db_path: str = "nba_predictor.db"):
        self.db_path = db_path
        self.game_model = None
        self.game_scaler = None
        self.futures_model = None
        self.futures_scaler = None
        self._load_models()

    # ── 推理接口 ──────────────────────────────────────────────────
    def get_game_adjustment(self, features: dict) -> float:
        """比赛盘ML修正值，范围 [-15, +15]"""
        return self._predict(features, "game")

    def get_futures_adjustment(self, features: dict) -> float:
        """期货盘ML修正值，范围 [-15, +15]"""
        return self._predict(features, "futures")

    def _predict(self, features: dict, mtype: str) -> float:
        n = self._sample_count(mtype)
        if n < MIN_SAMPLES:
            logger.debug(f"[ML] 样本不足({n}/{MIN_SAMPLES})，跳过修正")
            return 0.0

        model = self.game_model if mtype == "game" else self.futures_model
        scaler = self.game_scaler if mtype == "game" else self.futures_scaler
        feat_names = GAME_FEATURES if mtype == "game" else FUTURES_FEATURES

        if model is None or scaler is None:
            return 0.0

        try:
            vec = [features.get(f, 0.0) for f in feat_names]
            X = np.array([vec])
            X_scaled = scaler.transform(X)
            win_prob = model.predict_proba(X_scaled)[0][1]
            # 0.5 → 0修正，越偏离0.5修正越大，上限±15
            adj = (win_prob - 0.5) * 30
            return round(max(-15.0, min(15.0, adj)), 2)
        except Exception as e:
            logger.warning(f"[ML] 推理失败: {e}")
            return 0.0

    # ── 特征构建工具 ──────────────────────────────────────────────
    @staticmethod
    def build_game_features(
        edge: float,
        poly_price: float,
        injury_delta: float,
        home_b2b: bool,
        away_b2b: bool,
        line_moved: float,  # 连续值: 盘口变化幅度（研究推荐连续量化）
        volume_24h: float,
        source: str,
    ) -> dict:
        return {
            "edge": edge,
            "poly_price": poly_price,
            "injury_delta": injury_delta,
            "home_b2b": float(home_b2b),
            "away_b2b": float(away_b2b),
            "line_moved": float(line_moved),  # 连续值，非布尔
            "volume_ratio": min(volume_24h / 5000, 10.0),
            "source_quality": SOURCE_QUALITY.get(source, 0.5),
        }

    @staticmethod
    def build_futures_features(
        price: float,
        fundamental_edge: float,
        daily_change: float,
        weekly_change: float,
        volume_24h: float,
        baseline_volume: float,
        depth_score: float,
        days_to_resolution: int,
        market_type: str,
    ) -> dict:
        return {
            "price": price,
            "fundamental_edge": fundamental_edge,
            "daily_change": daily_change,
            "weekly_change": weekly_change,
            "volume_ratio": volume_24h / max(baseline_volume, 1),
            "depth_score": depth_score,
            "days_to_resolution": days_to_resolution,
            "market_type_enc": float(MARKET_TYPE_ENC.get(market_type, 5)),
        }

    # ── 数据存储 ──────────────────────────────────────────────────
    def save_push(self, push_id: str, mtype: str, features: dict,
                  rule_score: int, ml_adjustment: float):
        """推送时保存特征快照"""
        with self._conn() as c:
            c.execute("""
                INSERT OR IGNORE INTO ml_features
                    (push_id, market_type, features_json, rule_score, ml_adjustment)
                VALUES (?, ?, ?, ?, ?)
            """, (push_id, mtype, json.dumps(features), rule_score, ml_adjustment))

    def record_result(self, push_id: str, outcome: int, pnl: float):
        """
        比赛/期货结束后回填结果。
        outcome: 1=我们推的方向赢了, 0=输了
        pnl: 每单位盈亏（赢=(1/price)-1, 输=-1）
        """
        with self._conn() as c:
            c.execute("""
                UPDATE push_results
                SET actual_outcome=?, pnl_per_unit=?, resolved_at=CURRENT_TIMESTAMP
                WHERE push_id=?
            """, (outcome, pnl, push_id))

        self._maybe_retrain()

    # ── 训练 ──────────────────────────────────────────────────────
    def maybe_retrain(self):
        self._maybe_retrain()

    def force_retrain(self):
        logger.info("[ML] 强制重训练...")
        self._retrain("game")
        self._retrain("futures")

    def _maybe_retrain(self):
        for mtype in ["game", "futures"]:
            n = self._sample_count(mtype)
            if n >= MIN_SAMPLES and n % 10 == 0:
                self._retrain(mtype)

    def _retrain(self, mtype: str):
        rows = self._get_training_data(mtype)
        if len(rows) < MIN_SAMPLES:
            return

        feat_names = GAME_FEATURES if mtype == "game" else FUTURES_FEATURES
        X, y = [], []
        for features_json, outcome in rows:
            try:
                f = json.loads(features_json)
                # signal_log的breakdown_json可能嵌套结构，提取特征
                if "edge_score" in f and "edge" not in f:
                    # 这是breakdown格式，需要从中提取ML特征
                    f.setdefault("edge", f.get("effective_edge", f.get("raw_edge", 0)))
                    f.setdefault("source_quality", 1.0)
                    inj = f.get("injury_impact", {})
                    f.setdefault("injury_delta", inj.get("delta", 0) if isinstance(inj, dict) else 0)
                    b2b = f.get("b2b", {})
                    f.setdefault("home_b2b", float(b2b.get("home_b2b", False)) if isinstance(b2b, dict) else 0)
                    f.setdefault("away_b2b", float(b2b.get("away_b2b", False)) if isinstance(b2b, dict) else 0)
                    lm = f.get("line_movement", {})
                    f.setdefault("line_moved", lm.get("poly_shift", 0) if isinstance(lm, dict) else 0)
                    f.setdefault("volume_ratio", 0)
                    f.setdefault("poly_price", 0)
                X.append([f.get(k, 0.0) for k in feat_names])
                y.append(outcome)
            except Exception:
                continue

        X = np.array(X)
        y = np.array(y)

        # 根据样本量选择模型
        if len(rows) < 100:
            model, scaler = self._train_logistic(X, y)
            method = "LogisticRegression"
        else:
            model, scaler = self._train_lgbm(X, y)
            method = "LightGBM"

        # 保存
        suffix = mtype
        with open(f"ml_model_{suffix}.pkl", "wb") as f:
            pickle.dump(model, f)
        with open(f"ml_scaler_{suffix}.pkl", "wb") as f:
            pickle.dump(scaler, f)

        if mtype == "game":
            self.game_model, self.game_scaler = model, scaler
        else:
            self.futures_model, self.futures_scaler = model, scaler

        wins = int(sum(y))
        logger.info(f"[ML] {mtype} 模型更新 | {method} | 样本:{len(rows)} 胜率:{wins/len(rows):.1%}")

    def _train_logistic(self, X, y):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.calibration import CalibratedClassifierCV

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        base = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
        # 样本够时加概率校准
        if len(X) >= 50:
            model = CalibratedClassifierCV(base, cv=min(5, len(X) // 10))
        else:
            model = base
        model.fit(Xs, y)
        return model, scaler

    def _train_lgbm(self, X, y):
        """LightGBM + 概率校准（研究: 校准比准确率更重要，Kelly依赖良好校准）"""
        from sklearn.preprocessing import StandardScaler
        from sklearn.calibration import CalibratedClassifierCV
        try:
            import lightgbm as lgb
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
            base = lgb.LGBMClassifier(
                n_estimators=50,
                max_depth=4,
                min_child_samples=10,
                learning_rate=0.05,
                random_state=42,
                verbose=-1,
            )
            # 校准概率输出（研究推荐: 用calibration指标选模型，ROI显著更好）
            if len(X) >= 100:
                model = CalibratedClassifierCV(base, cv=min(5, len(X) // 20), method='isotonic')
            else:
                model = base
            model.fit(Xs, y)
            return model, scaler
        except ImportError:
            logger.warning("[ML] LightGBM未安装，降级到LogisticRegression")
            return self._train_logistic(X, y)

    # ── 报告 ──────────────────────────────────────────────────────
    def get_performance_report(self) -> dict:
        with self._conn() as c:
            c.row_factory = sqlite3.Row
            total = c.execute(
                "SELECT COUNT(*) FROM push_results WHERE actual_outcome IS NOT NULL"
            ).fetchone()[0]
            wins = c.execute(
                "SELECT COUNT(*) FROM push_results WHERE actual_outcome=1"
            ).fetchone()[0]
            avg_pnl = c.execute(
                "SELECT AVG(pnl_per_unit) FROM push_results WHERE actual_outcome IS NOT NULL"
            ).fetchone()[0]
            pending = c.execute(
                "SELECT COUNT(*) FROM push_results WHERE actual_outcome IS NULL"
            ).fetchone()[0]

            # 按edge分组的胜率
            edge_stats = c.execute("""
                SELECT
                    CASE
                        WHEN pr.edge_at_push >= 0.08 THEN 'edge_8%+'
                        WHEN pr.edge_at_push >= 0.05 THEN 'edge_5-8%'
                        WHEN pr.edge_at_push >= 0.03 THEN 'edge_3-5%'
                        ELSE 'edge_<3%'
                    END as bucket,
                    COUNT(*) as cnt,
                    AVG(pr.actual_outcome) as win_rate
                FROM push_results pr
                WHERE pr.actual_outcome IS NOT NULL
                GROUP BY bucket
            """).fetchall()

        return {
            "total_resolved": total,
            "win_rate": round(wins / total, 3) if total > 0 else None,
            "avg_pnl": round(avg_pnl, 4) if avg_pnl else None,
            "pending": pending,
            "ml_active": self.game_model is not None,
            "samples_needed": max(0, MIN_SAMPLES - total),
            "edge_breakdown": [dict(r) for r in edge_stats],
        }

    # ── 内部工具 ──────────────────────────────────────────────────
    def _conn(self):
        from contextlib import contextmanager

        @contextmanager
        def _ctx():
            con = sqlite3.connect(self.db_path, timeout=30)
            try:
                yield con
                con.commit()
            except Exception:
                con.rollback()
                raise
            finally:
                con.close()
        return _ctx()

    def _sample_count(self, mtype: str) -> int:
        """从signal_log获取已结算信号数量（全量，不只是推送的）"""
        try:
            with self._conn() as c:
                # 优先用signal_log全量数据
                count = c.execute("""
                    SELECT COUNT(*) FROM signal_log
                    WHERE market_type=? AND prediction_correct IS NOT NULL
                """, (mtype,)).fetchone()[0]
                if count > 0:
                    return count
                # 回退: 旧数据可能只在push_results里
                return c.execute("""
                    SELECT COUNT(*) FROM ml_features f
                    JOIN push_results r ON f.push_id = r.push_id
                    WHERE f.market_type=? AND r.actual_outcome IS NOT NULL
                """, (mtype,)).fetchone()[0]
        except Exception:
            return 0

    def _get_training_data(self, mtype: str) -> list:
        """从signal_log获取训练数据（全量信号，不只是推送的）"""
        with self._conn() as c:
            # 优先用signal_log全量（包括未推送的信号，更全面）
            rows = c.execute("""
                SELECT breakdown_json, prediction_correct
                FROM signal_log
                WHERE market_type=? AND prediction_correct IS NOT NULL
                  AND breakdown_json IS NOT NULL
                ORDER BY scanned_at DESC
                LIMIT 2000
            """, (mtype,)).fetchall()
            if rows:
                return rows
            # 回退: 旧数据
            return c.execute("""
                SELECT f.features_json, r.actual_outcome
                FROM ml_features f
                JOIN push_results r ON f.push_id = r.push_id
                WHERE f.market_type=? AND r.actual_outcome IS NOT NULL
                ORDER BY f.pushed_at DESC
                LIMIT 2000
            """, (mtype,)).fetchall()

    def _load_models(self):
        for mtype in ["game", "futures"]:
            try:
                with open(f"ml_model_{mtype}.pkl", "rb") as f:
                    model = pickle.load(f)
                with open(f"ml_scaler_{mtype}.pkl", "rb") as f:
                    scaler = pickle.load(f)
                if mtype == "game":
                    self.game_model, self.game_scaler = model, scaler
                else:
                    self.futures_model, self.futures_scaler = model, scaler
                logger.info(f"[ML] 加载历史模型: {mtype}")
            except FileNotFoundError:
                pass
