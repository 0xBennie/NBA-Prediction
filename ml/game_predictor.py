"""
game_predictor.py — NBA比赛胜负预测模型（v4 XGBoost+Optuna优化版）

优化来源:
  - kyleskom/NBA-Machine-Learning-Sports-Betting: XGBoost + Optuna超参搜索
  - 学术研究: TimeSeriesSplit防数据泄漏, 独立校准集, Sample Weights

模型进化:
  < 50 场  → Logistic Regression (强正则)
  50-200   → XGBoost (默认参数 + 校准)
  200+     → XGBoost + Optuna超参搜索(20轮) + 独立校准集 + TimeSeriesSplit

特征: 39维 (每队18维 + 3全局)
"""

import json
import logging
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

MODEL_PATH = Path("game_predictor_model.pkl")

# 特征名（v3: 39维）
FEATURE_NAMES = [
    "home_win_pct_10", "home_net_rating_10", "home_avg_pts_10",
    "home_efg_pct", "home_tov_rate", "home_oreb_rate", "home_ft_rate",
    "home_season_win_pct", "home_season_net",
    "home_venue_win_pct", "home_schedule_density", "home_rest_days", "home_fatigue_minutes",
    "home_efg_trend", "home_injury_elo",
    "home_off_rating", "home_def_rating", "home_streak",
    "away_win_pct_10", "away_net_rating_10", "away_avg_pts_10",
    "away_efg_pct", "away_tov_rate", "away_oreb_rate", "away_ft_rate",
    "away_season_win_pct", "away_season_net",
    "away_venue_win_pct", "away_schedule_density", "away_rest_days", "away_fatigue_minutes",
    "away_efg_trend", "away_injury_elo",
    "away_off_rating", "away_def_rating", "away_streak",
    "home_opp_strength", "away_opp_strength", "home_advantage",
]


class SigmoidCalibrator:
    """手动Sigmoid校准器（兼容sklearn 1.8+，可pickle）。"""
    def __init__(self, base_model, lr):
        self.base_model = base_model
        self.lr = lr
    def predict_proba(self, X):
        raw = self.base_model.predict_proba(X)[:, 1]
        return self.lr.predict_proba(raw.reshape(-1, 1))
    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)


class GamePredictor:
    """NBA比赛胜负预测器（v4 XGBoost+Optuna）。"""

    def __init__(self, db_path: str = "nba_predictor.db"):
        self.db_path = db_path
        self.model = None
        self.scaler = None
        self.calibrator = None
        self._load_model()

    def predict(self, features: dict) -> float:
        """预测主队赢的概率。返回校准后的 P(home_wins)。"""
        if self.model is None or self.scaler is None:
            return self._simple_predict(features)

        try:
            vec = [features.get(f, 0.0) for f in FEATURE_NAMES]
            X = np.array([vec])
            X_scaled = self.scaler.transform(X)

            # 如果有独立校准器，用校准器
            if self.calibrator is not None:
                prob = self.calibrator.predict_proba(X_scaled)[0][1]
            else:
                prob = self.model.predict_proba(X_scaled)[0][1]

            return float(prob)
        except Exception as e:
            logger.warning(f"[Predictor] 预测失败: {e}")
            return self._simple_predict(features)

    def _simple_predict(self, features: dict) -> float:
        """无模型时的简单预测。"""
        home_wp = features.get("home_win_pct_10", 0.5)
        away_wp = features.get("away_win_pct_10", 0.5)
        home_net = features.get("home_net_rating_10", 0)
        away_net = features.get("away_net_rating_10", 0)
        home_venue = features.get("home_venue_win_pct", 0.5)
        away_venue = features.get("away_venue_win_pct", 0.5)
        home_efg = features.get("home_efg_pct", 0.50)
        away_efg = features.get("away_efg_pct", 0.50)
        home_trend = features.get("home_efg_trend", 0)
        away_trend = features.get("away_efg_trend", 0)
        home_density = features.get("home_schedule_density", 2)
        away_density = features.get("away_schedule_density", 2)
        home_rest = features.get("home_rest_days", 1)
        away_rest = features.get("away_rest_days", 1)
        home_inj = features.get("home_injury_elo", 0)
        away_inj = features.get("away_injury_elo", 0)

        base = (home_wp - away_wp) * 0.25
        venue_adj = (home_venue - 0.5) * 0.15 + (0.5 - away_venue) * 0.10
        net_adj = (home_net - away_net) / 10 * 0.12
        efg_adj = (home_efg - away_efg) * 0.8
        trend_adj = (home_trend - away_trend) * 0.3
        home_adj = 0.03
        density_adj = (away_density - home_density) * 0.015
        rest_adj = (home_rest - away_rest) * 0.004
        inj_adj = (away_inj - home_inj) / 100 * 0.05

        prob = 0.5 + base + venue_adj + net_adj + efg_adj + trend_adj + home_adj + density_adj + rest_adj + inj_adj
        return max(0.15, min(0.85, prob))

    # ── 训练 ──────────────────────────────────────────────────────
    def train(self, force: bool = False) -> dict:
        """训练预测模型（自动选择最佳方案）。"""
        X, y = self._get_training_data()
        n = len(X)
        if n < 20 and not force:
            return {"samples": n, "accuracy": 0, "method": "none"}

        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if n < 50:
            model, calibrator, method, accuracy = self._train_logistic(X_scaled, y)
        elif n < 200:
            model, calibrator, method, accuracy = self._train_xgb_default(X_scaled, y)
        else:
            model, calibrator, method, accuracy = self._train_xgb_optuna(X_scaled, y)

        # 保存
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({"model": model, "scaler": scaler, "calibrator": calibrator}, f)
        self.model = model
        self.scaler = scaler
        self.calibrator = calibrator

        logger.info(f"[Predictor] 模型训练完成 | {method} | 样本:{n} 准确率:{accuracy:.1%}")
        return {"samples": n, "accuracy": round(accuracy, 3), "method": method}

    def _train_logistic(self, X, y):
        """小样本: Logistic Regression。"""
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(C=0.01, max_iter=1000, random_state=42)
        model.fit(X, y)
        acc = float(np.mean(model.predict(X) == y))
        return model, None, "LogisticRegression", acc

    def _train_xgb_default(self, X, y):
        """中等样本: XGBoost默认参数 + 校准。"""
        import xgboost as xgb
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.model_selection import TimeSeriesSplit

        base = xgb.XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=5, random_state=42,
            eval_metric="logloss", verbosity=0,
        )

        weights = self._compute_sample_weights(y)

        # TimeSeriesSplit校准（防数据泄漏）
        tss = TimeSeriesSplit(n_splits=3)
        calibrator = CalibratedClassifierCV(base, cv=tss, method="sigmoid")
        calibrator.fit(X, y, sample_weight=weights)

        acc = float(np.mean(calibrator.predict(X) == y))
        return base, calibrator, "XGBoost+calibrated", acc

    def _train_xgb_optuna(self, X, y):
        """大样本: XGBoost + Optuna超参搜索 + 独立校准。"""
        import xgboost as xgb
        import optuna
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.metrics import log_loss

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # 数据分割: 70% train, 15% valid, 15% calibration
        n = len(X)
        train_end = int(n * 0.70)
        valid_end = int(n * 0.85)

        X_train, y_train = X[:train_end], y[:train_end]
        X_valid, y_valid = X[train_end:valid_end], y[train_end:valid_end]
        X_calib, y_calib = X[valid_end:], y[valid_end:]

        weights = self._compute_sample_weights(y_train)

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 2, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 5.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
                "random_state": 42,
                "eval_metric": "logloss",
                "verbosity": 0,
            }

            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, sample_weight=weights)
            probs = model.predict_proba(X_valid)
            return log_loss(y_valid, probs)

        # Optuna搜索20轮（平衡速度和质量）
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=20, show_progress_bar=False)

        # 用最优参数重新训练
        best_params = study.best_params
        best_params["random_state"] = 42
        best_params["eval_metric"] = "logloss"
        best_params["verbosity"] = 0

        best_model = xgb.XGBClassifier(**best_params)
        # 用 train+valid 一起训练最终模型
        X_final = np.vstack([X_train, X_valid])
        y_final = np.concatenate([y_train, y_valid])
        w_final = self._compute_sample_weights(y_final)
        best_model.fit(X_final, y_final, sample_weight=w_final)

        # Sigmoid校准（兼容sklearn 1.8+）
        from sklearn.linear_model import LogisticRegression as LR
        raw_probs = best_model.predict_proba(X_calib)[:, 1]
        calib_lr = LR(C=1e10, solver="lbfgs", max_iter=1000)
        calib_lr.fit(raw_probs.reshape(-1, 1), y_calib)
        calibrator = SigmoidCalibrator(best_model, calib_lr)

        acc = float(np.mean(calibrator.predict(X) == y))

        logger.info(
            f"[Predictor] Optuna最优参数: depth={best_params.get('max_depth')} "
            f"lr={best_params.get('learning_rate', 0):.3f} "
            f"trees={best_params.get('n_estimators')} "
            f"logloss={study.best_value:.4f}"
        )

        return best_model, calibrator, "XGBoost+Optuna+calibrated", acc

    @staticmethod
    def _compute_sample_weights(y):
        """计算样本权重（平衡主客场类别不平衡）。"""
        counts = np.bincount(y, minlength=2)
        total = len(y)
        class_weights = {
            cls: (total / (2 * count)) if count > 0 else 1.0
            for cls, count in enumerate(counts)
        }
        return np.array([class_weights[label] for label in y])

    # ── 冷启动 ────────────────────────────────────────────────────
    def bootstrap_historical(self) -> int:
        """从nba_api拉取本赛季历史比赛。"""
        try:
            from nba_api.stats.endpoints import LeagueGameLog
            import sqlite3

            logger.info("[Predictor] 开始导入历史比赛数据...")
            log = LeagueGameLog(
                season="2025-26",
                season_type_all_star="Regular Season",
                timeout=30,
            )
            df = log.get_data_frames()[0]

            games = {}
            for _, row in df.iterrows():
                gid = row["GAME_ID"]
                if gid not in games:
                    games[gid] = {}
                matchup = row.get("MATCHUP", "")
                is_home = "vs." in matchup
                side = "home" if is_home else "away"
                games[gid][side] = {
                    "team": row["TEAM_ABBREVIATION"],
                    "pts": int(row["PTS"]),
                    "wl": row["WL"],
                    "date": str(row["GAME_DATE"])[:10],
                    "plus_minus": float(row.get("PLUS_MINUS", 0)),
                }

            con = sqlite3.connect(self.db_path, timeout=30)
            con.execute("""
                CREATE TABLE IF NOT EXISTS historical_games (
                    game_id TEXT PRIMARY KEY,
                    home_team TEXT NOT NULL, away_team TEXT NOT NULL,
                    game_date TEXT NOT NULL,
                    home_score INTEGER, away_score INTEGER,
                    home_won INTEGER, features_json TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            count = 0
            for gid, sides in games.items():
                if "home" not in sides or "away" not in sides:
                    continue
                h, a = sides["home"], sides["away"]
                home_won = 1 if h["wl"] == "W" else 0
                try:
                    con.execute("""
                        INSERT OR IGNORE INTO historical_games
                            (game_id, home_team, away_team, game_date, home_score, away_score, home_won)
                        VALUES (?,?,?,?,?,?,?)
                    """, (gid, h["team"], a["team"], h["date"], h["pts"], a["pts"], home_won))
                    count += 1
                except Exception:
                    continue

            con.commit()
            con.close()
            logger.info(f"[Predictor] 导入{count}场历史比赛")
            return count
        except Exception as e:
            logger.warning(f"[Predictor] 历史数据导入失败: {e}")
            return 0

    def _get_training_data(self):
        """获取训练数据。"""
        import sqlite3
        con = sqlite3.connect(self.db_path, timeout=30)
        con.row_factory = sqlite3.Row

        X, y = [], []

        # 从historical_games
        rows = con.execute("""
            SELECT features_json, home_won FROM historical_games
            WHERE features_json IS NOT NULL AND home_won IS NOT NULL
            LIMIT 5000
        """).fetchall()
        for r in rows:
            try:
                f = json.loads(r["features_json"])
                vec = [f.get(k, 0.0) for k in FEATURE_NAMES]
                X.append(vec)
                y.append(r["home_won"])
            except Exception:
                continue

        # 从signal_log补充
        rows2 = con.execute("""
            SELECT breakdown_json, actual_outcome, buy_side
            FROM signal_log
            WHERE actual_outcome IS NOT NULL AND breakdown_json IS NOT NULL
        """).fetchall()
        for r in rows2:
            try:
                bd = json.loads(r["breakdown_json"])
                if "home_win_pct_10" in bd:
                    vec = [bd.get(k, 0.0) for k in FEATURE_NAMES]
                    home_won = 0 if r["actual_outcome"] == 1 else 1
                    X.append(vec)
                    y.append(home_won)
            except Exception:
                continue

        con.close()
        return np.array(X) if X else np.array([]).reshape(0, len(FEATURE_NAMES)), np.array(y)

    def _load_model(self):
        try:
            with open(MODEL_PATH, "rb") as f:
                data = pickle.load(f)
            self.model = data["model"]
            self.scaler = data["scaler"]
            self.calibrator = data.get("calibrator")
            logger.info("[Predictor] 加载预测模型")
        except FileNotFoundError:
            pass

    def get_report(self) -> dict:
        import sqlite3
        con = sqlite3.connect(self.db_path, timeout=30)
        total = 0
        with_features = 0
        try:
            total = con.execute("SELECT COUNT(*) FROM historical_games").fetchone()[0]
            with_features = con.execute(
                "SELECT COUNT(*) FROM historical_games WHERE features_json IS NOT NULL"
            ).fetchone()[0]
        except Exception:
            pass
        con.close()
        return {
            "model_loaded": self.model is not None,
            "has_calibrator": self.calibrator is not None,
            "historical_games": total,
            "with_features": with_features,
        }
