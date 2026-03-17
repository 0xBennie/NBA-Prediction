"""
auto_learner.py — 自主学习引擎（受 Karpathy autoresearch 启发）

核心思想:
  autoresearch: 修改train.py → 跑5分钟 → 比baseline → keep/discard
  我们:        修改scoring_params → 回测历史数据 → 比baseline → keep/discard

实验循环:
  1. 加载当前参数 (scoring_params.json) 作为 baseline
  2. 对baseline做小扰动 (mutation)
  3. 用历史 signal_log 数据回测，计算 metric (Sharpe-like)
  4. 如果 metric 提升 → 保存新参数 (advance)
  5. 如果没提升 → 回滚 (revert)
  6. 记录实验到 experiments.tsv
  7. 循环

评估指标:
  primary: 推送信号的 Kelly-weighted ROI (hypo_pnl 之和)
  secondary: 推送信号胜率 × (1 - max_drawdown)
  combined: sharpe_like = mean(roi) / std(roi) if std > 0 else 0

依赖: numpy (已有)
"""

import copy
import json
import logging
import random
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

PARAMS_PATH = Path("scoring_params.json")
EXPERIMENTS_PATH = Path("experiments.tsv")
MIN_BACKTEST_SAMPLES = 20  # 至少20条已结算信号才开始学习


class AutoLearner:
    """自主参数优化器 — 每次结果回填后自动运行一轮实验。"""

    def __init__(self, db_path: str = "nba_predictor.db"):
        self.db_path = db_path
        self.params = self._load_params()
        self._ensure_experiments_file()

    # ── 主入口 ──────────────────────────────────────────────────────
    def run_learning_cycle(self, n_experiments: int = 0) -> dict:
        """运行一轮学习循环（每次结果回填后调用）。

        Args:
            n_experiments: 本轮实验数量。0=自动根据样本量决定:
                           样本<50: 20次（多探索）
                           样本50-200: 10次
                           样本>200: 5次（已稳定，微调）

        Returns:
            {"improved": bool, "best_metric": float, "experiments_run": int}
        """
        samples = self._get_backtest_data()
        if len(samples) < MIN_BACKTEST_SAMPLES:
            logger.info(f"[AutoLearn] 样本不足({len(samples)}/{MIN_BACKTEST_SAMPLES})，跳过学习")
            return {"improved": False, "experiments_run": 0, "reason": "insufficient_samples"}

        # 自动决定实验次数：前期多探索，后期微调
        if n_experiments <= 0:
            if len(samples) < 50:
                n_experiments = 20
            elif len(samples) < 200:
                n_experiments = 10
            else:
                n_experiments = 5
            logger.info(f"[AutoLearn] 样本{len(samples)}条 → 本轮{n_experiments}次实验")

        # Step 1: 计算当前参数的 baseline metric
        baseline_metric = self._evaluate_params(self.params, samples)
        logger.info(f"[AutoLearn] Baseline metric: {baseline_metric:.4f} (样本:{len(samples)})")

        best_params = self.params
        best_metric = baseline_metric
        experiments_run = 0
        improved = False

        # Step 2: 实验循环 (like autoresearch's forever loop)
        for i in range(n_experiments):
            # 产生一个参数变异
            mutated = self._mutate_params(best_params)
            mut_metric = self._evaluate_params(mutated, samples)

            # 记录实验
            status = "keep" if mut_metric > best_metric else "discard"
            description = self._describe_mutation(best_params, mutated)
            self._log_experiment(
                experiment_id=f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}",
                metric=mut_metric,
                baseline=baseline_metric,
                status=status,
                description=description,
            )

            experiments_run += 1

            if mut_metric > best_metric:
                logger.info(
                    f"[AutoLearn] 实验#{i+1} 提升! {best_metric:.4f} → {mut_metric:.4f} ({description})"
                )
                best_params = mutated
                best_metric = mut_metric
                improved = True
            else:
                logger.debug(
                    f"[AutoLearn] 实验#{i+1} 未提升 {mut_metric:.4f} <= {best_metric:.4f} ({description})"
                )

        # Step 3: 如果有改进，保存新参数
        if improved:
            best_params["_meta"]["version"] = self.params["_meta"].get("version", 1) + 1
            best_params["_meta"]["updated_at"] = datetime.now().isoformat()
            best_params["_meta"]["updated_by"] = "auto_learner"
            best_params["_meta"]["baseline_metric"] = round(best_metric, 6)
            self._save_params(best_params)
            self.params = best_params
            logger.info(
                f"[AutoLearn] 参数已更新 v{best_params['_meta']['version']} "
                f"metric: {baseline_metric:.4f} → {best_metric:.4f}"
            )

        return {
            "improved": improved,
            "baseline_metric": round(baseline_metric, 4),
            "best_metric": round(best_metric, 4),
            "experiments_run": experiments_run,
            "samples": len(samples),
        }

    # ── 评估函数 ────────────────────────────────────────────────────
    def _evaluate_params(self, params: dict, samples: list) -> float:
        """用给定参数对历史数据重新评分，计算综合指标。

        指标设计（autoresearch用val_bpb，我们用Sharpe-like）:
          1. 模拟每条信号在新参数下的评分
          2. 筛选出达到推送阈值的信号
          3. 计算这些信号的ROI分布
          4. metric = mean(roi) / max(std(roi), 0.01) + 胜率加成
        """
        pred_cfg = params.get("prediction", {})
        min_confidence = pred_cfg.get("min_model_confidence", 0.60)
        min_value = pred_cfg.get("min_value_edge", 0.05)
        max_value = pred_cfg.get("max_value_edge", 0.08)
        max_buy = pred_cfg.get("max_buy_price", 0.65)
        min_buy = pred_cfg.get("min_buy_price", 0.28)

        pushed_rois = []
        pushed_correct = []

        for s in samples:
            poly_price = s.get("poly_price", 0.5) or 0.5
            raw_edge = s.get("raw_edge", 0) or 0
            eff_edge = s.get("effective_edge", raw_edge) or 0
            pin_prob = s.get("pinnacle_prob", 0) or 0

            # 模拟推送判断 — 兼容新旧数据
            # 旧数据: 用pinnacle_prob作为model_prob近似, raw_edge作为value_edge
            # 新数据: 从breakdown中获取
            breakdown = s.get("breakdown", {})
            if isinstance(breakdown, str):
                try:
                    breakdown = json.loads(breakdown)
                except Exception:
                    breakdown = {}

            # 提取model_prob
            mc = breakdown.get("model_confidence", {})
            model_prob = mc.get("model_prob", 0) if isinstance(mc, dict) else 0
            if model_prob == 0 and pin_prob > 0.1:
                model_prob = pin_prob  # 旧数据用pinnacle近似

            # 提取value_edge
            ve = breakdown.get("value_edge", {})
            value_edge = ve.get("edge", 0) if isinstance(ve, dict) else 0
            if value_edge == 0:
                value_edge = max(0, raw_edge)  # 旧数据用raw_edge

            # 推送判断 — 旧数据edge很小(0.005-0.03)，新数据value_edge更大(0.05+)
            # 用score评分+价格范围作为主过滤条件（新旧通用）
            score = s.get("score", 0) or 0
            would_push = (
                poly_price >= min_buy
                and poly_price <= max_buy
                and score >= 15  # 最低评分
                and (value_edge >= min_value or raw_edge > 0.005)  # 兼容旧数据小edge
            )

            if would_push and s.get("prediction_correct") is not None:
                roi = s.get("actual_roi", 0) or 0
                correct = s.get("prediction_correct", 0) or 0
                pushed_rois.append(roi)
                pushed_correct.append(correct)

        if len(pushed_rois) < 3:
            return -999.0

        rois = np.array(pushed_rois)
        corrects = np.array(pushed_correct)

        win_rate = np.mean(corrects)
        mean_roi = np.mean(rois)
        std_roi = max(np.std(rois), 0.001)
        sharpe = mean_roi / std_roi
        push_ratio = len(pushed_rois) / max(len(samples), 1)

        # 选择性: 推送率在8-20%之间得正分，太多或太少扣分
        if 0.08 <= push_ratio <= 0.20:
            selectivity = 0.2  # 合理范围，加分
        else:
            selectivity = -abs(push_ratio - 0.14) * 1.5  # 偏离目标，扣分

        # 综合指标: 胜率最重要(50%) + ROI(30%) + Sharpe(10%) + 选择性(10%)
        metric = win_rate * 1.5 + mean_roi * 2.0 + sharpe * 0.3 + selectivity

        return metric

    def _rescore_signal(self, params: dict, signal: dict) -> int:
        """用新参数重新计算信号评分（简化版mismatch_engine.score_game）。"""
        weights = params.get("game_weights", {})
        edge_cfg = params.get("edge_thresholds", {})
        price_cfg = params.get("price_position", {})

        breakdown = signal.get("breakdown", {})
        if isinstance(breakdown, str):
            try:
                breakdown = json.loads(breakdown)
            except Exception:
                breakdown = {}

        # Edge评分
        raw_edge = abs(signal.get("raw_edge", 0) or 0)
        edge_max = weights.get("edge_max", 40)

        if raw_edge >= edge_cfg.get("noise_floor", 0.03):
            edge_score = edge_cfg.get("noise_score", 15)
        elif raw_edge >= edge_cfg.get("high_min", 0.015):
            r = (raw_edge - edge_cfg["high_min"]) / max(edge_cfg["noise_floor"] - edge_cfg["high_min"], 0.001)
            edge_score = edge_cfg.get("high_base", 25) + r * (edge_max - edge_cfg.get("high_base", 25))
        elif raw_edge >= edge_cfg.get("sweet_min", 0.008):
            r = (raw_edge - edge_cfg["sweet_min"]) / max(edge_cfg["high_min"] - edge_cfg["sweet_min"], 0.001)
            edge_score = edge_cfg.get("sweet_base", 30) + r * 10
        elif raw_edge >= edge_cfg.get("mid_min", 0.005):
            r = (raw_edge - edge_cfg["mid_min"]) / max(edge_cfg["sweet_min"] - edge_cfg["mid_min"], 0.001)
            edge_score = edge_cfg.get("mid_base", 15) + r * 15
        elif raw_edge >= edge_cfg.get("low_min", 0.002):
            edge_score = raw_edge / 0.005 * 15
        else:
            edge_score = 0
        edge_score = min(edge_max, edge_score)

        # 盘口移动（直接用历史值，权重可调）
        lm = breakdown.get("line_movement", {})
        lm_score = lm.get("score", 0) if isinstance(lm, dict) else 0
        lm_max = weights.get("line_movement_max", 25)
        lm_score = min(lm_max, lm_score * lm_max / 25)  # 按新权重缩放

        # 伤病（按新权重缩放）
        inj = breakdown.get("injury_impact", {})
        inj_score = inj.get("score", 0) if isinstance(inj, dict) else 0
        inj_max = weights.get("injury_max", 20)
        inj_score = min(inj_max, inj_score * inj_max / 20)

        # 价格位置（用新阈值重新计算）
        poly_price = signal.get("poly_price", 0.5) or 0.5
        price_max = weights.get("price_position_max", 15)
        if poly_price >= price_cfg.get("dead_zone_high", 0.75):
            price_score = 0
        elif poly_price >= price_cfg.get("low_high", 0.65):
            price_score = price_cfg.get("low_score", 5)
        elif poly_price >= price_cfg.get("mid_high", 0.55):
            price_score = price_cfg.get("mid_score", 10)
        elif poly_price >= price_cfg.get("sweet_high", 0.40):
            price_score = price_cfg.get("sweet_score", 15)
        elif poly_price >= price_cfg.get("value_high", 0.30):
            price_score = price_cfg.get("value_score", 12)
        elif poly_price >= price_cfg.get("longshot_high", 0.20):
            price_score = price_cfg.get("longshot_score", 5)
        else:
            price_score = 0
        price_score = min(price_max, price_score)

        # B2B（用历史值）
        b2b = breakdown.get("b2b", {})
        b2b_score = b2b.get("score", 0) if isinstance(b2b, dict) else 0
        b2b_max = weights.get("b2b_max", 5)
        b2b_score = min(b2b_max, b2b_score)

        # 数据源惩罚
        source_penalty = breakdown.get("source_penalty", 0) or 0

        # ML修正（用历史值，不重新跑模型）
        ml_adj = breakdown.get("ml_adjustment", 0) or 0
        ml_range = params.get("ml", {}).get("adjustment_range", 15)
        ml_adj = max(-ml_range, min(ml_range, ml_adj))

        total = edge_score + lm_score + inj_score + price_score + b2b_score + source_penalty + ml_adj
        return int(round(min(100, max(0, total))))

    # ── 参数变异 ────────────────────────────────────────────────────
    def _mutate_params(self, params: dict) -> dict:
        """对参数做小扰动（autoresearch的核心: 改一个变量跑实验）。

        策略:
          - 随机选一个维度
          - 做±10-20%的扰动
          - 保持约束（如权重非负，阈值合理）
        """
        mutated = copy.deepcopy(params)

        # 可变异的维度及其范围
        # scale_small = 常规小扰动, scale_big = 大幅探索（模拟退火）
        mutations = [
            # (path, key, min_val, max_val, scale_small, scale_big)
            ("game_weights", "edge_max", 15, 55, 5, 15),
            ("game_weights", "line_movement_max", 5, 35, 5, 12),
            ("game_weights", "injury_max", 5, 30, 5, 10),
            ("game_weights", "price_position_max", 5, 25, 3, 8),
            ("game_weights", "b2b_max", 0, 10, 2, 5),
            ("edge_thresholds", "noise_floor", 0.015, 0.06, 0.005, 0.015),
            ("edge_thresholds", "noise_score", 5, 30, 3, 10),
            ("edge_thresholds", "sweet_min", 0.002, 0.015, 0.002, 0.005),
            ("edge_thresholds", "sweet_base", 10, 40, 5, 15),
            ("edge_thresholds", "mid_min", 0.001, 0.008, 0.001, 0.003),
            ("push_rules", "game_threshold", 5, 60, 5, 20),
            ("push_rules", "min_raw_edge", 0.001, 0.01, 0.001, 0.003),
            ("push_rules", "min_buy_price", 0.03, 0.30, 0.03, 0.10),
            ("push_rules", "max_buy_price", 0.70, 0.95, 0.05, 0.15),
            ("kelly", "fraction", 0.1, 0.5, 0.05, 0.15),
            ("kelly", "max_size", 0.02, 0.10, 0.01, 0.03),
            ("price_position", "sweet_high", 0.30, 0.55, 0.05, 0.10),
            ("price_position", "sweet_score", 8, 22, 2, 5),
            ("prediction", "model_weight", 0.1, 0.7, 0.05, 0.15),
            ("prediction", "min_model_confidence", 0.50, 0.75, 0.03, 0.10),
            ("prediction", "min_value_edge", 0.02, 0.10, 0.01, 0.03),
            ("prediction", "max_buy_price", 0.45, 0.75, 0.05, 0.10),
        ]

        # 30%概率大幅探索（跳出局部最优），70%小扰动微调
        is_explore = random.random() < 0.3
        # 随机选1-3个维度变异
        n_mutations = random.choice([1, 1, 2]) if not is_explore else random.choice([2, 3, 4])
        chosen = random.sample(mutations, min(n_mutations, len(mutations)))

        for section, key, min_val, max_val, scale_small, scale_big in chosen:
            if section not in mutated:
                continue
            current = mutated[section].get(key)
            if current is None:
                continue
            scale = scale_big if is_explore else scale_small
            # 探索模式下可以直接随机跳到范围内任意位置
            if is_explore and random.random() < 0.3:
                new_val = random.uniform(min_val, max_val)
            else:
                delta = random.uniform(-scale, scale)
                new_val = current + delta
            new_val = max(min_val, min(max_val, new_val))
            # 根据类型四舍五入
            if isinstance(current, int):
                new_val = int(round(new_val))
            else:
                new_val = round(new_val, 4)
            mutated[section][key] = new_val

        return mutated

    def _describe_mutation(self, old: dict, new: dict) -> str:
        """描述参数变化（类似autoresearch的实验描述）。"""
        changes = []
        for section in ["game_weights", "edge_thresholds", "push_rules",
                         "kelly", "price_position", "ml"]:
            old_sec = old.get(section, {})
            new_sec = new.get(section, {})
            for key in new_sec:
                if key.startswith("_"):
                    continue
                if old_sec.get(key) != new_sec.get(key):
                    changes.append(f"{section}.{key}: {old_sec.get(key)} → {new_sec[key]}")
        return "; ".join(changes) if changes else "no change"

    # ── 数据获取 ────────────────────────────────────────────────────
    def _get_backtest_data(self) -> list:
        """从signal_log获取已结算信号数据。"""
        try:
            con = sqlite3.connect(self.db_path, timeout=30)
            con.row_factory = sqlite3.Row
            rows = con.execute("""
                SELECT game_id, poly_price, pinnacle_prob, raw_edge, effective_edge,
                       score, kelly, was_pushed, breakdown_json,
                       prediction_correct, actual_roi, hypo_pnl
                FROM signal_log
                WHERE prediction_correct IS NOT NULL
                ORDER BY scanned_at DESC
                LIMIT 2000
            """).fetchall()
            con.close()

            result = []
            for r in rows:
                d = dict(r)
                # 解析breakdown_json
                if d.get("breakdown_json"):
                    try:
                        d["breakdown"] = json.loads(d["breakdown_json"])
                    except Exception:
                        d["breakdown"] = {}
                else:
                    d["breakdown"] = {}
                result.append(d)
            return result
        except Exception as e:
            logger.warning(f"[AutoLearn] 获取回测数据失败: {e}")
            return []

    # ── 实验记录 ────────────────────────────────────────────────────
    def _ensure_experiments_file(self):
        """确保experiments.tsv存在且有header（类似autoresearch的results.tsv）。"""
        if not EXPERIMENTS_PATH.exists():
            EXPERIMENTS_PATH.write_text(
                "timestamp\texperiment_id\tmetric\tbaseline\tstatus\tdescription\n"
            )

    def _log_experiment(self, experiment_id: str, metric: float, baseline: float,
                        status: str, description: str):
        """记录实验结果到TSV（tab-separated，和autoresearch一致）。"""
        with open(EXPERIMENTS_PATH, "a") as f:
            f.write(
                f"{datetime.now().isoformat()}\t{experiment_id}\t"
                f"{metric:.6f}\t{baseline:.6f}\t{status}\t{description}\n"
            )

    # ── 参数IO ──────────────────────────────────────────────────────
    def _load_params(self) -> dict:
        if PARAMS_PATH.exists():
            try:
                return json.loads(PARAMS_PATH.read_text())
            except Exception as e:
                logger.warning(f"[AutoLearn] 加载参数失败: {e}")
        return self._default_params()

    def _save_params(self, params: dict):
        PARAMS_PATH.write_text(json.dumps(params, indent=4, ensure_ascii=False))

    @staticmethod
    def _default_params() -> dict:
        return {
            "_meta": {"version": 1, "updated_at": None, "updated_by": "baseline",
                      "baseline_metric": None},
            "game_weights": {"edge_max": 40, "line_movement_max": 25, "injury_max": 20,
                            "price_position_max": 15, "b2b_max": 5, "source_penalty_max": -15},
            "edge_thresholds": {"noise_floor": 0.03, "noise_score": 15, "high_min": 0.015,
                               "high_base": 25, "sweet_min": 0.008, "sweet_base": 30,
                               "mid_min": 0.005, "mid_base": 15, "low_min": 0.002},
            "price_position": {"dead_zone_high": 0.75, "low_high": 0.65, "low_score": 5,
                              "mid_high": 0.55, "mid_score": 10, "sweet_high": 0.40,
                              "sweet_score": 15, "value_high": 0.30, "value_score": 12,
                              "longshot_high": 0.20, "longshot_score": 5},
            "push_rules": {"game_threshold": 50, "min_raw_edge": 0.004,
                          "min_buy_price": 0.25, "max_buy_price": 0.72},
            "kelly": {"fraction": 0.25, "max_size": 0.05},
            "ml": {"min_samples": 30, "adjustment_range": 15, "adjustment_multiplier": 30},
        }

    # ── 报告 ────────────────────────────────────────────────────────
    def get_learning_report(self) -> str:
        """生成学习进度报告。"""
        meta = self.params.get("_meta", {})
        version = meta.get("version", 1)
        updated = meta.get("updated_at", "never")
        metric = meta.get("baseline_metric", "N/A")

        # 读实验历史
        experiments = []
        if EXPERIMENTS_PATH.exists():
            lines = EXPERIMENTS_PATH.read_text().strip().split("\n")[1:]  # skip header
            experiments = lines

        total_exp = len(experiments)
        kept = sum(1 for l in experiments if "\tkeep\t" in l)

        report = (
            f"📚 自学习系统报告\n"
            f"━━━━━━━━━━━━━━━\n"
            f"参数版本: v{version}\n"
            f"最后更新: {updated}\n"
            f"当前metric: {metric}\n"
            f"总实验数: {total_exp}\n"
            f"成功改进: {kept}\n"
            f"改进率: {kept/total_exp:.0%}\n" if total_exp > 0 else f"改进率: N/A\n"
        )

        # 最近5条实验
        if experiments:
            report += f"\n最近实验:\n"
            for line in experiments[-5:]:
                parts = line.split("\t")
                if len(parts) >= 6:
                    status_icon = "✅" if parts[4] == "keep" else "❌"
                    report += f"  {status_icon} {parts[3]} → {parts[2]} {parts[5][:60]}\n"

        return report
