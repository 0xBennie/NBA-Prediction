"""
backtester.py — 投注策略回测框架

基于 sports-betting (scikit-learn兼容) 框架思路，实现：
  - 历史数据回测（walk-forward，防数据泄漏）
  - 多策略对比（Kelly, 固定仓位, 阈值优化）
  - ROI / Sharpe / MaxDrawdown 计算
  - 校准图（calibration curve）

依赖: numpy, scikit-learn
可选: sports-betting (如安装则使用其 Bettor 类)
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class Backtester:
    """NBA投注策略回测器。"""

    def __init__(self, db_path: str = "nba_predictor.db"):
        self.db_path = db_path

    def run_backtest(self, strategy: str = "kelly") -> dict:
        """运行完整回测。

        Args:
            strategy: "kelly" | "fixed" | "threshold"

        Returns:
            回测报告字典
        """
        signals = self._load_resolved_signals()
        if not signals:
            return {"status": "no_data", "reason": "无已结算信号数据"}

        results = []
        bankroll = 1000.0
        initial_bankroll = bankroll
        peak_bankroll = bankroll
        max_drawdown = 0.0
        daily_returns = {}

        for sig in signals:
            # 确定仓位
            if strategy == "kelly":
                bet_size = self._kelly_size(sig, bankroll)
            elif strategy == "fixed":
                bet_size = bankroll * 0.02  # 固定2%
            elif strategy == "threshold":
                bet_size = self._threshold_size(sig, bankroll)
            else:
                bet_size = bankroll * 0.02

            if bet_size <= 0:
                continue

            # 计算盈亏
            buy_price = sig["poly_price"]
            won = sig["actual_outcome"] == 1

            if won:
                pnl = bet_size * (1.0 - buy_price) / buy_price  # 赢的回报
            else:
                pnl = -bet_size  # 输掉本金

            bankroll += pnl
            peak_bankroll = max(peak_bankroll, bankroll)
            drawdown = (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)

            # 按日期归组
            date = sig.get("date", "unknown")
            if date not in daily_returns:
                daily_returns[date] = 0
            daily_returns[date] += pnl

            results.append({
                "game_id": sig.get("game_id", ""),
                "buy_team": sig.get("buy_team", ""),
                "buy_price": buy_price,
                "score": sig.get("score", 0),
                "edge": sig.get("effective_edge", 0),
                "bet_size": round(bet_size, 2),
                "won": won,
                "pnl": round(pnl, 2),
                "bankroll": round(bankroll, 2),
            })

        # 统计
        n_bets = len(results)
        if n_bets == 0:
            return {"status": "no_bets", "reason": "无可回测的投注"}

        wins = sum(1 for r in results if r["won"])
        total_pnl = bankroll - initial_bankroll
        roi = total_pnl / initial_bankroll

        # Sharpe ratio（日收益）
        daily_rets = list(daily_returns.values())
        if len(daily_rets) > 1:
            sharpe = (np.mean(daily_rets) / np.std(daily_rets)) * np.sqrt(252) if np.std(daily_rets) > 0 else 0
        else:
            sharpe = 0

        # 按评分分桶
        score_buckets = self._analyze_by_score(results)
        edge_buckets = self._analyze_by_edge(results)

        # 校准分析
        calibration = self._calibration_analysis(signals)

        return {
            "status": "ok",
            "strategy": strategy,
            "n_bets": n_bets,
            "wins": wins,
            "losses": n_bets - wins,
            "win_rate": round(wins / n_bets, 4),
            "total_pnl": round(total_pnl, 2),
            "roi": round(roi, 4),
            "final_bankroll": round(bankroll, 2),
            "max_drawdown": round(max_drawdown, 4),
            "sharpe_ratio": round(sharpe, 2),
            "score_buckets": score_buckets,
            "edge_buckets": edge_buckets,
            "calibration": calibration,
            "bets": results,
        }

    def compare_strategies(self) -> dict:
        """对比多种策略的表现。"""
        strategies = ["kelly", "fixed", "threshold"]
        comparison = {}
        for s in strategies:
            result = self.run_backtest(strategy=s)
            if result["status"] == "ok":
                comparison[s] = {
                    "roi": result["roi"],
                    "win_rate": result["win_rate"],
                    "sharpe": result["sharpe_ratio"],
                    "max_drawdown": result["max_drawdown"],
                    "n_bets": result["n_bets"],
                    "total_pnl": result["total_pnl"],
                }
        return comparison

    def walk_forward_test(self, n_folds: int = 5) -> dict:
        """Walk-forward 回测（防数据泄漏）。

        将数据分为 n_folds 个时间窗口，每个窗口用前面的数据训练，后面的测试。
        """
        signals = self._load_resolved_signals()
        if len(signals) < n_folds * 5:
            return {"status": "insufficient_data", "min_needed": n_folds * 5}

        fold_size = len(signals) // n_folds
        fold_results = []

        for i in range(1, n_folds):
            train_end = i * fold_size
            test_start = train_end
            test_end = min((i + 1) * fold_size, len(signals))

            train_data = signals[:train_end]
            test_data = signals[test_start:test_end]

            if not test_data:
                continue

            # 用训练集计算最优阈值
            best_threshold = self._optimize_threshold(train_data)

            # 在测试集上评估
            wins = 0
            total = 0
            pnl = 0.0
            for sig in test_data:
                if sig.get("score", 0) >= best_threshold:
                    total += 1
                    won = sig["actual_outcome"] == 1
                    if won:
                        wins += 1
                        pnl += (1 - sig["poly_price"]) / sig["poly_price"] * 20
                    else:
                        pnl -= 20

            fold_results.append({
                "fold": i,
                "train_size": len(train_data),
                "test_size": len(test_data),
                "threshold": best_threshold,
                "bets": total,
                "wins": wins,
                "win_rate": round(wins / total, 4) if total > 0 else 0,
                "pnl": round(pnl, 2),
            })

        return {
            "status": "ok",
            "n_folds": n_folds,
            "folds": fold_results,
            "avg_win_rate": round(
                np.mean([f["win_rate"] for f in fold_results if f["bets"] > 0]), 4
            ) if fold_results else 0,
            "avg_pnl": round(
                np.mean([f["pnl"] for f in fold_results]), 2
            ) if fold_results else 0,
        }

    def print_report(self, strategy: str = "kelly"):
        """打印格式化的回测报告。"""
        result = self.run_backtest(strategy)

        if result["status"] != "ok":
            print(f"\n⚠️ 回测失败: {result.get('reason', 'unknown')}")
            return

        print(f"\n━━━ 投注策略回测报告 ({strategy.upper()}) ━━━")
        print(f"  总投注:     {result['n_bets']}笔")
        print(f"  胜/负:      {result['wins']}/{result['losses']}")
        print(f"  胜率:       {result['win_rate']:.1%}")
        print(f"  总PnL:      ${result['total_pnl']:+.2f}")
        print(f"  ROI:        {result['roi']:+.1%}")
        print(f"  最终资金:   ${result['final_bankroll']:.2f} (初始$1000)")
        print(f"  最大回撤:   {result['max_drawdown']:.1%}")
        print(f"  Sharpe:     {result['sharpe_ratio']:.2f}")

        print("\n  📊 按评分分桶:")
        for b in result["score_buckets"]:
            print(f"    {b['bucket']:12s}  {b['count']:3d}笔  "
                  f"胜率:{b['win_rate']:.0%}  ROI:{b['avg_roi']:+.1%}")

        print("\n  📈 按Edge分桶:")
        for b in result["edge_buckets"]:
            print(f"    {b['bucket']:12s}  {b['count']:3d}笔  "
                  f"胜率:{b['win_rate']:.0%}  ROI:{b['avg_roi']:+.1%}")

        if result.get("calibration"):
            print("\n  🎯 校准分析:")
            for c in result["calibration"]:
                print(f"    预测{c['bin']:5s}  实际:{c['actual']:.0%}  "
                      f"({c['count']}笔)  {'✅' if abs(c['predicted_avg'] - c['actual']) < 0.05 else '⚠️'}")

        # 多策略对比
        print("\n  🔄 策略对比:")
        comparison = self.compare_strategies()
        for name, stats in comparison.items():
            marker = " ← 当前" if name == strategy else ""
            print(f"    {name:12s}  ROI:{stats['roi']:+.1%}  "
                  f"胜率:{stats['win_rate']:.0%}  Sharpe:{stats['sharpe']:.2f}{marker}")

    # ── 内部方法 ──────────────────────────────────────────────

    def _load_resolved_signals(self) -> list:
        """加载已结算的信号数据。"""
        con = sqlite3.connect(self.db_path, timeout=30)
        con.row_factory = sqlite3.Row
        try:
            rows = con.execute("""
                SELECT game_id, buy_side, buy_team, poly_price, pinnacle_prob,
                       raw_edge, effective_edge, score, kelly, was_pushed,
                       actual_outcome, prediction_correct, scanned_at,
                       breakdown_json
                FROM signal_log
                WHERE actual_outcome IS NOT NULL
                ORDER BY scanned_at ASC
            """).fetchall()
        except Exception as e:
            logger.warning(f"[Backtest] 查询失败: {e}")
            con.close()
            return []

        signals = []
        for row in rows:
            r = dict(row)
            r["date"] = str(r.get("scanned_at", ""))[:10]
            # 解析 breakdown
            try:
                bd = json.loads(r.get("breakdown_json") or "{}")
                r["model_prob"] = bd.get("model_confidence", {}).get("model_prob", 0.5)
            except Exception:
                r["model_prob"] = 0.5
            signals.append(r)

        con.close()
        return signals

    def _kelly_size(self, sig: dict, bankroll: float) -> float:
        """Kelly 仓位计算。"""
        buy_price = sig.get("poly_price", 0.5)
        edge = sig.get("effective_edge", 0)
        score = sig.get("score", 0)

        if edge <= 0 or buy_price <= 0 or buy_price >= 1 or score < 35:
            return 0

        b = (1 - buy_price) / buy_price  # 赔率
        p = buy_price + edge  # 估算胜率
        p = min(0.95, max(0.05, p))

        kelly = p - (1 - p) / b
        # 1/4 Kelly
        kelly = max(0, kelly * 0.25)
        return min(bankroll * kelly, bankroll * 0.05)  # 最多5%

    def _threshold_size(self, sig: dict, bankroll: float) -> float:
        """基于评分阈值的仓位。评分越高 → 仓位越大。"""
        score = sig.get("score", 0)
        if score < 40:
            return 0
        elif score >= 70:
            return bankroll * 0.04
        elif score >= 55:
            return bankroll * 0.03
        else:
            return bankroll * 0.02

    def _optimize_threshold(self, signals: list) -> int:
        """在训练集上找最优评分阈值。"""
        best_threshold = 35
        best_metric = -999

        for threshold in range(30, 75, 5):
            filtered = [s for s in signals if s.get("score", 0) >= threshold]
            if len(filtered) < 3:
                continue
            wins = sum(1 for s in filtered if s["actual_outcome"] == 1)
            wr = wins / len(filtered)
            # 优化指标: 胜率 * log(投注数+1)，平衡质量和数量
            metric = wr * np.log1p(len(filtered))
            if metric > best_metric:
                best_metric = metric
                best_threshold = threshold

        return best_threshold

    def _analyze_by_score(self, results: list) -> list:
        """按评分分桶分析。"""
        buckets = [
            ("30-44", 30, 45),
            ("45-54", 45, 55),
            ("55-64", 55, 65),
            ("65-74", 65, 75),
            ("75+", 75, 200),
        ]
        output = []
        for label, lo, hi in buckets:
            bets = [r for r in results if lo <= r.get("score", 0) < hi]
            if not bets:
                continue
            wins = sum(1 for b in bets if b["won"])
            total_pnl = sum(b["pnl"] for b in bets)
            total_bet = sum(b["bet_size"] for b in bets)
            output.append({
                "bucket": label,
                "count": len(bets),
                "wins": wins,
                "win_rate": wins / len(bets),
                "avg_roi": total_pnl / total_bet if total_bet > 0 else 0,
            })
        return output

    def _analyze_by_edge(self, results: list) -> list:
        """按edge分桶分析。"""
        buckets = [
            ("0-1%", 0, 0.01),
            ("1-2%", 0.01, 0.02),
            ("2-3%", 0.02, 0.03),
            ("3-5%", 0.03, 0.05),
            ("5%+", 0.05, 1.0),
        ]
        output = []
        for label, lo, hi in buckets:
            bets = [r for r in results if lo <= abs(r.get("edge", 0)) < hi]
            if not bets:
                continue
            wins = sum(1 for b in bets if b["won"])
            total_pnl = sum(b["pnl"] for b in bets)
            total_bet = sum(b["bet_size"] for b in bets)
            output.append({
                "bucket": label,
                "count": len(bets),
                "wins": wins,
                "win_rate": wins / len(bets),
                "avg_roi": total_pnl / total_bet if total_bet > 0 else 0,
            })
        return output

    def _calibration_analysis(self, signals: list) -> list:
        """校准分析：预测概率 vs 实际胜率。"""
        bins = [
            ("40-50%", 0.40, 0.50),
            ("50-55%", 0.50, 0.55),
            ("55-60%", 0.55, 0.60),
            ("60-65%", 0.60, 0.65),
            ("65-70%", 0.65, 0.70),
            ("70%+", 0.70, 1.00),
        ]
        output = []
        for label, lo, hi in bins:
            group = [s for s in signals if lo <= s.get("model_prob", 0.5) < hi]
            if not group:
                continue
            actual_wins = sum(1 for s in group if s["actual_outcome"] == 1)
            output.append({
                "bin": label,
                "count": len(group),
                "predicted_avg": round(np.mean([s.get("model_prob", 0.5) for s in group]), 4),
                "actual": round(actual_wins / len(group), 4),
            })
        return output
