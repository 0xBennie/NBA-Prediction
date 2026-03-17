"""
result_resolver.py — 比赛结果自动回填
每天定时运行，查询未结算推送 → 对比实际结果 → 写入push_results → 触发ML重训练

数据源: The Odds API /scores 端点（免费，不消耗配额）
用法: python -m ml.result_resolver
      python -m ml.result_resolver --report    # 只看报告不回填
"""

import os
import logging
import time
import argparse
from datetime import datetime, timedelta

from core.database import Database
from core.sportsbook_client import SportsbookClient, get_nba_scores, _match
from ml.ml_predictor import MLPredictor

logger = logging.getLogger(__name__)


class ResultResolver:

    def __init__(self, db: Database, ml: MLPredictor, odds_api_key: str = ""):
        self.db = db
        self.ml = ml
        self.odds_api_key = odds_api_key or os.getenv("ODDS_API_KEY", "")

    def resolve_all(self) -> dict:
        """查询所有未结算推送并尝试回填"""
        pending = self.db.execute("""
            SELECT push_id, game_id, away_team, home_team,
                   poly_price_at_push, market_type
            FROM push_results
            WHERE actual_outcome IS NULL
              AND created_at < datetime('now', '-3 hours')
        """)

        # 预加载最近3天的比赛结果（免费端点，一次调用）
        scores = []
        if self.odds_api_key:
            scores = get_nba_scores(self.odds_api_key, days_from=3)
            logger.info(f"[Resolver] 获取到 {len(scores)} 场比赛结果")

        resolved = 0
        failed = 0

        for row in pending:
            r = dict(row)
            if r["market_type"] == "game":
                success = self._resolve_game(r, scores)
            else:
                success = self._resolve_futures(r)

            if success:
                resolved += 1
            else:
                failed += 1

        if resolved > 0:
            logger.info(f"[Resolver] 回填完成: {resolved}条，失败: {failed}条")
            self.ml.maybe_retrain()

        return {"resolved": resolved, "failed": failed, "pending": len(pending)}

    def _resolve_game(self, row: dict, scores: list) -> bool:
        """通过 The Odds API /scores（免费）查询比赛结果"""
        try:
            away = row["away_team"]
            home = row["home_team"]

            for s in scores:
                if not s.get("completed"):
                    continue
                if not (_match(s["home_team"], home) and _match(s["away_team"], away)):
                    continue

                home_score = s.get("home_score")
                away_score = s.get("away_score")
                if home_score is None or away_score is None:
                    continue

                away_won = away_score > home_score

                # 查push_id对应的buy_side（从signal_log或breakdown推断）
                buy_side = "away"  # 默认
                sig = self.db.execute_one(
                    "SELECT buy_side FROM signal_log WHERE game_id=? AND was_pushed=1 LIMIT 1",
                    (row["game_id"],)
                )
                if sig and sig.get("buy_side"):
                    buy_side = sig["buy_side"]

                # 判断我们推的方向对不对
                if buy_side == "away":
                    prediction_correct = 1 if away_won else 0
                else:
                    prediction_correct = 1 if not away_won else 0

                price = row["poly_price_at_push"] or 0.5
                if prediction_correct == 1 and price > 0:
                    pnl = (1.0 / price) - 1.0 - 0.01  # 扣费
                else:
                    pnl = -1.0

                self.ml.record_result(row["push_id"], prediction_correct, pnl)
                logger.info(
                    f"[Resolver] {away}@{home} {away_score}-{home_score} "
                    f"买{buy_side} → {'✅对' if prediction_correct else '❌错'} PnL:{pnl:+.2f}"
                )
                return True

            logger.debug(f"[Resolver] 比赛未找到或未完成: {away}@{home}")
            return False

        except Exception as e:
            logger.warning(f"[Resolver] 回填失败 {row.get('game_id')}: {e}")
            return False

    def _resolve_futures(self, row: dict) -> bool:
        """期货盘结果需要手动或通过Polymarket API确认"""
        # Polymarket API 查询结果
        try:
            import requests
            cid = row.get("game_id", "")
            if not cid:
                return False

            r = requests.get(
                f"https://gamma-api.polymarket.com/markets/{cid}",
                timeout=10,
            )
            data = r.json()
            resolved = data.get("resolved", False)
            if not resolved:
                return False

            # resolved 时 winning_outcome 字段会有值
            winner = data.get("outcomePrices", [])
            # 简化判断：如果我们买的token结算价是1则赢
            outcome = 1  # 需要根据实际API返回结构调整
            price = row["poly_price_at_push"]
            pnl = (1 / price - 1) if outcome == 1 else -1.0

            self.ml.record_result(row["push_id"], outcome, pnl)
            return True

        except Exception as e:
            logger.warning(f"[Resolver] futures回填失败: {e}")
            return False


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="store_true", help="只看ML报告")
    args = parser.parse_args()

    db = Database()
    ml = MLPredictor()

    if args.report:
        report = ml.get_performance_report()
        print("\n━━━ ML推送表现报告 ━━━")
        print(f"  已结算推送:    {report['total_resolved']}")
        print(f"  胜率:          {report['win_rate']:.1%}" if report['win_rate'] else "  胜率: N/A")
        print(f"  平均每单位PnL: {report['avg_pnl']:.4f}" if report['avg_pnl'] else "  平均PnL: N/A")
        print(f"  待结算:        {report['pending']}")
        ml_active = report['ml_active']
        samples_needed = report.get('samples_needed', 0)
        print(f"  ML启用:        {'✅' if ml_active else f'❌ 还需{samples_needed}条数据'}")
        if report["edge_breakdown"]:
            print("\n  Edge分组胜率:")
            for b in report["edge_breakdown"]:
                print(f"    {b['bucket']:12s} {b['cnt']:3d}场  胜率:{b['win_rate']:.1%}")
    else:
        resolver = ResultResolver(db, ml)
        result = resolver.resolve_all()
        print(f"\n[Resolver] 完成: 回填{result['resolved']}条，失败{result['failed']}条，待处理{result['pending']}条")
