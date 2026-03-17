"""
notion_sync.py — Notion自动同步模块

每次推送信号 → 写入Notion数据库
每次结算结果 → 更新Notion对应行

数据库列：比赛、日期、买入方、买入价格、模型预测、评分、比分、结果、ROI、Kelly仓位、价值边际
"""

import logging
import requests

logger = logging.getLogger(__name__)

NOTION_API = "https://api.notion.com/v1"
NOTION_VERSION = "2022-06-28"


class NotionSync:
    def __init__(self, token: str, database_id: str):
        self.token = token
        self.db_id = database_id
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Notion-Version": NOTION_VERSION,
            "Content-Type": "application/json",
        }

    def add_push(self, game_id: str, matchup: str, date: str, buy_team: str,
                 buy_price: float, model_prob: float, score: int,
                 kelly: float, value_edge: float) -> str:
        """推送时写入一行。返回Notion page_id。"""
        try:
            data = {
                "parent": {"database_id": self.db_id},
                "properties": {
                    "比赛": {"title": [{"text": {"content": matchup}}]},
                    "日期": {"date": {"start": date}},
                    "买入方": {"rich_text": [{"text": {"content": buy_team}}]},
                    "买入价格": {"number": round(buy_price, 4)},
                    "模型预测": {"number": round(model_prob, 4)},
                    "评分": {"number": score},
                    "结果": {"select": {"name": "⏳待结算"}},
                    "Kelly仓位": {"number": round(kelly, 4)},
                    "价值边际": {"number": round(value_edge, 4)},
                },
            }
            resp = requests.post(
                f"{NOTION_API}/pages",
                headers=self.headers,
                json=data,
                timeout=15,
            )
            resp.raise_for_status()
            page_id = resp.json()["id"]
            logger.info(f"[Notion] 写入推送: {matchup} → {page_id}")
            return page_id
        except Exception as e:
            logger.warning(f"[Notion] 写入失败: {e}")
            return ""

    def update_result(self, game_id: str, matchup: str, score_str: str,
                      prediction_correct: int, actual_roi: float):
        """结算时更新对应行。"""
        try:
            # 搜索匹配的行
            resp = requests.post(
                f"{NOTION_API}/databases/{self.db_id}/query",
                headers=self.headers,
                json={
                    "filter": {
                        "property": "比赛",
                        "title": {"equals": matchup},
                    },
                    "page_size": 1,
                },
                timeout=15,
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            if not results:
                logger.debug(f"[Notion] 未找到 {matchup} 的记录")
                return

            page_id = results[0]["id"]
            result_label = "✅赢" if prediction_correct else "❌输"

            requests.patch(
                f"{NOTION_API}/pages/{page_id}",
                headers=self.headers,
                json={
                    "properties": {
                        "比分": {"rich_text": [{"text": {"content": score_str}}]},
                        "结果": {"select": {"name": result_label}},
                        "ROI": {"number": round(actual_roi, 4)},
                    },
                },
                timeout=15,
            )
            logger.info(f"[Notion] 更新结果: {matchup} → {result_label} ROI{actual_roi:+.0%}")

        except Exception as e:
            logger.warning(f"[Notion] 更新失败: {e}")

    def sync_historical(self, db):
        """同步历史推送数据到Notion。"""
        rows = db.execute("""
            SELECT game_id, away_team, home_team, buy_team, buy_side,
                   poly_price, pinnacle_prob, score, kelly, effective_edge,
                   prediction_correct, actual_roi, actual_score_away, actual_score_home,
                   scanned_at
            FROM signal_log
            WHERE was_pushed=1
            ORDER BY scanned_at DESC
            LIMIT 50
        """)

        # 去重：同一场比赛只同步一条（最高评分）
        seen = {}
        for r in rows:
            d = dict(r)
            gid = d["game_id"]
            if gid not in seen or d["score"] > seen[gid]["score"]:
                seen[gid] = d

        count = 0
        for gid, d in seen.items():
            matchup = f"{d['away_team']}@{d['home_team']}"
            date = (d["scanned_at"] or "")[:10]
            buy_team = d.get("buy_team") or d.get("buy_side", "?")

            page_id = self.add_push(
                game_id=gid,
                matchup=matchup,
                date=date,
                buy_team=buy_team,
                buy_price=d.get("poly_price", 0) or 0,
                model_prob=d.get("pinnacle_prob", 0) or 0,
                score=d.get("score", 0) or 0,
                kelly=d.get("kelly", 0) or 0,
                value_edge=d.get("effective_edge", 0) or 0,
            )

            if page_id and d.get("prediction_correct") is not None:
                score_str = f"{d.get('actual_score_away', '?')}-{d.get('actual_score_home', '?')}"
                self.update_result(
                    game_id=gid,
                    matchup=matchup,
                    score_str=score_str,
                    prediction_correct=d["prediction_correct"],
                    actual_roi=d.get("actual_roi", 0) or 0,
                )

            count += 1

        return count
