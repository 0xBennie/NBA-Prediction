"""
price_watcher.py — 价格监控模块（低价买入检测）

核心逻辑:
  1. 比赛一开盘 → 模型预测谁赢 → 计算目标买入价
  2. 持续监控价格 → 价格跌到目标以下 → 立刻推送
  3. 追踪价格历史 → 找到真正的低位买入机会

示例:
  模型预测ATL 71%会赢 → 目标买入价 = 71% × 0.70 = 50¢
  ATL当前59¢ → 不推送（太贵）
  ATL跌到40¢ → 推送！"ATL跌到低位40¢，模型认为值71%，买入空间+78%"
"""

import json
import logging
import time
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class PriceWatcher:
    """价格监控 — 检测低价买入机会。"""

    def __init__(self, db):
        self.db = db
        self._ensure_table()

    def _ensure_table(self):
        try:
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS price_watchlist (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id         TEXT UNIQUE NOT NULL,
                    away_team       TEXT NOT NULL,
                    home_team       TEXT NOT NULL,
                    game_date       TEXT,
                    buy_side        TEXT,
                    buy_team        TEXT,
                    model_prob      REAL,
                    target_price    REAL,
                    current_price   REAL,
                    lowest_price    REAL,
                    highest_price   REAL,
                    alert_sent      INTEGER DEFAULT 0,
                    slug            TEXT,
                    token_id        TEXT,
                    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at      DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
        except Exception:
            pass

    def update_watchlist(self, game: dict, model_prob: float, buy_side: str,
                         buy_price: float):
        """更新监控名单 — 每次扫描时调用。

        如果比赛不在名单里 → 新增
        如果已在名单里 → 更新价格
        """
        gid = game["game_id"]
        buy_team = game["away_team"] if buy_side == "away" else game["home_team"]

        # 计算目标买入价: 模型概率 × 折扣系数
        # 例: 71% × 0.70 = 50¢ → 低于50¢才值得买
        discount = 0.70  # 要求至少30%折扣
        target_price = round(model_prob * discount, 4)

        # token_id
        clob_tokens = game.get("clob_token_ids", [])
        token_id = ""
        if clob_tokens and len(clob_tokens) >= 2:
            token_id = clob_tokens[0] if buy_side == "away" else clob_tokens[1]

        existing = self.db.execute_one(
            "SELECT * FROM price_watchlist WHERE game_id=?", (gid,)
        )

        if existing:
            e = dict(existing)
            lowest = min(e.get("lowest_price", 1.0) or 1.0, buy_price)
            highest = max(e.get("highest_price", 0) or 0, buy_price)
            self.db.insert("""
                UPDATE price_watchlist
                SET current_price=?, lowest_price=?, highest_price=?,
                    model_prob=?, target_price=?, updated_at=CURRENT_TIMESTAMP
                WHERE game_id=?
            """, (buy_price, lowest, highest, model_prob, target_price, gid))
        else:
            self.db.insert("""
                INSERT OR IGNORE INTO price_watchlist
                    (game_id, away_team, home_team, game_date, buy_side, buy_team,
                     model_prob, target_price, current_price, lowest_price, highest_price,
                     slug, token_id)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                gid, game["away_team"], game["home_team"],
                game.get("game_date", ""),
                buy_side, buy_team,
                model_prob, target_price, buy_price, buy_price, buy_price,
                game.get("slug", ""), token_id,
            ))
            logger.info(
                f"[Watcher] 新增监控: {game['away_team']}@{game['home_team']} "
                f"买{buy_team} 目标价{target_price:.0%} 当前{buy_price:.0%}"
            )

    def check_opportunities(self) -> list:
        """检查是否有价格跌到目标以下的机会。

        排除已开赛/已结束的比赛（game_date必须是今天或未来）。
        """
        rows = self.db.execute("""
            SELECT * FROM price_watchlist
            WHERE alert_sent=0
              AND current_price <= target_price
              AND current_price > 0.10
              AND game_date >= date('now')
            ORDER BY (model_prob - current_price) DESC
        """)

        alerts = []
        for r in rows:
            d = dict(r)
            value_edge = d["model_prob"] - d["current_price"]
            potential_roi = (1.0 / d["current_price"] - 1.0) if d["current_price"] > 0 else 0

            alerts.append({
                "game_id": d["game_id"],
                "away_team": d["away_team"],
                "home_team": d["home_team"],
                "buy_team": d["buy_team"],
                "buy_side": d["buy_side"],
                "model_prob": d["model_prob"],
                "target_price": d["target_price"],
                "current_price": d["current_price"],
                "lowest_price": d["lowest_price"],
                "highest_price": d["highest_price"],
                "value_edge": value_edge,
                "potential_roi": potential_roi,
                "slug": d.get("slug", ""),
                "token_id": d.get("token_id", ""),
                "game_date": d.get("game_date", ""),
            })

        return alerts

    def mark_alerted(self, game_id: str):
        """标记已发送警报。"""
        self.db.insert(
            "UPDATE price_watchlist SET alert_sent=1 WHERE game_id=?",
            (game_id,)
        )

    def check_price_drops(self) -> list:
        """检查价格大幅回调的机会（不一定低于目标价，但跌了很多）。

        逻辑: 从最高点跌了10%+ → 可能是好买点
        """
        rows = self.db.execute("""
            SELECT * FROM price_watchlist
            WHERE alert_sent=0
              AND highest_price > 0
              AND current_price < highest_price * 0.90
              AND current_price <= target_price * 1.10
            ORDER BY (highest_price - current_price) DESC
        """)

        drops = []
        for r in rows:
            d = dict(r)
            drop_pct = (d["highest_price"] - d["current_price"]) / d["highest_price"]
            drops.append({
                **d,
                "drop_pct": drop_pct,
                "value_edge": d["model_prob"] - d["current_price"],
            })

        return drops

    def get_watchlist_summary(self) -> str:
        """获取监控名单摘要。"""
        rows = self.db.execute("""
            SELECT buy_team, away_team, home_team, model_prob, target_price,
                   current_price, lowest_price, highest_price, alert_sent, game_date
            FROM price_watchlist
            ORDER BY (model_prob - current_price) DESC
            LIMIT 15
        """)

        if not rows:
            return "监控名单为空"

        lines = ["📋 <b>价格监控名单</b>\n"]
        for r in rows:
            d = dict(r)
            edge = d["model_prob"] - d["current_price"]
            status = "🟢" if d["current_price"] <= d["target_price"] else "⚪"
            if d["alert_sent"]:
                status = "✅"
            lines.append(
                f"{status} {d['away_team']}@{d['home_team']} "
                f"买{d['buy_team']} {d['current_price']:.0%}"
                f"(目标{d['target_price']:.0%}) "
                f"模型{d['model_prob']:.0%} "
                f"低{d['lowest_price']:.0%}/高{d['highest_price']:.0%}"
            )

        return "\n".join(lines)

    def cleanup_old(self):
        """清理过期比赛。"""
        self.db.insert("""
            DELETE FROM price_watchlist
            WHERE game_date < date('now', '-1 day')
        """)
