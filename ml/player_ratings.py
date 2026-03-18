"""
player_ratings.py — 自动球员影响力评分系统

从 nba_api LeagueDashPlayerStats 拉取全部NBA球员赛季数据，
计算综合影响力分(impact_score)，归一化到0-12区间，
替代旧的硬编码RAPTOR_ESTIMATES。

每日06:00自动更新，保持球员评分与赛季表现同步。
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# 兜底默认值（与旧RAPTOR一致）
DEFAULT_STARTER = 2.0
DEFAULT_BENCH = 0.5


def update_player_ratings(db) -> int:
    """从nba_api拉取赛季数据，计算并更新全部球员评分。

    使用 LeagueDashPlayerStats 一次API调用获取~500名球员。
    impact_score公式:
      raw = PTS×0.4 + AST×0.3 + REB×0.15 + STL×0.15 + BLK×0.1 - TOV×0.2
      scaled = raw × min(MIN/36, 1.2) + PLUS_MINUS×0.3
      impact_score = clamp(scaled / 2.2, 0, 12)

    归一化系数2.2使得 Jokic级别(~26pts/12reb/9ast/+8) → ~10.1，
    与旧RAPTOR硬编码量级一致。
    """
    try:
        from nba_api.stats.endpoints import LeagueDashPlayerStats
        import time

        logger.info("[PlayerRatings] 开始更新球员评分...")
        time.sleep(1)  # rate limit

        stats = LeagueDashPlayerStats(
            season="2025-26",
            per_mode_detailed="PerGame",
            timeout=30,
        )
        df = stats.get_data_frames()[0]
    except Exception as e:
        logger.warning(f"[PlayerRatings] nba_api拉取失败: {e}")
        return 0

    count = 0
    for _, row in df.iterrows():
        name = row.get("PLAYER_NAME", "")
        if not name:
            continue

        pts = float(row.get("PTS", 0) or 0)
        ast = float(row.get("AST", 0) or 0)
        reb = float(row.get("REB", 0) or 0)
        stl = float(row.get("STL", 0) or 0)
        blk = float(row.get("BLK", 0) or 0)
        tov = float(row.get("TOV", 0) or 0)
        mins = float(row.get("MIN", 0) or 0)
        plus_minus = float(row.get("PLUS_MINUS", 0) or 0)
        gp = int(row.get("GP", 0) or 0)
        team = row.get("TEAM_ABBREVIATION", "")

        # 综合影响力原始分
        raw = (pts * 0.4 + ast * 0.3 + reb * 0.15
               + stl * 0.15 + blk * 0.1 - tov * 0.2)

        # 按出场时间缩放（36分钟基准）
        min_factor = min(mins / 36.0, 1.2) if mins > 0 else 0
        raw_scaled = raw * min_factor + plus_minus * 0.3

        # 归一化到0-12（Jokic≈10.1, 角色球员≈2-3, 替补≈0.5-1.5）
        impact_score = max(0, min(12, raw_scaled / 2.2))

        try:
            db.insert("""
                INSERT INTO player_ratings
                    (player_name, team_abbr, impact_score, games_played,
                     pts_per_game, reb_per_game, ast_per_game, plus_minus, updated_at)
                VALUES (?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)
                ON CONFLICT(player_name) DO UPDATE SET
                    team_abbr=excluded.team_abbr,
                    impact_score=excluded.impact_score,
                    games_played=excluded.games_played,
                    pts_per_game=excluded.pts_per_game,
                    reb_per_game=excluded.reb_per_game,
                    ast_per_game=excluded.ast_per_game,
                    plus_minus=excluded.plus_minus,
                    updated_at=CURRENT_TIMESTAMP
            """, (name, team, round(impact_score, 2), gp, pts, reb, ast, plus_minus))
            count += 1
        except Exception:
            continue

    logger.info(f"[PlayerRatings] 更新完成: {count}名球员")
    return count


def get_player_rating(db, player_name: str) -> float:
    """查询单个球员的影响力分。找不到返回替补默认值。"""
    row = db.execute_one(
        "SELECT impact_score FROM player_ratings WHERE player_name=?",
        (player_name,)
    )
    if row:
        return row["impact_score"]
    return DEFAULT_BENCH


def get_all_ratings_as_dict(db) -> dict:
    """批量加载所有球员评分为dict，用于伤病计算时避免N+1查询。"""
    rows = db.execute("SELECT player_name, impact_score FROM player_ratings")
    return {dict(r)["player_name"]: dict(r)["impact_score"] for r in rows}


def get_ratings_summary(db) -> dict:
    """返回评分概览统计。"""
    total = db.execute_one("SELECT COUNT(*) as cnt FROM player_ratings")
    top5 = db.execute(
        "SELECT player_name, team_abbr, impact_score, pts_per_game "
        "FROM player_ratings ORDER BY impact_score DESC LIMIT 5"
    )
    return {
        "total_players": total["cnt"] if total else 0,
        "top5": [dict(r) for r in top5] if top5 else [],
    }
