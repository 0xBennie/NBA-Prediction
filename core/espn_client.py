"""
espn_client.py — ESPN 数据获取
功能：战绩、伤病报告、赛程(B2B判断)、球员阵容
"""

import time
import logging
import requests
from datetime import datetime, timedelta, date
from typing import Optional
from core.database import Database

logger = logging.getLogger(__name__)

# 球员RAPTOR近似值（正数越大越重要，需定期更新）
RAPTOR_ESTIMATES = {
    "Nikola Jokic": 10.1, "Shai Gilgeous-Alexander": 8.3,
    "LeBron James": 9.5, "Stephen Curry": 9.2,
    "Giannis Antetokounmpo": 9.0, "Luka Doncic": 9.3,
    "Kevin Durant": 8.8, "Joel Embiid": 8.5,
    "Jayson Tatum": 7.8, "Damian Lillard": 6.5,
    "Devin Booker": 6.8, "Zach LaVine": 5.9,
    "Bam Adebayo": 5.5, "Draymond Green": 5.2,
    "Khris Middleton": 5.0, "Anthony Davis": 7.2,
    "Kawhi Leonard": 7.0, "Paul George": 5.8,
    "Donovan Mitchell": 6.2, "Trae Young": 6.0,
    "Karl-Anthony Towns": 5.8, "Ja Morant": 6.5,
    "Tyrese Haliburton": 6.0, "Alperen Sengun": 5.5,
    "_starter": 2.5,   # 普通首发默认值
    "_bench": 0.8,     # 替补默认值
}

STATUS_MULTIPLIER = {
    "Out": 1.0,
    "Doubtful": 0.75,
    "Questionable": 0.40,
    "Day-To-Day": 0.25,
    "Probable": 0.10,
}

ESPN_TEAM_IDS = {
    "ATL": "1", "BOS": "2", "BKN": "17", "CHA": "30", "CHI": "4",
    "CLE": "5", "DAL": "6", "DEN": "7", "DET": "8", "GSW": "9",
    "HOU": "10", "IND": "11", "LAC": "12", "LAL": "13", "MEM": "29",
    "MIA": "14", "MIL": "15", "MIN": "16", "NOP": "3", "NYK": "18",
    "OKC": "25", "ORL": "19", "PHI": "20", "PHX": "21", "POR": "22",
    "SAC": "23", "SAS": "24", "TOR": "28", "UTA": "26", "WAS": "27",
}


class ESPNClient:

    STANDINGS_TTL = 1800   # 30分钟
    INJURY_TTL = 3600      # 1小时
    SCHEDULE_TTL = 86400   # 24小时

    def __init__(self, db: Database):
        self.db = db
        self._sched_cache: dict = {}
        self._sched_ts: dict = {}

    # ── 战绩 ──────────────────────────────────────────────────────
    def get_standings(self, team_abbr: str) -> Optional[dict]:
        """获取球队战绩，带DB缓存"""
        row = self.db.execute_one(
            "SELECT *, strftime('%s', updated_at) as ts FROM standings WHERE team_abbr=?",
            (team_abbr,)
        )
        if row and (time.time() - float(row["ts"])) < self.STANDINGS_TTL:
            return row

        try:
            r = requests.get(
                "https://site.api.espn.com/apis/v2/sports/basketball/nba/standings",
                timeout=10,
            )
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            logger.warning(f"[ESPN] standings error: {e}")
            return row  # 返回旧缓存

        self._parse_and_cache_standings(data)
        return self.db.execute_one(
            "SELECT * FROM standings WHERE team_abbr=?", (team_abbr,)
        )

    def _parse_and_cache_standings(self, data: dict):
        for group in data.get("children", []):
            for entry in group.get("standings", {}).get("entries", []):
                team = entry.get("team", {})
                abbr = team.get("abbreviation", "")
                if not abbr:
                    continue

                stats = {s["name"]: s.get("value", 0)
                         for s in entry.get("stats", [])}
                wins = int(stats.get("wins", 0))
                losses = int(stats.get("losses", 0))
                gp = wins + losses
                win_rate = wins / gp if gp > 0 else 0
                ppg = stats.get("pointsFor", 0) / gp if gp > 0 else 0
                opp = stats.get("pointsAgainst", 0) / gp if gp > 0 else 0

                self.db.insert("""
                    INSERT INTO standings
                        (team_abbr, wins, losses, win_rate, ppg, opp_ppg, ppg_diff, games_played, updated_at)
                    VALUES (?,?,?,?,?,?,?,?, CURRENT_TIMESTAMP)
                    ON CONFLICT(team_abbr) DO UPDATE SET
                        wins=excluded.wins, losses=excluded.losses,
                        win_rate=excluded.win_rate, ppg=excluded.ppg,
                        opp_ppg=excluded.opp_ppg, ppg_diff=excluded.ppg_diff,
                        games_played=excluded.games_played,
                        updated_at=CURRENT_TIMESTAMP
                """, (abbr, wins, losses, win_rate, ppg, opp, ppg - opp, gp))

    # ── 伤病 ──────────────────────────────────────────────────────
    def get_injury_impact(self, team_abbr: str) -> float:
        """返回伤病Elo惩罚值（越大=伤病越严重）"""
        team_id = ESPN_TEAM_IDS.get(team_abbr.upper())
        if not team_id:
            return 0.0

        # 检查缓存时效
        rows = self.db.execute(
            """SELECT *, strftime('%s', updated_at) as ts
               FROM injuries WHERE team_abbr=? LIMIT 1""",
            (team_abbr,)
        )
        if rows and (time.time() - float(dict(rows[0])["ts"])) < self.INJURY_TTL:
            total = sum(dict(r)["impact"] for r in rows)
            return total

        # 重新拉取
        try:
            r = requests.get(
                f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team_id}/injuries",
                timeout=10,
            )
            r.raise_for_status()
            injuries = r.json().get("injuries", [])
        except Exception as e:
            logger.warning(f"[ESPN] injuries {team_abbr}: {e}")
            return 0.0

        # 清旧数据
        self.db.execute("DELETE FROM injuries WHERE team_abbr=?", (team_abbr,))

        total_impact = 0.0
        for inj in injuries:
            athlete = inj.get("athlete", {})
            name = athlete.get("displayName", "")
            status = inj.get("status", "Out")
            raptor = RAPTOR_ESTIMATES.get(name, RAPTOR_ESTIMATES["_starter"])
            mult = STATUS_MULTIPLIER.get(status, 1.0)
            impact = raptor * mult * 12  # 转换为Elo惩罚点数

            self.db.insert("""
                INSERT INTO injuries (team_abbr, player_name, status, impact, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (team_abbr, name, status, impact))
            total_impact += impact

        return total_impact

    # ── 背靠背判断 ────────────────────────────────────────────────
    def is_back_to_back(self, team_abbr: str, game_date: str) -> bool:
        """
        判断球队在 game_date 是否是背靠背第二场。
        game_date: "2026-03-14"
        """
        ck = f"{team_abbr}_{game_date[:7]}"  # 按月缓存
        if ck in self._sched_cache and time.time() - self._sched_ts.get(ck, 0) < self.SCHEDULE_TTL:
            schedule = self._sched_cache[ck]
        else:
            schedule = self._fetch_schedule(team_abbr)
            self._sched_cache[ck] = schedule
            self._sched_ts[ck] = time.time()

        target = datetime.strptime(game_date, "%Y-%m-%d").date()
        prev = target - timedelta(days=1)
        return prev in schedule

    def _fetch_schedule(self, team_abbr: str) -> set:
        team_id = ESPN_TEAM_IDS.get(team_abbr.upper())
        if not team_id:
            return set()
        try:
            r = requests.get(
                f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team_id}/schedule",
                timeout=10,
            )
            r.raise_for_status()
            events = r.json().get("events", [])
            dates = set()
            for e in events:
                ds = e.get("date", "")[:10]
                try:
                    dates.add(datetime.strptime(ds, "%Y-%m-%d").date())
                except Exception:
                    pass
            return dates
        except Exception as e:
            logger.warning(f"[ESPN] schedule {team_abbr}: {e}")
            return set()
