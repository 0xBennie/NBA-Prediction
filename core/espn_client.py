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

# 球员RAPTOR兜底值（仅在player_ratings表为空时使用）
_RAPTOR_FALLBACK = {
    "Nikola Jokic": 10.1, "Shai Gilgeous-Alexander": 8.3,
    "LeBron James": 9.5, "Stephen Curry": 9.2,
    "Giannis Antetokounmpo": 9.0, "Luka Doncic": 9.3,
    "Kevin Durant": 8.8, "Joel Embiid": 8.5,
    "Jayson Tatum": 7.8, "Anthony Davis": 7.2,
    "Kawhi Leonard": 7.0, "Devin Booker": 6.8,
    "Damian Lillard": 6.5, "Ja Morant": 6.5,
    "Donovan Mitchell": 6.2, "Trae Young": 6.0,
    "Tyrese Haliburton": 6.0, "Anthony Edwards": 7.0,
    "Jaylen Brown": 6.5, "Kyrie Irving": 6.8,
    "Jalen Brunson": 6.5, "De'Aaron Fox": 6.0,
    "Victor Wembanyama": 7.5,
    "Zach LaVine": 5.9, "Karl-Anthony Towns": 5.8,
    "Paul George": 5.8, "Bam Adebayo": 5.5,
    "Alperen Sengun": 5.5, "Draymond Green": 5.2,
    "Khris Middleton": 5.0, "Franz Wagner": 5.8,
    "Jalen Williams": 5.5, "Chet Holmgren": 5.5,
    "Pascal Siakam": 5.2, "Tyrese Maxey": 5.8,
    "Fred VanVleet": 5.0, "Domantas Sabonis": 5.5,
    "Jarrett Allen": 4.8, "Myles Turner": 4.5,
    "Lauri Markkanen": 5.2, "Andrew Wiggins": 4.0,
    "Klay Thompson": 4.0, "Bradley Beal": 4.5,
    "Terry Rozier": 4.0, "D'Angelo Russell": 3.8,
    "Anfernee Simons": 4.5, "Shaedon Sharpe": 3.8,
    "Keegan Murray": 3.8, "Ivica Zubac": 3.5,
    "_starter": 2.0,
    "_bench": 0.5,
}
# 向后兼容：旧代码可能引用 RAPTOR_ESTIMATES
RAPTOR_ESTIMATES = _RAPTOR_FALLBACK

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


# ESPN team ID → 缩写 反向映射
_ESPN_ID_TO_ABBR = {v: k for k, v in ESPN_TEAM_IDS.items()}


class ESPNClient:

    STANDINGS_TTL = 1800   # 30分钟
    INJURY_TTL = 3600      # 1小时
    SCHEDULE_TTL = 86400   # 24小时

    def __init__(self, db: Database):
        self.db = db
        self._sched_cache: dict = {}
        self._sched_ts: dict = {}
        self._injury_cache_ts: float = 0  # 全局伤病缓存时间戳
        self._ratings_cache: dict = {}    # 球员评分缓存
        self._ratings_ts: float = 0

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

    def force_refresh_standings(self):
        """强制刷新战绩，忽略TTL（启动时调用）。"""
        try:
            r = requests.get(
                "https://site.api.espn.com/apis/v2/sports/basketball/nba/standings",
                timeout=10,
            )
            r.raise_for_status()
            self._parse_and_cache_standings(r.json())
            logger.info("[ESPN] 战绩强制刷新完成（30队）")
        except Exception as e:
            logger.error(f"[ESPN] 战绩强制刷新失败: {e}")
            # 检查缓存战绩是否过期（>12小时 = 数据不可靠）
            stale = self.db.execute_one("""
                SELECT MIN(strftime('%s','now') - strftime('%s', updated_at)) as age
                FROM standings
            """)
            if stale and stale.get("age") is not None:
                age_hours = float(stale["age"]) / 3600
                if age_hours > 12:
                    logger.error(
                        f"[ESPN] ⚠️ 战绩数据已过期 {age_hours:.0f} 小时！"
                        f"预测准确性可能严重下降"
                    )
            else:
                logger.error("[ESPN] ⚠️ 无任何战绩缓存数据，预测将回退到默认值")

    def _get_player_ratings(self) -> dict:
        """获取球员评分dict，优先DB，兜底硬编码。每30分钟刷新一次。"""
        if time.time() - self._ratings_ts < 1800 and self._ratings_cache:
            return self._ratings_cache
        try:
            from ml.player_ratings import get_all_ratings_as_dict
            ratings = get_all_ratings_as_dict(self.db)
            if ratings:
                self._ratings_cache = ratings
                self._ratings_ts = time.time()
                return self._ratings_cache
        except Exception:
            pass
        return {}

    def _get_player_impact(self, player_name: str) -> float:
        """获取单个球员影响力分。优先DB评分 → 兜底RAPTOR。"""
        ratings = self._get_player_ratings()
        if ratings:
            return ratings.get(player_name,
                               _RAPTOR_FALLBACK.get(player_name, _RAPTOR_FALLBACK["_bench"]))
        return _RAPTOR_FALLBACK.get(player_name, _RAPTOR_FALLBACK["_bench"])

    # ── 伤病 ──────────────────────────────────────────────────────
    def get_injury_impact(self, team_abbr: str) -> float:
        """返回伤病Elo惩罚值（越大=伤病越严重）"""
        team_id = ESPN_TEAM_IDS.get(team_abbr.upper())
        if not team_id:
            return 0.0

        # 检查DB缓存时效
        rows = self.db.execute(
            """SELECT *, strftime('%s', updated_at) as ts
               FROM injuries WHERE team_abbr=? LIMIT 1""",
            (team_abbr,)
        )
        if rows and (time.time() - float(dict(rows[0])["ts"])) < self.INJURY_TTL:
            total = sum(dict(r)["impact"] for r in rows)
            return total

        # 全局伤病端点批量刷新（1次请求覆盖30队，1小时内不重复）
        if time.time() - self._injury_cache_ts > self.INJURY_TTL:
            self._refresh_all_injuries()

        # 从DB读取该队伤病
        rows = self.db.execute(
            "SELECT impact FROM injuries WHERE team_abbr=?", (team_abbr,)
        )
        return sum(dict(r)["impact"] for r in rows) if rows else 0.0

    def _refresh_all_injuries(self, track_changes: bool = False):
        """从ESPN全局伤病端点批量拉取所有球队伤病。

        Args:
            track_changes: 是否记录伤病变动到 injury_history 表
        """
        try:
            r = requests.get(
                "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries",
                timeout=20,
            )
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            logger.warning(f"[ESPN] 全局伤病拉取失败: {e}")
            return

        team_blocks = data.get("injuries", [])
        if not team_blocks:
            return

        # ── 快照旧数据（用于对比变动）──
        old_injuries = {}  # {(team, player): status}
        if track_changes:
            old_rows = self.db.execute("SELECT team_abbr, player_name, status, impact FROM injuries")
            for row in old_rows:
                d = dict(row)
                old_injuries[(d["team_abbr"], d["player_name"])] = {
                    "status": d["status"], "impact": d["impact"]
                }

        # ── 拉取新数据 ──
        self.db.execute("DELETE FROM injuries")

        new_injuries = {}  # {(team, player): {"status": ..., "impact": ...}}
        total_count = 0
        for block in team_blocks:
            espn_id = str(block.get("id", ""))
            team_abbr = _ESPN_ID_TO_ABBR.get(espn_id, "")
            if not team_abbr:
                continue

            for inj in block.get("injuries", []):
                athlete = inj.get("athlete", {})
                name = athlete.get("displayName", "")
                if not name:
                    continue
                status = inj.get("status", "Out")
                raptor = self._get_player_impact(name)
                mult = STATUS_MULTIPLIER.get(status, 1.0)
                impact = raptor * mult * 12

                self.db.insert("""
                    INSERT INTO injuries (team_abbr, player_name, status, impact, updated_at)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (team_abbr, name, status, impact))
                new_injuries[(team_abbr, name)] = {"status": status, "impact": impact}
                total_count += 1

        self._injury_cache_ts = time.time()
        logger.info(f"[ESPN] 全局伤病更新: {total_count} 条伤病, {len(team_blocks)} 支球队")

        # ── 记录变动 ──
        if track_changes:
            self._record_injury_changes(old_injuries, new_injuries)

    def _record_injury_changes(self, old: dict, new: dict):
        """对比新旧伤病名单，记录变动到 injury_history。

        变动类型:
          - recovered: 旧名单有、新名单没有 → 康复
          - new_injury: 新名单有、旧名单没有 → 新伤
          - status_change: 两边都有但状态变了 → 状态变化（好转/恶化）
        """
        changes = []

        # 康复球员: 旧有新无
        for (team, player), info in old.items():
            if (team, player) not in new:
                changes.append({
                    "team": team, "player": player,
                    "event_type": "recovered",
                    "old_status": info["status"], "new_status": None,
                    "impact": info["impact"],
                })

        # 新伤球员: 新有旧无
        for (team, player), info in new.items():
            if (team, player) not in old:
                changes.append({
                    "team": team, "player": player,
                    "event_type": "new_injury",
                    "old_status": None, "new_status": info["status"],
                    "impact": info["impact"],
                })

        # 状态变化: 两边都有但状态不同
        for (team, player), new_info in new.items():
            if (team, player) in old:
                old_info = old[(team, player)]
                if old_info["status"] != new_info["status"]:
                    changes.append({
                        "team": team, "player": player,
                        "event_type": "status_change",
                        "old_status": old_info["status"],
                        "new_status": new_info["status"],
                        "impact": new_info["impact"],
                    })

        # 写入数据库
        for c in changes:
            try:
                self.db.insert("""
                    INSERT INTO injury_history
                        (team_abbr, player_name, event_type, old_status, new_status, impact)
                    VALUES (?,?,?,?,?,?)
                """, (c["team"], c["player"], c["event_type"],
                      c["old_status"], c["new_status"], c["impact"]))
            except Exception:
                pass

        # 日志
        recovered = [c for c in changes if c["event_type"] == "recovered"]
        new_inj = [c for c in changes if c["event_type"] == "new_injury"]
        status_chg = [c for c in changes if c["event_type"] == "status_change"]

        if recovered:
            names = ", ".join(f"{c['team']}-{c['player']}" for c in recovered[:5])
            logger.info(f"[ESPN] ✅ 康复({len(recovered)}人): {names}")
        if new_inj:
            names = ", ".join(f"{c['team']}-{c['player']}({c['new_status']})" for c in new_inj[:5])
            logger.info(f"[ESPN] 🚨 新伤({len(new_inj)}人): {names}")
        if status_chg:
            names = ", ".join(
                f"{c['team']}-{c['player']}({c['old_status']}→{c['new_status']})"
                for c in status_chg[:5]
            )
            logger.info(f"[ESPN] 🔄 状态变化({len(status_chg)}人): {names}")

    def get_recent_recoveries(self, teams: list = None, days: int = 3) -> list:
        """获取近期康复球员列表。

        Args:
            teams: 只看这些球队（None=全部）
            days: 回看天数

        Returns:
            [{"team_abbr", "player_name", "old_status", "impact", "event_date"}]
        """
        if teams:
            placeholders = ",".join(["?"] * len(teams))
            rows = self.db.execute(f"""
                SELECT team_abbr, player_name, old_status, impact, event_date
                FROM injury_history
                WHERE event_type='recovered'
                  AND event_date >= date('now', '-{days} days')
                  AND team_abbr IN ({placeholders})
                ORDER BY event_date DESC, impact DESC
            """, tuple(t.upper() for t in teams))
        else:
            rows = self.db.execute(f"""
                SELECT team_abbr, player_name, old_status, impact, event_date
                FROM injury_history
                WHERE event_type='recovered'
                  AND event_date >= date('now', '-{days} days')
                ORDER BY event_date DESC, impact DESC
            """)
        return [dict(r) for r in rows] if rows else []

    def get_recent_injury_changes(self, teams: list = None, days: int = 1) -> list:
        """获取近期所有伤病变动（康复+新伤+状态变化）。"""
        if teams:
            placeholders = ",".join(["?"] * len(teams))
            rows = self.db.execute(f"""
                SELECT team_abbr, player_name, event_type, old_status, new_status,
                       impact, event_date
                FROM injury_history
                WHERE event_date >= date('now', '-{days} days')
                  AND team_abbr IN ({placeholders})
                ORDER BY event_date DESC, impact DESC
            """, tuple(t.upper() for t in teams))
        else:
            rows = self.db.execute(f"""
                SELECT team_abbr, player_name, event_type, old_status, new_status,
                       impact, event_date
                FROM injury_history
                WHERE event_date >= date('now', '-{days} days')
                ORDER BY event_date DESC, impact DESC
            """)
        return [dict(r) for r in rows] if rows else []

    def get_injury_report(self, today_teams: list = None) -> dict:
        """生成每日伤病报告，返回结构化数据。

        Args:
            today_teams: 今天有比赛的球队缩写列表，高亮显示

        Returns:
            {
                "total": 总伤病数,
                "teams": {abbr: [{"name", "status", "impact", "is_star"}]},
                "star_injuries": [核心球员伤病列表],
                "today_impact": {abbr: total_impact} (仅今天有比赛的队)
            }
        """
        # 强制刷新
        self._injury_cache_ts = 0
        self._refresh_all_injuries()

        today_teams = set(t.upper() for t in (today_teams or []))

        rows = self.db.execute(
            "SELECT team_abbr, player_name, status, impact FROM injuries ORDER BY team_abbr, impact DESC"
        )

        teams = {}
        star_injuries = []
        today_impact = {}

        for r in rows:
            d = dict(r)
            abbr = d["team_abbr"]
            player_impact = self._get_player_impact(d["player_name"])
            is_star = player_impact >= 5.0  # All-Star级及以上
            entry = {
                "name": d["player_name"],
                "status": d["status"],
                "impact": d["impact"],
                "is_star": is_star,
            }
            teams.setdefault(abbr, []).append(entry)
            if is_star and d["status"] in ("Out", "Doubtful"):
                star_injuries.append({"team": abbr, **entry})
            if abbr in today_teams:
                today_impact[abbr] = today_impact.get(abbr, 0) + d["impact"]

        return {
            "total": len(rows),
            "teams": teams,
            "star_injuries": sorted(star_injuries, key=lambda x: x["impact"], reverse=True),
            "today_impact": today_impact,
        }

    def cleanup_recovered_players(self) -> dict:
        """每日伤病清理 — 重新拉取ESPN数据，记录康复/新伤/状态变化。

        流程:
          1. 快照旧伤病名单
          2. 拉取ESPN最新数据（DELETE + INSERT）
          3. 对比差异 → 记录到 injury_history
          4. 返回变动摘要
        """
        before = self.db.execute_one("SELECT COUNT(*) as c FROM injuries")
        before_count = before["c"] if before else 0

        # 带变动追踪的刷新
        self._injury_cache_ts = 0
        self._refresh_all_injuries(track_changes=True)

        after = self.db.execute_one("SELECT COUNT(*) as c FROM injuries")
        after_count = after["c"] if after else 0

        # 查询今日变动统计
        today_changes = self.db.execute("""
            SELECT event_type, COUNT(*) as cnt
            FROM injury_history
            WHERE event_date = date('now')
            GROUP BY event_type
        """)
        change_stats = {dict(r)["event_type"]: dict(r)["cnt"] for r in today_changes} if today_changes else {}

        recovered_count = change_stats.get("recovered", 0)
        new_injury_count = change_stats.get("new_injury", 0)
        status_change_count = change_stats.get("status_change", 0)

        logger.info(
            f"[ESPN] 伤病刷新: {before_count}→{after_count} | "
            f"康复:{recovered_count} 新伤:{new_injury_count} 状态变化:{status_change_count}"
        )

        # 清理30天前的历史记录
        self.db.insert("DELETE FROM injury_history WHERE event_date < date('now', '-30 days')")

        return {
            "before": before_count,
            "after": after_count,
            "recovered": recovered_count,
            "new_injuries": new_injury_count,
            "status_changes": status_change_count,
        }

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
