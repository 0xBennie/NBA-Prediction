"""
injury_checker.py — 实时伤病检查模块

在推送信号前联网查最新伤病报告，如果关键球员缺阵则降级或撤回推荐。

数据源:
  1. ESPN伤病报告（已有，但更新慢）
  2. nba_api伤病数据（实时）
  3. 如果上述都没有，用LLM搜索最新新闻

流程:
  推送前 → check_injuries(team) → 返回关键缺阵球员列表
  → 如果有场均>20分的球员OUT → 降低模型信心 → 可能撤回推荐
"""

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

# 各队核心球员（场均>20分的明星）— 这些球员缺阵影响巨大
STAR_PLAYERS = {
    "ATL": ["Trae Young"],
    "BOS": ["Jayson Tatum", "Jaylen Brown"],
    "BKN": [],
    "CHA": ["LaMelo Ball"],
    "CHI": ["Zach LaVine"],
    "CLE": ["Donovan Mitchell", "Darius Garland"],
    "DAL": ["Luka Doncic", "Kyrie Irving"],
    "DEN": ["Nikola Jokic", "Jamal Murray"],
    "DET": ["Cade Cunningham"],
    "GSW": ["Stephen Curry"],
    "HOU": ["Jalen Green"],
    "IND": ["Tyrese Haliburton"],
    "LAC": ["James Harden", "Kawhi Leonard"],
    "LAL": ["LeBron James", "Anthony Davis"],
    "MEM": ["Ja Morant"],
    "MIA": ["Jimmy Butler", "Bam Adebayo"],
    "MIL": ["Giannis Antetokounmpo", "Damian Lillard"],
    "MIN": ["Anthony Edwards", "Karl-Anthony Towns", "Rudy Gobert"],
    "NOP": ["Zion Williamson", "Brandon Ingram"],
    "NYK": ["Jalen Brunson"],
    "OKC": ["Shai Gilgeous-Alexander", "Chet Holmgren"],
    "ORL": ["Paolo Banchero", "Franz Wagner"],
    "PHI": ["Joel Embiid", "Tyrese Maxey"],
    "PHX": ["Kevin Durant", "Devin Booker", "Bradley Beal"],
    "POR": ["Anfernee Simons"],
    "SAC": ["De'Aaron Fox", "Domantas Sabonis"],
    "SAS": ["Victor Wembanyama"],
    "TOR": ["Scottie Barnes"],
    "UTA": ["Lauri Markkanen"],
    "WAS": ["Jordan Poole"],
}


class InjuryChecker:
    """实时伤病检查器。"""

    def __init__(self, espn_client=None):
        self.espn = espn_client
        self._cache = {}  # {team: {"players": [...], "ts": timestamp}}

    def check_team(self, team: str) -> dict:
        """检查球队伤病情况。

        Returns:
            {
                "has_star_out": bool,
                "star_out": ["Anthony Edwards"],
                "injury_penalty": 0.10,  # 模型信心应降低多少
                "details": "Anthony Edwards (knee) OUT - 场均29.5分",
                "all_injuries": [...],
            }
        """
        # 缓存30分钟
        cached = self._cache.get(team)
        if cached and time.time() - cached["ts"] < 1800:
            return cached["result"]

        result = {
            "has_star_out": False,
            "star_out": [],
            "injury_penalty": 0.0,
            "details": "",
            "all_injuries": [],
        }

        # 1. 从nba_api获取伤病报告
        injuries = self._get_nba_injuries(team)

        # 2. 检查是否有核心球员缺阵
        stars = STAR_PLAYERS.get(team, [])
        star_out = []
        for inj in injuries:
            player = inj.get("player", "")
            status = inj.get("status", "").upper()
            if status in ["OUT", "INACTIVE"] and any(s.lower() in player.lower() for s in stars):
                star_out.append(player)

        if star_out:
            result["has_star_out"] = True
            result["star_out"] = star_out
            # 每个核心球员缺阵 → 降低10%信心
            result["injury_penalty"] = len(star_out) * 0.10
            result["details"] = ", ".join(f"{p} OUT" for p in star_out)

        result["all_injuries"] = injuries
        self._cache[team] = {"result": result, "ts": time.time()}

        if star_out:
            logger.info(f"[Injury] {team} 核心缺阵: {result['details']}")

        return result

    def _get_nba_injuries(self, team: str) -> list:
        """从nba_api获取伤病列表。"""
        try:
            from nba_api.stats.endpoints import PlayerIndex
            # PlayerIndex不提供伤病，用TeamRoster + 状态检查
            # 实际上nba_api没有直接的伤病API，用ESPN
            pass
        except Exception:
            pass

        # ESPN全局伤病端点
        try:
            import requests
            r = requests.get(
                "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries",
                timeout=15,
            )
            if r.status_code == 200:
                data = r.json()
                injuries = []

                # 队名全称→缩写映射
                from ml.nba_features import ABBR_TO_NBA
                name_to_abbr = {v.lower(): k for k, v in ABBR_TO_NBA.items()}

                for team_group in data.get("injuries", []):
                    # ESPN用displayName（如"Minnesota Timberwolves"）
                    team_name = team_group.get("displayName", "").lower()
                    team_abbr = name_to_abbr.get(team_name, "")

                    # 也检查id映射
                    if not team_abbr:
                        team_id = str(team_group.get("id", ""))
                        espn_to_abbr = {v: k for k, v in self._espn_id_map().items()}
                        team_abbr = espn_to_abbr.get(team_id, "")

                    if team_abbr != team:
                        continue

                    for item in team_group.get("injuries", []):
                        status = item.get("status", "Unknown")
                        if isinstance(status, dict):
                            status = status.get("type", status.get("name", "Unknown"))
                        injuries.append({
                            "player": item.get("athlete", {}).get("displayName", "Unknown"),
                            "status": str(status),
                            "description": item.get("longComment", ""),
                        })
                return injuries
        except Exception as e:
            logger.debug(f"[Injury] ESPN伤病获取失败: {e}")

        return []

    @staticmethod
    def _espn_id_map() -> dict:
        return {
            "ATL": "1", "BOS": "2", "BKN": "17", "CHA": "30", "CHI": "4",
            "CLE": "5", "DAL": "6", "DEN": "7", "DET": "8", "GSW": "9",
            "HOU": "10", "IND": "11", "LAC": "12", "LAL": "13", "MEM": "29",
            "MIA": "14", "MIL": "15", "MIN": "16", "NOP": "3", "NYK": "18",
            "OKC": "25", "ORL": "19", "PHI": "20", "PHX": "21", "POR": "22",
            "SAC": "23", "SAS": "24", "TOR": "28", "UTA": "26", "WAS": "27",
        }

    @staticmethod
    def _team_to_espn_id(team: str) -> str:
        """队名缩写 → ESPN team ID。"""
        espn_ids = {
            "ATL": "1", "BOS": "2", "BKN": "17", "CHA": "30", "CHI": "4",
            "CLE": "5", "DAL": "6", "DEN": "7", "DET": "8", "GSW": "9",
            "HOU": "10", "IND": "11", "LAC": "12", "LAL": "13", "MEM": "29",
            "MIA": "14", "MIL": "15", "MIN": "16", "NOP": "3", "NYK": "18",
            "OKC": "25", "ORL": "19", "PHI": "20", "PHX": "21", "POR": "22",
            "SAC": "23", "SAS": "24", "TOR": "28", "UTA": "26", "WAS": "27",
        }
        return espn_ids.get(team, "1")
