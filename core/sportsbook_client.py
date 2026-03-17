"""
sportsbook_client.py — The Odds API 赔率数据源（4级瀑布回退）

数据源: The Odds API v4
  - GET /odds: 多家博彩公司h2h赔率（1 credit/次，批量获取+5分钟缓存）
  - GET /scores: 比赛结果（免费）
  - GET /events: 赛事列表（免费）
  - GET /sports: 健康检查（免费）
  - 配额跟踪: 响应头 x-requests-remaining / x-requests-used

4级瀑布赔率源: Pinnacle → DraftKings → FanDuel → 共识均值
去vig: Shin法（考虑信息不对称，比简单归一化更准确）
"""

import os
import time
import math
import logging
import requests
from typing import Optional, List

logger = logging.getLogger(__name__)

TEAM_ALIASES = {
    "ATL": ["atlanta", "hawks"],
    "BOS": ["boston", "celtics"],
    "BKN": ["brooklyn", "nets"],
    "CHA": ["charlotte", "hornets"],
    "CHI": ["chicago", "bulls"],
    "CLE": ["cleveland", "cavaliers"],
    "DAL": ["dallas", "mavericks"],
    "DEN": ["denver", "nuggets"],
    "DET": ["detroit", "pistons"],
    "GSW": ["golden state", "warriors"],
    "HOU": ["houston", "rockets"],
    "IND": ["indiana", "pacers"],
    "LAC": ["la clippers", "clippers"],
    "LAL": ["lakers", "los angeles lakers"],
    "MEM": ["memphis", "grizzlies"],
    "MIA": ["miami", "heat"],
    "MIL": ["milwaukee", "bucks"],
    "MIN": ["minnesota", "timberwolves"],
    "NOP": ["new orleans", "pelicans"],
    "NYK": ["new york", "knicks"],
    "OKC": ["oklahoma", "thunder"],
    "ORL": ["orlando", "magic"],
    "PHI": ["philadelphia", "76ers", "sixers"],
    "PHX": ["phoenix", "suns"],
    "POR": ["portland", "trail blazers", "blazers"],
    "SAC": ["sacramento", "kings"],
    "SAS": ["san antonio", "spurs"],
    "TOR": ["toronto", "raptors"],
    "UTA": ["utah", "jazz"],
    "WAS": ["washington", "wizards"],
}


def _match(name: str, abbr: str) -> bool:
    n = name.lower()
    return any(k in n for k in TEAM_ALIASES.get(abbr.upper(), [abbr.lower()]))


# ── Shin法去vig ──────────────────────────────────────────────────
def _shin_devig(h_dec: float, a_dec: float) -> dict:
    """Shin法去vig — 考虑信息不对称，比简单归一化更准确。

    Shin模型假设: 部分投注来自内幕信息者，bookmaker的margin
    主要用于对冲内幕风险，而非均匀加在所有结果上。
    因此热门(概率高)的结果被去掉的vig更少，冷门被去掉更多。
    """
    r1, r2 = 1 / h_dec, 1 / a_dec
    total = r1 + r2

    if total <= 1.0:
        return {"home": round(r1 / total, 4), "away": round(r2 / total, 4)}

    z = total - 1

    def shin_prob(r):
        discriminant = z * z + 4 * (1 - z) * r * r
        return (math.sqrt(discriminant) - z) / (2 * (1 - z))

    p_home = shin_prob(r1)
    p_away = shin_prob(r2)
    t = p_home + p_away
    return {"home": round(p_home / t, 4), "away": round(p_away / t, 4)}


# ── 4级瀑布赔率源优先级 ──────────────────────────────────────────
BOOKMAKER_WATERFALL = [
    ("pinnacle", "Pinnacle", 1.0),
    ("draftkings", "DraftKings", 0.85),
    ("fanduel", "FanDuel", 0.85),
]


# ── The Odds API (批量获取+配额跟踪) ────────────────────────────
_odds_api_cache: dict = {"data": None, "ts": 0, "ttl": 300}  # 5分钟缓存
_odds_api_quota: dict = {"remaining": None, "used": None}


def _odds_api_fetch_all(key: str) -> Optional[list]:
    """批量获取所有NBA h2h赔率（2次API调用 = 2 credits）。
    1) Pinnacle专项请求（Pinnacle不在US区域，需要单独请求）
    2) US区域全部博彩公司（DraftKings, FanDuel等）
    合并后缓存5分钟。
    """
    now = time.time()
    if _odds_api_cache["data"] is not None and now - _odds_api_cache["ts"] < _odds_api_cache["ttl"]:
        return _odds_api_cache["data"]

    merged = {}  # key = game_id -> game data with merged bookmakers

    # 请求1: Pinnacle (单独请求，因为Pinnacle在EU区域)
    try:
        r1 = requests.get(
            "https://api.the-odds-api.com/v4/sports/basketball_nba/odds",
            params={"apiKey": key, "regions": "us", "markets": "h2h",
                    "bookmakers": "pinnacle", "oddsFormat": "decimal"},
            timeout=10,
        )
        r1.raise_for_status()
        for g in r1.json():
            gid = g.get("id", "")
            merged[gid] = g
    except Exception as e:
        logger.warning(f"[OddsAPI] Pinnacle请求失败: {e}")

    # 请求2: US区域所有博彩公司
    try:
        r2 = requests.get(
            "https://api.the-odds-api.com/v4/sports/basketball_nba/odds",
            params={"apiKey": key, "regions": "us", "markets": "h2h",
                    "oddsFormat": "decimal"},
            timeout=10,
        )
        r2.raise_for_status()
        _odds_api_quota["remaining"] = r2.headers.get("x-requests-remaining")
        _odds_api_quota["used"] = r2.headers.get("x-requests-used")
        logger.info(f"[OddsAPI] 配额: 剩余{_odds_api_quota['remaining']}, 已用{_odds_api_quota['used']}")

        for g in r2.json():
            gid = g.get("id", "")
            if gid in merged:
                # 合并博彩公司数据
                existing_bm_keys = {bm["key"] for bm in merged[gid].get("bookmakers", [])}
                for bm in g.get("bookmakers", []):
                    if bm["key"] not in existing_bm_keys:
                        merged[gid]["bookmakers"].append(bm)
            else:
                merged[gid] = g
    except Exception as e:
        logger.warning(f"[OddsAPI] US区域请求失败: {e}")

    if not merged:
        return None

    data = list(merged.values())
    _odds_api_cache["data"] = data
    _odds_api_cache["ts"] = now
    return data


def _extract_h2h_from_bookmaker(bm: dict, home: str, away: str) -> Optional[tuple]:
    """从单个博彩公司数据中提取h2h赔率，返回(home_decimal, away_decimal)或None"""
    for mkt in bm.get("markets", []):
        if mkt["key"] != "h2h":
            continue
        oc = mkt["outcomes"]
        hd = next((o["price"] for o in oc if _match(o["name"], home)), None)
        ad = next((o["price"] for o in oc if _match(o["name"], away)), None)
        if hd and ad:
            return (hd, ad)
    return None


def _odds_api(home: str, away: str, key: str) -> Optional[dict]:
    """从批量缓存中查找特定比赛的赔率，按瀑布优先级回退。
    Pinnacle → DraftKings → FanDuel → 共识均值
    """
    games = _odds_api_fetch_all(key)
    if not games:
        return None

    for g in games:
        if not (_match(g.get("home_team", ""), home) and
                _match(g.get("away_team", ""), away)):
            continue

        bookmakers = {bm["key"]: bm for bm in g.get("bookmakers", [])}

        # 瀑布回退：按优先级依次尝试
        for bm_key, bm_name, quality in BOOKMAKER_WATERFALL:
            if bm_key in bookmakers:
                odds = _extract_h2h_from_bookmaker(bookmakers[bm_key], home, away)
                if odds:
                    result = _shin_devig(odds[0], odds[1])
                    result["source"] = bm_key
                    result["source_quality"] = quality
                    if bm_key != "pinnacle":
                        logger.info(f"[OddsAPI] {away}@{home} Pinnacle无数据，回退到{bm_name}")
                    return result

        # 最终回退：所有可用博彩公司的共识均值
        all_home_probs = []
        all_away_probs = []
        for bm in g.get("bookmakers", []):
            odds = _extract_h2h_from_bookmaker(bm, home, away)
            if odds:
                devigged = _shin_devig(odds[0], odds[1])
                all_home_probs.append(devigged["home"])
                all_away_probs.append(devigged["away"])

        if all_home_probs:
            avg_home = sum(all_home_probs) / len(all_home_probs)
            avg_away = sum(all_away_probs) / len(all_away_probs)
            total = avg_home + avg_away
            logger.info(f"[OddsAPI] {away}@{home} 使用{len(all_home_probs)}家博彩公司共识均值")
            return {
                "home": round(avg_home / total, 4),
                "away": round(avg_away / total, 4),
                "source": "consensus",
                "source_quality": 0.7,
            }

    return None


def get_nba_scores(key: str, days_from: int = 1) -> List[dict]:
    """获取NBA比赛结果（免费端点，不消耗配额）。"""
    try:
        r = requests.get(
            "https://api.the-odds-api.com/v4/sports/basketball_nba/scores",
            params={"apiKey": key, "daysFrom": days_from},
            timeout=10,
        )
        r.raise_for_status()
        results = []
        for g in r.json():
            scores = g.get("scores")
            if not scores:
                continue
            home_name = g.get("home_team", "")
            away_name = g.get("away_team", "")
            home_score = None
            away_score = None
            for s in scores:
                if s.get("name") == home_name:
                    home_score = int(s.get("score", 0))
                elif s.get("name") == away_name:
                    away_score = int(s.get("score", 0))
            results.append({
                "home_team": home_name,
                "away_team": away_name,
                "home_score": home_score,
                "away_score": away_score,
                "completed": g.get("completed", False),
                "commence_time": g.get("commence_time", ""),
            })
        return results
    except Exception as e:
        logger.warning(f"[OddsAPI] scores获取失败: {e}")
        return []


def get_odds_api_quota() -> dict:
    """返回当前API配额状态"""
    return dict(_odds_api_quota)


# ── 主类 ──────────────────────────────────────────────────────────
class SportsbookClient:

    def __init__(self, odds_api_key: Optional[str] = None):
        self.key = odds_api_key or os.getenv("ODDS_API_KEY", "")

    def get_fair_prob(self, home: str, away: str) -> Optional[dict]:
        """获取公平概率（4级瀑布回退）。无数据返回None。"""
        if not self.key:
            logger.error("[OddsAPI] 未配置ODDS_API_KEY")
            return None

        result = _odds_api(home, away, self.key)
        if result is None:
            logger.warning(f"[OddsAPI] 未找到 {away}@{home} 的任何赔率数据")
            return None

        source = result.get("source", "unknown")
        logger.info(f"[OddsAPI] {away}@{home} | {source} H:{result['home']} A:{result['away']}")
        return result

    def get_all_nba_odds(self) -> list:
        """批量获取所有NBA比赛的赔率（瀑布回退，1次API调用）。"""
        if not self.key:
            return []
        games = _odds_api_fetch_all(self.key)
        if not games:
            return []

        results = []
        for g in games:
            home_name = g.get("home_team", "")
            away_name = g.get("away_team", "")
            # 用和 _odds_api 相同的瀑布逻辑
            result = _odds_api(home_name, away_name, self.key)
            if result:
                results.append({
                    "home_team": home_name,
                    "away_team": away_name,
                    "home_prob": result["home"],
                    "away_prob": result["away"],
                    "source": result.get("source", "unknown"),
                })
        return results

    def get_scores(self, days_from: int = 1) -> list:
        """获取NBA比赛结果（免费端点）"""
        return get_nba_scores(self.key, days_from) if self.key else []

    def get_quota(self) -> dict:
        """返回API配额状态"""
        return get_odds_api_quota()

    def health_check(self) -> dict:
        status = {}
        if self.key:
            try:
                r = requests.get("https://api.the-odds-api.com/v4/sports",
                                 params={"apiKey": self.key}, timeout=8)
                status["odds_api"] = "✅" if r.status_code == 200 else f"❌ {r.status_code}"
                remaining = r.headers.get("x-requests-remaining", "?")
                used = r.headers.get("x-requests-used", "?")
                status["quota"] = f"剩余:{remaining} 已用:{used}"
            except Exception as e:
                status["odds_api"] = f"❌ {e}"
        else:
            status["odds_api"] = "❌ 未配置ODDS_API_KEY"
        return status
