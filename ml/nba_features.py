"""
nba_features.py — NBA比赛特征工程（v2 研究优化版）

特征维度（每队15维 + 3全局 = 33维）:
  基础: 近10场胜率、净胜分、场均得分
  四因素(Dean Oliver): eFG%, TOV%, OREB%, FT_RATE — 解释96%胜率方差
  赛季: 胜率、净rating
  主客场: 主/客场胜率分裂
  疲劳: 赛程密度(7天内场次) + 近5场累计分钟 + 休息天数
  效率趋势: 近5场eFG% vs 赛季eFG%（衰退检测）
  伤病: Elo惩罚
  全局: 对手强度、主场优势、客队信心惩罚

研究依据:
  - Dean Oliver Four Factors (96% variance explained)
  - 赛程密度 > 单纯B2B (Wharton, CMU 2025)
  - 累计分钟疲劳 (arXiv 2112.14649)
  - 校准 > 准确率 (ScienceDirect, ROI +35% vs -35%)
  - 近5场eFG%趋势预测68%下场表现 (Medium/Jesse Fu)

依赖: nba_api, requests
"""

import json
import logging
import time
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# NBA 队名缩写 → nba_api team_id
# nba_api 用全名匹配，这里做缩写→全名映射
ABBR_TO_NBA = {
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets", "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "LA Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat", "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans", "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors", "UTA": "Utah Jazz", "WAS": "Washington Wizards",
}

# nba_api team_id 缓存
_TEAM_IDS = {}


def _get_team_id(abbr: str) -> Optional[int]:
    """获取nba_api team_id。"""
    if abbr in _TEAM_IDS:
        return _TEAM_IDS[abbr]
    try:
        from nba_api.stats.static import teams as nba_teams
        all_teams = nba_teams.get_teams()
        for t in all_teams:
            for a, full in ABBR_TO_NBA.items():
                if t["full_name"] == full:
                    _TEAM_IDS[a] = t["id"]
        return _TEAM_IDS.get(abbr)
    except Exception as e:
        logger.warning(f"[Features] 获取team_id失败: {e}")
        return None


class NBAFeatureBuilder:
    """NBA比赛特征构建器。"""

    def __init__(self, db, espn):
        self.db = db
        self.espn = espn
        self._game_log_cache = {}  # {team_abbr: {"data": [...], "ts": timestamp}}
        self._bbref_cache = {}  # basketball-reference 高级数据缓存
        self._bbref_cache_ts = 0
        self._team_ratings = None  # OpenSkill评分 (lazy init)
        self._ensure_table()

    def _ensure_table(self):
        try:
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS team_features (
                    team_abbr TEXT NOT NULL,
                    game_date TEXT NOT NULL,
                    features_json TEXT NOT NULL,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(team_abbr, game_date)
                )
            """)
        except Exception:
            pass

    def get_game_features(self, home: str, away: str, game_date: str = "") -> Optional[dict]:
        """构建一场比赛的完整特征向量。

        Returns:
            {
                "home_win_pct_10": 0.7, "home_net_rating_10": 5.2,
                "home_season_win_pct": 0.65, "home_season_net": 4.8,
                "home_home_win_pct": 0.75, "home_rest_days": 2,
                "home_injury_elo": -15.0, "home_opp_strength": 0.55,
                "away_win_pct_10": ..., ...,
                "home_advantage": 1  # 主场优势标记
            }
        """
        if not game_date:
            game_date = datetime.now().strftime("%Y-%m-%d")

        # 检查缓存
        cached = self._get_cached(home, away, game_date)
        if cached:
            return cached

        try:
            home_f = self._build_team_features(home, game_date, is_home=True)
            time.sleep(0.7)  # nba_api rate limit
            away_f = self._build_team_features(away, game_date, is_home=False)

            if not home_f or not away_f:
                missing = []
                if not home_f:
                    missing.append(f"主队{home}")
                if not away_f:
                    missing.append(f"客队{away}")
                logger.warning(
                    f"[Features] {away}@{home} {','.join(missing)}特征构建失败，"
                    f"回退到standings兜底（预测准确性降低）"
                )
                return self._fallback_features(home, away, game_date)

            features = {}
            for k, v in home_f.items():
                features[f"home_{k}"] = v
            for k, v in away_f.items():
                features[f"away_{k}"] = v

            # 对手强度交叉
            features["home_opp_strength"] = away_f.get("season_win_pct", 0.5)
            features["away_opp_strength"] = home_f.get("season_win_pct", 0.5)
            features["home_advantage"] = 1.0

            # 客队信心惩罚：客队客场胜率差 + 赛程密集 → 额外风险
            away_venue_wp = away_f.get("venue_win_pct", 0.5)
            away_density = away_f.get("schedule_density", 2)
            # 客场胜率<40% + 7天3+场 → 惩罚系数
            penalty = 0.0
            if away_venue_wp < 0.40:
                penalty += (0.40 - away_venue_wp) * 0.5  # 客场弱队惩罚
            if away_density >= 3:
                penalty += 0.03 * (away_density - 2)  # 赛程密集惩罚
            features["away_confidence_penalty"] = round(penalty, 4)

            # ── OpenSkill 贝叶斯技能评分 ──
            skill_features = self._get_skill_features(home, away)
            features.update(skill_features)

            # ── Basketball-Reference 高级数据 ──
            bbref = self._get_bbref_features(home, away)
            features.update(bbref)

            # 缓存
            self._save_cache(home, away, game_date, features)
            return features

        except Exception as e:
            logger.warning(f"[Features] 特征构建失败 {away}@{home}: {e}")
            return self._fallback_features(home, away, game_date)

    def _build_team_features(self, team: str, game_date: str, is_home: bool) -> Optional[dict]:
        """构建单支球队的特征（v2: +Four Factors, 赛程密度, 疲劳, 效率趋势）。"""
        game_log = self._get_game_log(team)
        standings = self._get_standings(team)

        if not game_log and not standings:
            return None

        features = {}
        recent = game_log[:10] if game_log else []
        recent5 = game_log[:5] if game_log else []

        # ── 基础: 近10场表现 ──
        if recent:
            wins = sum(1 for g in recent if g.get("wl") == "W")
            features["win_pct_10"] = wins / len(recent)
            pts = [g.get("pts", 0) for g in recent]
            features["avg_pts_10"] = sum(pts) / len(recent)
            # net_rating: 用实际 PLUS_MINUS（每场净胜分）的均值
            margins = [g.get("plus_minus", 0) for g in recent]
            features["net_rating_10"] = sum(margins) / len(recent)
        else:
            features["win_pct_10"] = standings.get("win_rate", 0.5) if standings else 0.5
            features["net_rating_10"] = standings.get("ppg_diff", 0) if standings else 0
            features["avg_pts_10"] = standings.get("ppg", 105) if standings else 105

        # ── Four Factors (Dean Oliver) — 解释96%胜率方差 ──
        if recent:
            features["efg_pct"] = sum(g.get("efg", 0.45) for g in recent) / len(recent)
            features["tov_rate"] = sum(g.get("tov_rate", 0.12) for g in recent) / len(recent)
            features["oreb_rate"] = sum(g.get("oreb_rate", 0.25) for g in recent) / len(recent)
            features["ft_rate"] = sum(g.get("ft_rate", 0.20) for g in recent) / len(recent)
        else:
            features["efg_pct"] = 0.50
            features["tov_rate"] = 0.12
            features["oreb_rate"] = 0.25
            features["ft_rate"] = 0.20

        # ── 赛季总体 ──
        if standings:
            total = (standings.get("wins", 0) + standings.get("losses", 0)) or 1
            features["season_win_pct"] = standings.get("wins", 0) / total
            features["season_net"] = standings.get("ppg_diff", 0)
        else:
            features["season_win_pct"] = features["win_pct_10"]
            features["season_net"] = features["net_rating_10"]

        # ── 主客场分裂 ──
        if game_log:
            home_games = [g for g in game_log if g.get("is_home")]
            away_games = [g for g in game_log if not g.get("is_home")]
            if is_home and home_games:
                hw = sum(1 for g in home_games if g.get("wl") == "W")
                features["venue_win_pct"] = hw / len(home_games)
            elif not is_home and away_games:
                aw = sum(1 for g in away_games if g.get("wl") == "W")
                features["venue_win_pct"] = aw / len(away_games)
            else:
                features["venue_win_pct"] = features["season_win_pct"]
        else:
            features["venue_win_pct"] = features["season_win_pct"]

        # ── 赛程密度 (7天内比赛场次) — 比单纯B2B更有预测力 ──
        schedule_density = 0
        if game_log:
            try:
                today = datetime.strptime(game_date, "%Y-%m-%d")
                seven_days_ago = today - timedelta(days=7)
                for g in game_log[:10]:
                    gd = datetime.strptime(g["date"], "%Y-%m-%d")
                    if seven_days_ago <= gd < today:
                        schedule_density += 1
            except Exception:
                schedule_density = 2
        features["schedule_density"] = schedule_density

        # ── 休息天数 ──
        if game_log and game_log[0].get("date"):
            try:
                last_game = datetime.strptime(game_log[0]["date"], "%Y-%m-%d")
                today = datetime.strptime(game_date, "%Y-%m-%d")
                features["rest_days"] = max(0, (today - last_game).days - 1)
            except Exception:
                features["rest_days"] = 1
        else:
            features["rest_days"] = 1

        # ── 近5场累计分钟疲劳 — 比休息天数更有预测力 ──
        if recent5:
            features["fatigue_minutes"] = sum(g.get("minutes", 240) for g in recent5) / len(recent5)
        else:
            features["fatigue_minutes"] = 240

        # ── 效率趋势（衰退检测）— 近5场eFG% vs 赛季eFG% ──
        if recent5 and features["efg_pct"] > 0:
            recent5_efg = sum(g.get("efg", 0.45) for g in recent5) / len(recent5)
            features["efg_trend"] = recent5_efg - features["efg_pct"]  # 正=上升，负=衰退
        else:
            features["efg_trend"] = 0

        # ── 攻防Rating — 用实际得失分计算 ──
        if recent:
            features["off_rating"] = features["avg_pts_10"] / 100  # 标准化到~1.1
            # def_rating: 用实际对手得分（越低=防守越好）
            opp_pts_list = [g.get("opp_pts", 0) for g in recent]
            avg_opp_pts = sum(opp_pts_list) / len(recent) if opp_pts_list else 105
            features["def_rating"] = avg_opp_pts / 100  # 标准化到~1.05
        else:
            features["off_rating"] = 1.05
            features["def_rating"] = 1.05

        # ── 连胜/连败势头 — 研究: 动量效应对胜率有3-5%影响 ──
        if game_log:
            streak = 0
            for g in game_log[:10]:
                if g.get("wl") == "W":
                    if streak >= 0:
                        streak += 1
                    else:
                        break
                else:
                    if streak <= 0:
                        streak -= 1
                    else:
                        break
            features["streak"] = streak  # 正=连胜, 负=连败
        else:
            features["streak"] = 0

        # ── 伤病 ──
        try:
            features["injury_elo"] = self.espn.get_injury_impact(team)
        except Exception:
            features["injury_elo"] = 0

        return features

    def _get_game_log(self, team: str) -> list:
        """获取球队近期比赛日志（LeagueGameLog，有PLUS_MINUS，带缓存）。

        改用 LeagueGameLog 而非 TeamGameLog，因为前者有 PLUS_MINUS 列，
        可以算出真实对手得分和净胜分，是预测准确性的关键数据。
        LeagueGameLog 一次拉全联盟数据，批量缓存所有球队。
        """
        # 内存缓存2小时
        cached = self._game_log_cache.get(team)
        if cached and time.time() - cached["ts"] < 7200:
            return cached["data"]

        try:
            # 批量加载全联盟（如果还没加载过 或 缓存全部过期）
            self._load_league_game_logs()

            cached = self._game_log_cache.get(team)
            return cached["data"] if cached else []

        except Exception as e:
            logger.warning(f"[Features] nba_api获取{team}日志失败: {e}")
            return []

    def _load_league_game_logs(self):
        """批量从 LeagueGameLog 拉取全联盟比赛日志，按球队缓存。

        LeagueGameLog 有 PLUS_MINUS 列（TeamGameLog 没有），
        这是计算真实 net_rating 和 def_rating 的关键。
        """
        # 如果最近2小时内已加载过任何球队，跳过（全联盟一起加载的）
        if self._game_log_cache:
            any_ts = next(iter(self._game_log_cache.values()), {}).get("ts", 0)
            if time.time() - any_ts < 7200:
                return

        from nba_api.stats.endpoints import LeagueGameLog as LGL

        logger.info("[Features] 加载全联盟比赛日志(LeagueGameLog)...")
        log = LGL(
            season="2025-26",
            season_type_all_star="Regular Season",
            timeout=30,
        )
        df = log.get_data_frames()[0]

        # 按球队分组
        team_games = {}
        for _, row in df.iterrows():
            abbr = row.get("TEAM_ABBREVIATION", "")
            if not abbr:
                continue

            matchup = row.get("MATCHUP", "")
            is_home = "vs." in matchup

            fga = float(row.get("FGA", 1) or 1)
            fgm = float(row.get("FGM", 0) or 0)
            fg3m = float(row.get("FG3M", 0) or 0)
            fta = float(row.get("FTA", 0) or 0)
            tov = float(row.get("TOV", 0) or 0)
            oreb = float(row.get("OREB", 0) or 0)
            dreb = float(row.get("DREB", 0) or 0)
            mins = float(row.get("MIN", 240) or 240)

            efg = (fgm + 0.5 * fg3m) / max(fga, 1)
            tov_rate = tov / max(fga + 0.44 * fta + tov, 1)
            oreb_rate = oreb / max(oreb + dreb, 1)
            ft_rate = fta / max(fga, 1)

            raw_date = str(row.get("GAME_DATE", ""))
            try:
                parsed = datetime.strptime(raw_date, "%b %d, %Y")
                game_date_str = parsed.strftime("%Y-%m-%d")
            except Exception:
                game_date_str = raw_date[:10]

            # LeagueGameLog 有 PLUS_MINUS — 这是关键
            plus_minus = float(row.get("PLUS_MINUS", 0) or 0)
            pts_val = int(row.get("PTS", 0) or 0)
            opp_pts_val = pts_val - int(plus_minus)

            game_entry = {
                "date": game_date_str,
                "wl": row.get("WL", ""),
                "pts": pts_val,
                "opp_pts": opp_pts_val,
                "is_home": is_home,
                "plus_minus": plus_minus,
                "efg": round(efg, 4),
                "tov_rate": round(tov_rate, 4),
                "oreb_rate": round(oreb_rate, 4),
                "ft_rate": round(ft_rate, 4),
                "minutes": mins,
            }
            team_games.setdefault(abbr, []).append(game_entry)

        # 每队按日期排序（最新在前），只保留最近30场，缓存
        now = time.time()
        for abbr, games in team_games.items():
            games.sort(key=lambda g: g["date"], reverse=True)
            self._game_log_cache[abbr] = {"data": games[:30], "ts": now}

        logger.info(
            f"[Features] 全联盟日志加载完成: {len(team_games)}队, "
            f"{len(df)}条记录"
        )

    def _get_standings(self, team: str) -> Optional[dict]:
        """从数据库获取球队战绩。"""
        row = self.db.execute_one(
            "SELECT * FROM standings WHERE team_abbr=?", (team,)
        )
        return dict(row) if row else None

    def _fallback_features(self, home: str, away: str, game_date: str) -> dict:
        """当nba_api不可用时，用数据库中的standings做兜底。"""
        features = {}
        for prefix, team in [("home", home), ("away", away)]:
            s = self._get_standings(team)
            if s:
                total = (s.get("wins", 0) + s.get("losses", 0)) or 1
                wp = s.get("wins", 0) / total
                features[f"{prefix}_win_pct_10"] = wp
                features[f"{prefix}_net_rating_10"] = s.get("ppg_diff", 0)
                features[f"{prefix}_avg_pts_10"] = s.get("ppg", 105)
                features[f"{prefix}_efg_pct"] = 0.50
                features[f"{prefix}_tov_rate"] = 0.12
                features[f"{prefix}_oreb_rate"] = 0.25
                features[f"{prefix}_ft_rate"] = 0.20
                features[f"{prefix}_season_win_pct"] = wp
                features[f"{prefix}_season_net"] = s.get("ppg_diff", 0)
                features[f"{prefix}_venue_win_pct"] = wp
                features[f"{prefix}_schedule_density"] = 2
                features[f"{prefix}_rest_days"] = 1
                features[f"{prefix}_fatigue_minutes"] = 240
                features[f"{prefix}_efg_trend"] = 0
                features[f"{prefix}_injury_elo"] = 0
                features[f"{prefix}_off_rating"] = s.get("ppg", 105) / 100
                features[f"{prefix}_def_rating"] = s.get("opp_ppg", 105) / 100
                features[f"{prefix}_streak"] = 0
            else:
                for k in ["win_pct_10", "season_win_pct", "venue_win_pct"]:
                    features[f"{prefix}_{k}"] = 0.5
                for k in ["net_rating_10", "season_net", "injury_elo", "efg_trend", "streak"]:
                    features[f"{prefix}_{k}"] = 0
                features[f"{prefix}_avg_pts_10"] = 105
                features[f"{prefix}_efg_pct"] = 0.50
                features[f"{prefix}_tov_rate"] = 0.12
                features[f"{prefix}_oreb_rate"] = 0.25
                features[f"{prefix}_ft_rate"] = 0.20
                features[f"{prefix}_schedule_density"] = 2
                features[f"{prefix}_rest_days"] = 1
                features[f"{prefix}_fatigue_minutes"] = 240
                features[f"{prefix}_off_rating"] = 1.05
                features[f"{prefix}_def_rating"] = 1.05

        features["home_opp_strength"] = features.get("away_season_win_pct", 0.5)
        features["away_opp_strength"] = features.get("home_season_win_pct", 0.5)
        features["home_advantage"] = 1.0
        features["away_confidence_penalty"] = 0.0

        # OpenSkill 默认值
        features["home_skill_mu"] = 25.0
        features["home_skill_sigma"] = 8.333
        features["away_skill_mu"] = 25.0
        features["away_skill_sigma"] = 8.333
        features["skill_win_prob"] = 0.5

        # BBRef 默认值
        features["home_pace"] = 100.0
        features["home_ortg_bbref"] = 110.0
        features["home_drtg_bbref"] = 110.0
        features["away_pace"] = 100.0
        features["away_ortg_bbref"] = 110.0
        features["away_drtg_bbref"] = 110.0

        return features

    def _get_skill_features(self, home: str, away: str) -> dict:
        """获取 OpenSkill 贝叶斯技能评分特征。"""
        try:
            if self._team_ratings is None:
                from ml.team_ratings import TeamSkillRatings
                self._team_ratings = TeamSkillRatings(self.db)
            return self._team_ratings.get_features(home, away)
        except Exception as e:
            logger.debug(f"[Features] OpenSkill获取失败: {e}")
            return {
                "home_skill_mu": 25.0,
                "home_skill_sigma": 8.333,
                "away_skill_mu": 25.0,
                "away_skill_sigma": 8.333,
                "skill_win_prob": 0.5,
            }

    def _get_bbref_features(self, home: str, away: str) -> dict:
        """从 basketball_reference_web_scraper 获取高级数据。

        获取 Pace, ORtg, DRtg 等 nba_api 不提供的高级统计。
        4小时缓存（数据更新慢）。
        """
        defaults = {
            "home_pace": 100.0,
            "home_ortg_bbref": 110.0,
            "home_drtg_bbref": 110.0,
            "away_pace": 100.0,
            "away_ortg_bbref": 110.0,
            "away_drtg_bbref": 110.0,
        }

        # 检查缓存（4小时）
        if self._bbref_cache and time.time() - self._bbref_cache_ts < 14400:
            home_data = self._bbref_cache.get(home, {})
            away_data = self._bbref_cache.get(away, {})
            if home_data and away_data:
                return {
                    "home_pace": home_data.get("pace", 100.0),
                    "home_ortg_bbref": home_data.get("ortg", 110.0),
                    "home_drtg_bbref": home_data.get("drtg", 110.0),
                    "away_pace": away_data.get("pace", 100.0),
                    "away_ortg_bbref": away_data.get("ortg", 110.0),
                    "away_drtg_bbref": away_data.get("drtg", 110.0),
                }

        try:
            self._load_bbref_data()
            home_data = self._bbref_cache.get(home, {})
            away_data = self._bbref_cache.get(away, {})
            return {
                "home_pace": home_data.get("pace", 100.0),
                "home_ortg_bbref": home_data.get("ortg", 110.0),
                "home_drtg_bbref": home_data.get("drtg", 110.0),
                "away_pace": away_data.get("pace", 100.0),
                "away_ortg_bbref": away_data.get("ortg", 110.0),
                "away_drtg_bbref": away_data.get("drtg", 110.0),
            }
        except Exception as e:
            logger.debug(f"[Features] basketball-reference 获取失败: {e}")
            return defaults

    def _load_bbref_data(self):
        """批量加载 basketball-reference 赛季统计。"""
        try:
            from basketball_reference_web_scraper import client as bbref_client
            from basketball_reference_web_scraper.data import Team as BBRefTeam
        except ImportError:
            logger.debug("[Features] basketball_reference_web_scraper 未安装")
            return

        # BBRef Team enum → 我们的缩写映射
        bbref_to_abbr = {
            "ATLANTA_HAWKS": "ATL", "BOSTON_CELTICS": "BOS",
            "BROOKLYN_NETS": "BKN", "CHARLOTTE_HORNETS": "CHA",
            "CHICAGO_BULLS": "CHI", "CLEVELAND_CAVALIERS": "CLE",
            "DALLAS_MAVERICKS": "DAL", "DENVER_NUGGETS": "DEN",
            "DETROIT_PISTONS": "DET", "GOLDEN_STATE_WARRIORS": "GSW",
            "HOUSTON_ROCKETS": "HOU", "INDIANA_PACERS": "IND",
            "LOS_ANGELES_CLIPPERS": "LAC", "LOS_ANGELES_LAKERS": "LAL",
            "MEMPHIS_GRIZZLIES": "MEM", "MIAMI_HEAT": "MIA",
            "MILWAUKEE_BUCKS": "MIL", "MINNESOTA_TIMBERWOLVES": "MIN",
            "NEW_ORLEANS_PELICANS": "NOP", "NEW_YORK_KNICKS": "NYK",
            "OKLAHOMA_CITY_THUNDER": "OKC", "ORLANDO_MAGIC": "ORL",
            "PHILADELPHIA_76ERS": "PHI", "PHOENIX_SUNS": "PHX",
            "PORTLAND_TRAIL_BLAZERS": "POR", "SACRAMENTO_KINGS": "SAC",
            "SAN_ANTONIO_SPURS": "SAS", "TORONTO_RAPTORS": "TOR",
            "UTAH_JAZZ": "UTA", "WASHINGTON_WIZARDS": "WAS",
        }

        try:
            # 获取赛季总数据
            standings = bbref_client.standings(season_end_year=2026)

            for row in standings:
                team_name = row.get("team", "")
                # BBRef 返回的 team 字段是 Team enum 的 name
                if hasattr(team_name, "name"):
                    team_key = team_name.name
                else:
                    team_key = str(team_name).replace("Team.", "")

                abbr = bbref_to_abbr.get(team_key, "")
                if not abbr:
                    continue

                # 从 standings 提取基础数据；高级数据需要额外计算
                wins = row.get("wins", 0)
                losses = row.get("losses", 0)
                total = wins + losses

                # 尝试从字段获取高级数据
                # BBRef standings 可能包含不同字段，取决于版本
                pts_for = row.get("points_per_game", 0) or 0
                pts_against = row.get("opponent_points_per_game", 0) or 0

                # 估算 Pace 和 Rating
                # Pace ≈ 基于全联盟平均(100) + 球队特点的调整
                # 用进攻回合数近似：Pace ≈ (PTS + OPP_PTS) / 2 / 1.06
                if pts_for > 0 and pts_against > 0:
                    pace = (pts_for + pts_against) / 2 / 1.06
                    ortg = pts_for / pace * 100 if pace > 0 else 110
                    drtg = pts_against / pace * 100 if pace > 0 else 110
                else:
                    pace = 100.0
                    ortg = 110.0
                    drtg = 110.0

                self._bbref_cache[abbr] = {
                    "pace": round(pace, 1),
                    "ortg": round(ortg, 1),
                    "drtg": round(drtg, 1),
                }

            self._bbref_cache_ts = time.time()
            logger.info(f"[Features] BBRef数据加载: {len(self._bbref_cache)}队")

        except Exception as e:
            logger.warning(f"[Features] BBRef数据加载失败: {e}")

    def _get_cached(self, home: str, away: str, game_date: str) -> Optional[dict]:
        key = f"{away}@{home}"
        row = self.db.execute_one(
            "SELECT features_json FROM team_features WHERE team_abbr=? AND game_date=?",
            (key, game_date)
        )
        if row:
            try:
                return json.loads(row["features_json"])
            except Exception:
                pass
        return None

    def _save_cache(self, home: str, away: str, game_date: str, features: dict):
        key = f"{away}@{home}"
        try:
            self.db.insert(
                "INSERT OR REPLACE INTO team_features (team_abbr, game_date, features_json) VALUES (?,?,?)",
                (key, game_date, json.dumps(features))
            )
        except Exception:
            pass
