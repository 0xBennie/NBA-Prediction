"""
team_ratings.py — OpenSkill贝叶斯团队评分系统

用 OpenSkill (Weng-Lin / Plackett-Luce 模型) 为30支NBA球队计算
动态实力评分。比Elo更优：
  - 同时追踪 mu (实力均值) 和 sigma (不确定性)
  - 新赛季/阵容变动时 sigma 自动增大（不确定性上升）
  - 比赛多后 sigma 收窄（评分更确定）
  - 胜率预测: P(A>B) = Φ((mu_A - mu_B) / sqrt(sigma_A² + sigma_B²))

输出特征:
  - skill_mu: 队伍实力均值 (越高=越强)
  - skill_sigma: 不确定性 (越低=评分越确定)
  - skill_win_prob: 基于技能评分的预测胜率

依赖: openskill
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

RATINGS_PATH = Path("team_skill_ratings.json")

# NBA 30队
ALL_TEAMS = [
    "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET",
    "GSW", "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN",
    "NOP", "NYK", "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS",
    "TOR", "UTA", "WAS",
]


class TeamSkillRatings:
    """OpenSkill 团队实力评分。"""

    def __init__(self, db=None):
        self.db = db
        self.ratings = {}  # {team_abbr: {"mu": float, "sigma": float}}
        self._last_update = 0
        self._load_ratings()

    def get_team_rating(self, team: str) -> dict:
        """获取球队的技能评分。"""
        if team in self.ratings:
            return self.ratings[team]
        # 默认评分 (OpenSkill 默认 mu=25, sigma=25/3)
        return {"mu": 25.0, "sigma": 8.333}

    def predict_win_prob(self, home: str, away: str) -> float:
        """用 OpenSkill 评分预测主队胜率。

        P(home > away) = Φ((mu_h - mu_a) / sqrt(sigma_h² + sigma_a²))
        加上主场优势修正 (+1.5 mu)
        """
        import math

        h = self.get_team_rating(home)
        a = self.get_team_rating(away)

        # 主场优势: ~3分分差 ≈ mu+1.5
        home_advantage = 1.5

        mu_diff = (h["mu"] + home_advantage) - a["mu"]
        sigma_combined = math.sqrt(h["sigma"] ** 2 + a["sigma"] ** 2)

        if sigma_combined == 0:
            return 0.5

        # 标准正态CDF近似
        z = mu_diff / sigma_combined
        return _norm_cdf(z)

    def update_from_historical(self) -> dict:
        """从历史比赛数据更新所有球队评分。

        按时间顺序回放比赛，用 OpenSkill rate() 更新评分。
        """
        try:
            from openskill.models import PlackettLuce
        except ImportError:
            logger.warning("[TeamRatings] openskill 未安装，跳过评分更新")
            return {"status": "skipped", "reason": "openskill not installed"}

        if not self.db:
            return {"status": "skipped", "reason": "no database"}

        import sqlite3
        con = sqlite3.connect(
            self.db.db_path if hasattr(self.db, "db_path") else "nba_predictor.db",
            timeout=30,
        )
        con.row_factory = sqlite3.Row

        try:
            rows = con.execute("""
                SELECT home_team, away_team, home_won, game_date
                FROM historical_games
                WHERE home_won IS NOT NULL
                ORDER BY game_date ASC
            """).fetchall()
        except Exception as e:
            logger.warning(f"[TeamRatings] 查询历史比赛失败: {e}")
            con.close()
            return {"status": "error", "reason": str(e)}

        if not rows:
            con.close()
            return {"status": "skipped", "reason": "no historical games"}

        # 初始化 OpenSkill 模型
        model = PlackettLuce()

        # 初始化所有队伍评分
        team_ratings = {}
        for team in ALL_TEAMS:
            team_ratings[team] = model.rating()

        # 按时间顺序回放比赛
        games_processed = 0
        for row in rows:
            home = row["home_team"]
            away = row["away_team"]
            home_won = row["home_won"]

            if home not in team_ratings:
                team_ratings[home] = model.rating()
            if away not in team_ratings:
                team_ratings[away] = model.rating()

            # OpenSkill rate: 传入 [[winner], [loser]]
            if home_won:
                result = model.rate(
                    [[team_ratings[home]], [team_ratings[away]]]
                )
            else:
                result = model.rate(
                    [[team_ratings[away]], [team_ratings[home]]]
                )

            # 更新评分
            if home_won:
                team_ratings[home] = result[0][0]
                team_ratings[away] = result[1][0]
            else:
                team_ratings[away] = result[0][0]
                team_ratings[home] = result[1][0]

            games_processed += 1

        con.close()

        # 保存结果
        self.ratings = {}
        for team, rating in team_ratings.items():
            self.ratings[team] = {
                "mu": round(rating.mu, 4),
                "sigma": round(rating.sigma, 4),
            }

        self._save_ratings()
        self._last_update = time.time()

        # 排名输出
        sorted_teams = sorted(
            self.ratings.items(), key=lambda x: x[1]["mu"], reverse=True
        )
        top5 = ", ".join(f"{t}({r['mu']:.1f})" for t, r in sorted_teams[:5])
        logger.info(
            f"[TeamRatings] 更新完成 | {games_processed}场比赛 | "
            f"Top5: {top5}"
        )

        return {
            "status": "ok",
            "games_processed": games_processed,
            "teams_rated": len(self.ratings),
        }

    def get_features(self, home: str, away: str) -> dict:
        """返回用于预测模型的特征。"""
        h = self.get_team_rating(home)
        a = self.get_team_rating(away)
        win_prob = self.predict_win_prob(home, away)

        return {
            "home_skill_mu": h["mu"],
            "home_skill_sigma": h["sigma"],
            "away_skill_mu": a["mu"],
            "away_skill_sigma": a["sigma"],
            "skill_win_prob": win_prob,
        }

    def _load_ratings(self):
        """从文件加载评分。"""
        if RATINGS_PATH.exists():
            try:
                data = json.loads(RATINGS_PATH.read_text())
                self.ratings = data.get("ratings", {})
                self._last_update = data.get("updated_at", 0)
                logger.info(
                    f"[TeamRatings] 加载{len(self.ratings)}支球队评分"
                )
            except Exception as e:
                logger.warning(f"[TeamRatings] 加载评分失败: {e}")

    def _save_ratings(self):
        """保存评分到文件。"""
        try:
            RATINGS_PATH.write_text(json.dumps({
                "ratings": self.ratings,
                "updated_at": time.time(),
            }, ensure_ascii=False, indent=2))
        except Exception as e:
            logger.warning(f"[TeamRatings] 保存评分失败: {e}")

    def get_report(self) -> str:
        """生成评分报告。"""
        if not self.ratings:
            return "⚠️ 无球队评分数据，请先运行 --bootstrap"

        sorted_teams = sorted(
            self.ratings.items(), key=lambda x: x[1]["mu"], reverse=True
        )

        lines = ["━━━ OpenSkill 球队实力排名 ━━━", ""]
        for i, (team, r) in enumerate(sorted_teams, 1):
            bar = "█" * int(r["mu"] - 20)
            lines.append(
                f"  {i:2d}. {team}  μ={r['mu']:6.2f}  σ={r['sigma']:.2f}  {bar}"
            )
        return "\n".join(lines)


def _norm_cdf(z: float) -> float:
    """标准正态分布CDF近似（Abramowitz & Stegun）。"""
    import math
    if z > 6:
        return 1.0
    if z < -6:
        return 0.0
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = 1 if z >= 0 else -1
    z_abs = abs(z)
    t = 1.0 / (1.0 + p * z_abs)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-z_abs * z_abs / 2)
    return 0.5 * (1.0 + sign * y)
