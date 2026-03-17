"""
game_memory.py — 长期比赛记忆（赛后复盘系统）

每场比赛结束后：
  1. 拉取box score（球员数据）
  2. 找出表现异常的球员（得分远超/远低于赛季均值）
  3. LLM分析为什么预测对/错
  4. 存入记忆库，下次遇到同队时自动调取

记忆格式示例：
  ❌ 2026-03-15 IND@MIL 买MIL:
  "Giannis赛季场均31.2分，本场仅得12分(轮休/伤病)，
   MIL缺少核心输出导致输球。教训：关注赛前轮休消息。"
"""

import json
import logging
import re
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class GameMemory:
    """长期比赛记忆系统。"""

    def __init__(self, db, minimax_key: str = "", minimax_url: str = "",
                 minimax_model: str = ""):
        self.db = db
        self.api_key = minimax_key
        self.api_url = minimax_url
        self.model = minimax_model
        self._ensure_table()

    def _ensure_table(self):
        try:
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS game_memory (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id         TEXT NOT NULL,
                    away_team       TEXT NOT NULL,
                    home_team       TEXT NOT NULL,
                    game_date       TEXT,
                    prediction_correct INTEGER,
                    our_buy_side    TEXT,
                    our_buy_price   REAL,
                    actual_score_home INTEGER,
                    actual_score_away INTEGER,
                    anomalies_json  TEXT,
                    insight         TEXT,
                    lesson_tags     TEXT,
                    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.db.execute("""
                CREATE INDEX IF NOT EXISTS idx_game_memory_teams
                    ON game_memory(away_team, home_team)
            """)
        except Exception:
            pass

    # ── 赛后复盘 ────────────────────────────────────────────────────
    def run_post_mortem(self, resolved_signals: list) -> int:
        """对已结算的推送信号做赛后复盘。

        Args:
            resolved_signals: signal_log中刚结算的行（dict列表）

        Returns:
            新增记忆数量
        """
        count = 0
        for sig in resolved_signals:
            if not sig.get("was_pushed"):
                continue

            game_id = sig.get("game_id", "")
            # 检查是否已有记忆
            existing = self.db.execute_one(
                "SELECT 1 FROM game_memory WHERE game_id=?", (game_id,)
            )
            if existing:
                continue

            try:
                # 1. 获取box score
                anomalies = self._fetch_anomalies(sig)

                # 2. LLM生成复盘洞察
                insight, tags = self._generate_insight(sig, anomalies)

                # 3. 存入记忆
                self.db.insert("""
                    INSERT INTO game_memory
                        (game_id, away_team, home_team, game_date,
                         prediction_correct, our_buy_side, our_buy_price,
                         actual_score_home, actual_score_away,
                         anomalies_json, insight, lesson_tags)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    game_id,
                    sig.get("away_team", ""),
                    sig.get("home_team", ""),
                    sig.get("game_date", ""),
                    sig.get("prediction_correct", 0),
                    sig.get("buy_side", ""),
                    sig.get("poly_price", 0),
                    sig.get("actual_score_home"),
                    sig.get("actual_score_away"),
                    json.dumps(anomalies, ensure_ascii=False),
                    insight,
                    tags,
                ))
                count += 1
                icon = "✅" if sig.get("prediction_correct") else "❌"
                logger.info(
                    f"[Memory] {icon} 复盘 {sig.get('away_team')}@{sig.get('home_team')}: "
                    f"{insight[:80]}"
                )

            except Exception as e:
                logger.warning(f"[Memory] 复盘失败 {game_id}: {e}")

        return count

    # ── 记忆检索 ────────────────────────────────────────────────────
    def get_relevant_memories(self, teams: list, limit: int = 5) -> list:
        """检索与给定球队相关的历史记忆。"""
        if not teams:
            return []

        placeholders = ",".join(["?"] * len(teams))
        rows = self.db.execute(f"""
            SELECT * FROM game_memory
            WHERE away_team IN ({placeholders})
               OR home_team IN ({placeholders})
            ORDER BY created_at DESC
            LIMIT ?
        """, tuple(teams) + tuple(teams) + (limit,))

        return [dict(r) for r in rows] if rows else []

    # ── Box Score 分析 ──────────────────────────────────────────────
    def _fetch_anomalies(self, sig: dict) -> list:
        """获取球员表现异常数据。"""
        anomalies = []
        try:
            from nba_api.stats.endpoints import BoxScoreTraditionalV2
            from ml.nba_features import _get_team_id

            # 需要NBA game_id（不是我们的game_id）
            # 尝试通过队名和日期匹配
            nba_game_id = self._find_nba_game_id(
                sig.get("home_team", ""),
                sig.get("away_team", ""),
                sig.get("game_date", "")
            )
            if not nba_game_id:
                return anomalies

            time.sleep(0.7)
            box = BoxScoreTraditionalV2(game_id=nba_game_id, timeout=15)
            player_stats = box.get_data_frames()[0]

            for _, row in player_stats.iterrows():
                pts = int(row.get("PTS", 0) or 0)
                min_played = row.get("MIN", "0")
                # 解析分钟
                try:
                    mins = float(str(min_played).split(":")[0]) if min_played else 0
                except (ValueError, TypeError):
                    mins = 0

                if mins < 10:  # 上场少于10分钟跳过
                    continue

                player = row.get("PLAYER_NAME", "Unknown")
                team = row.get("TEAM_ABBREVIATION", "")

                # 简单的异常检测：得分>30或<5（对于主要球员）
                if pts >= 35:
                    anomalies.append({
                        "player": player, "team": team,
                        "stat": "PTS", "value": pts,
                        "type": "超常发挥",
                    })
                elif mins >= 25 and pts <= 8:
                    anomalies.append({
                        "player": player, "team": team,
                        "stat": "PTS", "value": pts,
                        "minutes": mins,
                        "type": "表现低迷",
                    })

        except Exception as e:
            logger.debug(f"[Memory] Box score获取失败: {e}")

        return anomalies

    def _find_nba_game_id(self, home: str, away: str, game_date: str) -> Optional[str]:
        """通过队名和日期查找NBA game_id。"""
        try:
            from nba_api.stats.endpoints import LeagueGameLog

            time.sleep(0.7)
            log = LeagueGameLog(
                season="2025-26",
                season_type_all_star="Regular Season",
                timeout=15,
            )
            df = log.get_data_frames()[0]

            for _, row in df.iterrows():
                if str(row.get("GAME_DATE", ""))[:10] != game_date:
                    continue
                matchup = row.get("MATCHUP", "")
                team = row.get("TEAM_ABBREVIATION", "")
                if team == home and ("vs." in matchup):
                    return row.get("GAME_ID")
                if team == away and ("@" in matchup):
                    return row.get("GAME_ID")

        except Exception as e:
            logger.debug(f"[Memory] 查找game_id失败: {e}")

        return None

    # ── LLM复盘 ────────────────────────────────────────────────────
    def _generate_insight(self, sig: dict, anomalies: list) -> tuple:
        """LLM生成复盘洞察。返回 (insight_text, lesson_tags)。"""
        if not self.api_key:
            # 无LLM时用模板
            return self._template_insight(sig, anomalies), ""

        correct = sig.get("prediction_correct", 0)
        result_str = "预测正确✅" if correct else "预测错误❌"

        anomaly_text = ""
        if anomalies:
            anomaly_text = "球员表现异常:\n"
            for a in anomalies:
                anomaly_text += f"  - {a['player']}({a['team']}): {a['stat']}={a['value']} ({a['type']})\n"

        prompt = f"""你是NBA赛后分析师。简洁分析这场比赛预测的对错原因。

比赛: {sig.get('away_team')}@{sig.get('home_team')}
我们买入: {sig.get('buy_side', '?')}方 价格{sig.get('poly_price', 0):.0%}
比分: {sig.get('actual_score_away', '?')}-{sig.get('actual_score_home', '?')}
结果: {result_str}

{anomaly_text}

请输出2部分（总共不超过100字）：
1. 原因分析：为什么预测对了/错了（必须具体，提到球员或战术）
2. 教训标签：用逗号分隔的标签（如：轮休,伤病回归,背靠背疲劳,爆冷）"""

        try:
            resp = requests.post(
                f"{self.api_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 800,
                },
                timeout=60,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

            # 提取教训标签
            tags = ""
            tag_match = re.search(r'标签[：:]\s*(.+)', content)
            if tag_match:
                tags = tag_match.group(1).strip()

            return content[:500], tags

        except Exception as e:
            logger.warning(f"[Memory] LLM复盘失败: {e}")
            return self._template_insight(sig, anomalies), ""

    def _template_insight(self, sig: dict, anomalies: list) -> str:
        """无LLM时的模板复盘。"""
        correct = sig.get("prediction_correct", 0)
        away = sig.get("away_team", "?")
        home = sig.get("home_team", "?")
        score = f"{sig.get('actual_score_away', '?')}-{sig.get('actual_score_home', '?')}"

        text = f"{away}@{home} {score}. "
        if correct:
            text += "预测正确。"
        else:
            text += "预测错误。"

        if anomalies:
            for a in anomalies[:3]:
                text += f" {a['player']}({a['team']}){a['type']}({a['stat']}={a['value']})。"

        return text
