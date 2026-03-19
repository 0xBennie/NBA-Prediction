"""
committee.py — 多Agent专家委员会

4个专家Agent各司其职，最终由决策者综合判断：
  1. 数据专家 — 量化数据汇总（战绩、伤病、状态）
  2. 对位专家 — 战术分析（风格匹配、关键球员）
  3. 风控专家 — 风险评估（价格、仓位、历史教训）
  4. 决策者   — 综合判断（买入/观望/等待 + 信心度）

调用时机（控制成本）：
  - 推送前：确认是否值得推送
  - Telegram提问：用户主动询问时
  - 赛后复盘：分析输赢原因
"""

import json
import logging
import re

import requests

logger = logging.getLogger(__name__)

# ── Agent System Prompts ──────────────────────────────────────────

DATA_AGENT_PROMPT = """你是NBA数据分析师。只汇总数字，不给意见。

输出格式（严格按此结构，不超过150字）：
【近况】主队近10场X胜X负，客队近10场X胜X负
【伤病】主队缺阵：XX(影响Elo-XX) / 客队缺阵：XX
【休息】主队休息X天 / 客队休息X天（背靠背标注）
【战绩】主队赛季XX胜XX负(X%)，客队赛季XX胜XX负(X%)
【主客】主队主场胜率X% / 客队客场胜率X%"""

MATCHUP_AGENT_PROMPT = """你是NBA战术分析专家。基于数据摘要，分析两队如何对位。

分析要点（不超过150字）：
1. 节奏匹配：快/慢节奏队伍碰撞的影响
2. 关键球员对位：谁是比赛X因素
3. 风格克制：进攻型vs防守型的历史规律
4. 给出倾向性判断：更看好哪一方，一句话理由"""

RISK_AGENT_PROMPT = """你是预测市场风控经理。评估这笔交易的风险。

你会收到数据摘要、战术分析、以及关于这两支球队的历史教训。

分析要点（不超过150字）：
1. 价格是否合理：当前市场价格 vs 我们的预测概率
2. 风险点：什么情况会让预测失败（轮休、关键球员状态波动）
3. 历史教训：过去类似情况的经验（如果有）
4. 仓位建议：建议入场价、止损价、仓位比例"""

DECISION_AGENT_PROMPT = """你是首席投资官。阅读3位分析师的报告后做最终决策。

你必须且只能输出以下JSON格式（不要输出其他内容）：
{
  "verdict": "buy或pass或wait",
  "confidence": 0.0到1.0之间的数字,
  "reasoning": "一句话决策理由",
  "entry_price": 建议入场价(小数),
  "stop_loss": 止损价(小数),
  "take_profit": 止盈价(小数)
}

规则：
- verdict="buy": 明确推荐买入，confidence>=0.6
- verdict="wait": 看好但价格不对，等回调
- verdict="pass": 不推荐，风险太大或没有优势
- 只有3位分析师中至少2位倾向同一方向时才给"buy"
- 如果任何分析师提出了重大风险警告，confidence打折"""


class ExpertCommittee:
    """4-Agent专家委员会。"""

    def __init__(self, minimax_key: str, minimax_url: str, minimax_model: str):
        self.api_key = minimax_key
        self.api_url = minimax_url
        self.model = minimax_model

    def deliberate(self, game_context: dict, memories: list = None) -> dict:
        """运行完整的4-Agent审议流程。

        Args:
            game_context: 比赛数据（由build_game_context构建）
            memories: 相关历史教训列表

        Returns:
            {"verdict": "buy/pass/wait", "confidence": 0.0-1.0,
             "reasoning": str, "entry_price": float, ...
             "full_analysis": str}  # 完整分析文本（用于Telegram推送）
        """
        if not self.api_key:
            return {"verdict": "pass", "confidence": 0, "reasoning": "AI未配置",
                    "full_analysis": ""}

        ctx_text = self._format_context(game_context)
        mem_text = self._format_memories(memories or [])

        # 1. 数据专家
        data_summary = self._call_agent(
            DATA_AGENT_PROMPT,
            f"比赛数据:\n{ctx_text}",
        )

        # 2. 对位专家
        matchup_analysis = self._call_agent(
            MATCHUP_AGENT_PROMPT,
            f"数据摘要:\n{data_summary}\n\n原始数据:\n{ctx_text}",
        )

        # 3. 风控专家
        risk_input = f"数据摘要:\n{data_summary}\n\n战术分析:\n{matchup_analysis}"
        if mem_text:
            risk_input += f"\n\n历史教训:\n{mem_text}"
        risk_input += f"\n\n市场数据:\n买入价:{game_context.get('buy_price', 'N/A')}"
        risk_input += f" 模型概率:{game_context.get('model_prob', 'N/A')}"
        risk_input += f" 价值边际:{game_context.get('value_edge', 'N/A')}"
        risk_input += f" Kelly:{game_context.get('kelly', 'N/A')}"

        risk_assessment = self._call_agent(RISK_AGENT_PROMPT, risk_input)

        # 4. 决策者
        decision_input = (
            f"【数据专家】\n{data_summary}\n\n"
            f"【对位专家】\n{matchup_analysis}\n\n"
            f"【风控专家】\n{risk_assessment}\n\n"
            f"关键数字: 模型预测{game_context.get('model_prob', '?')}, "
            f"市场价{game_context.get('buy_price', '?')}, "
            f"价值边际{game_context.get('value_edge', '?')}"
        )
        decision_raw = self._call_agent(DECISION_AGENT_PROMPT, decision_input, temperature=0.2)

        # 解析决策JSON
        verdict = self._parse_decision(decision_raw)

        # 组装完整分析
        verdict["full_analysis"] = (
            f"📊 <b>数据:</b> {data_summary[:200]}\n\n"
            f"⚔️ <b>对位:</b> {matchup_analysis[:200]}\n\n"
            f"⚠️ <b>风控:</b> {risk_assessment[:200]}"
        )

        logger.info(
            f"[Committee] 决策: {verdict['verdict']} "
            f"信心:{verdict['confidence']:.0%} — {verdict.get('reasoning', '')[:60]}"
        )

        return verdict

    def quick_analysis(self, question: str, context: str, memories: list = None) -> str:
        """快速分析模式（Telegram交互用，2次LLM调用而非4次）。"""
        mem_text = self._format_memories(memories or [])

        # 分析Agent（合并数据+对位）
        analysis = self._call_agent(
            "你是NBA交易顾问。结合数据和历史教训，给出简洁分析。不超过200字。",
            f"用户问题: {question}\n\n数据:\n{context}\n\n历史教训:\n{mem_text}",
        )

        # 决策Agent
        decision_raw = self._call_agent(
            DECISION_AGENT_PROMPT,
            f"分析:\n{analysis}\n\n用户问题: {question}",
            temperature=0.2,
        )
        verdict = self._parse_decision(decision_raw)

        result = f"{analysis}\n\n"
        if verdict["verdict"] == "buy":
            result += f"✅ <b>建议买入</b> (信心{verdict['confidence']:.0%})"
            if verdict.get("entry_price"):
                result += f"\n入场:{verdict['entry_price']:.0%}"
            if verdict.get("stop_loss"):
                result += f" 止损:{verdict['stop_loss']:.0%}"
            if verdict.get("take_profit"):
                result += f" 止盈:{verdict['take_profit']:.0%}"
        elif verdict["verdict"] == "wait":
            result += f"⏳ <b>建议等待</b> — {verdict.get('reasoning', '等价格回调')}"
        else:
            result += f"❌ <b>不推荐</b> — {verdict.get('reasoning', '风险过高')}"

        return result

    # ── LLM调用 ────────────────────────────────────────────────────
    def _call_agent(self, system_prompt: str, user_content: str,
                    temperature: float = 0.3, max_tokens: int = 1500) -> str:
        """调用单个Agent。"""
        for attempt in range(2):
            try:
                resp = requests.post(
                    f"{self.api_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_content[:3000]},
                        ],
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    },
                    timeout=60,
                )
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                return content[:1500]
            except requests.exceptions.Timeout:
                if attempt == 0:
                    continue
                return "[Agent超时]"
            except Exception as e:
                logger.warning(f"[Committee] Agent调用失败: {e}")
                return f"[Agent失败: {e}]"
        return "[Agent失败]"

    # ── 辅助方法 ────────────────────────────────────────────────────
    def _format_context(self, ctx: dict) -> str:
        """将game_context格式化为Agent可读文本。"""
        parts = []
        parts.append(f"比赛: {ctx.get('away_team', '?')} @ {ctx.get('home_team', '?')}")
        parts.append(f"买入方: {ctx.get('buy_team', '?')} ({ctx.get('buy_side', '?')})")
        parts.append(f"模型预测胜率: {ctx.get('model_prob', 0):.1%}")
        parts.append(f"Pinnacle公平值: {ctx.get('pinnacle_prob', 0):.1%}")
        parts.append(f"市场价格: {ctx.get('buy_price', 0):.1%}")
        parts.append(f"价值边际: +{ctx.get('value_edge', 0):.1%}")
        parts.append(f"评分: {ctx.get('score', 0)}/100")
        parts.append(f"Kelly仓位: {ctx.get('kelly', 0):.2%}")

        bd = ctx.get("breakdown", {})

        # 伤病（详细）
        inj = bd.get("injury_impact", {})
        if isinstance(inj, dict):
            parts.append(f"伤病Elo影响: 主队-{inj.get('home_elo_penalty', 0):.0f} / 客队-{inj.get('away_elo_penalty', 0):.0f}")

        # 具体伤病球员名单
        home_inj = ctx.get("home_injuries", {})
        away_inj = ctx.get("away_injuries", {})
        if home_inj and home_inj.get("all_injuries"):
            inj_list = ", ".join(
                f"{p['name']}({p['status']})" for p in home_inj["all_injuries"][:5]
            )
            star_note = f" ⚠️核心缺阵:{','.join(home_inj['star_out'])}" if home_inj.get("star_out") else ""
            parts.append(f"主队伤病: {inj_list}{star_note}")
        if away_inj and away_inj.get("all_injuries"):
            inj_list = ", ".join(
                f"{p['name']}({p['status']})" for p in away_inj["all_injuries"][:5]
            )
            star_note = f" ⚠️核心缺阵:{','.join(away_inj['star_out'])}" if away_inj.get("star_out") else ""
            parts.append(f"客队伤病: {inj_list}{star_note}")

        # 近期康复球员（伤愈归来=球队利好）
        recoveries = ctx.get("recent_recoveries", [])
        if recoveries:
            rec_list = ", ".join(
                f"{r['team_abbr']}-{r['player_name']}(原{r['old_status']})" for r in recoveries[:5]
            )
            parts.append(f"⭐近期康复归来: {rec_list}")

        # B2B
        b2b = bd.get("b2b", {})
        if isinstance(b2b, dict):
            if b2b.get("home_b2b"):
                parts.append("主队背靠背")
            if b2b.get("away_b2b"):
                parts.append("客队背靠背")

        # 特征数据
        feats = ctx.get("features", {})
        if feats:
            parts.append(f"主队近10场胜率: {feats.get('home_win_pct_10', 'N/A')}")
            parts.append(f"客队近10场胜率: {feats.get('away_win_pct_10', 'N/A')}")
            parts.append(f"主队净胜分: {feats.get('home_net_rating_10', 'N/A')}")
            parts.append(f"客队净胜分: {feats.get('away_net_rating_10', 'N/A')}")
            parts.append(f"主队主场胜率: {feats.get('home_venue_win_pct', 'N/A')}")
            parts.append(f"客队客场胜率: {feats.get('away_venue_win_pct', 'N/A')}")

        # Pinnacle一致性
        agree = bd.get("pinnacle_agreement", {})
        if isinstance(agree, dict):
            parts.append(f"Pinnacle一致: {'是' if agree.get('agrees') else '否'}")

        return "\n".join(parts)

    def _format_memories(self, memories: list) -> str:
        if not memories:
            return "无历史教训"
        lines = []
        for m in memories[:5]:
            icon = "✅" if m.get("prediction_correct") else "❌"
            lines.append(
                f"{icon} {m.get('game_date', '')} {m.get('away_team', '')}@{m.get('home_team', '')} "
                f"买{m.get('our_buy_side', '?')}: {m.get('insight', '')[:100]}"
            )
        return "\n".join(lines)

    def _parse_decision(self, raw: str) -> dict:
        """解析决策Agent的JSON输出。"""
        default = {"verdict": "pass", "confidence": 0.3, "reasoning": "解析失败"}

        # 尝试直接解析JSON
        try:
            # 提取JSON块
            match = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
            if match:
                data = json.loads(match.group())
                return {
                    "verdict": data.get("verdict", "pass"),
                    "confidence": float(data.get("confidence", 0.3)),
                    "reasoning": data.get("reasoning", ""),
                    "entry_price": data.get("entry_price"),
                    "stop_loss": data.get("stop_loss"),
                    "take_profit": data.get("take_profit"),
                }
        except (json.JSONDecodeError, ValueError):
            pass

        # JSON解析失败，从文本中提取
        lower = raw.lower()
        if "buy" in lower or "买入" in lower:
            default["verdict"] = "buy"
            default["confidence"] = 0.6
        elif "wait" in lower or "等待" in lower:
            default["verdict"] = "wait"
            default["confidence"] = 0.4
        default["reasoning"] = raw[:100]
        return default
