"""
telegram_bot.py — 交互式 Telegram AI 交易顾问

功能:
  用户在Telegram里发消息 → Bot实时查数据 → LLM生成回答 → 回复用户

示例对话:
  用户: "雄鹿今晚比赛怎么看？"
  Bot:  [查Polymarket价格+Pinnacle赔率+伤病+历史战绩] → LLM分析 → 回复

  用户: "我想买雄鹿，挂单挂多少合适？"
  Bot:  [查CLOB买卖价+spread+fair prob] → LLM给建议挂单价

  用户: "今天有什么好机会？"
  Bot:  [跑一次scan_games] → 返回最佳信号

  用户: "系统表现怎么样？"
  Bot:  [查signal_log统计] → 返回胜率/ROI/学习进度

依赖: python-telegram-bot >= 20.0 (已安装)
"""

import json
import logging
import os
import re
import threading
from datetime import datetime

import requests
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

logger = logging.getLogger(__name__)

# 队名别名 → 标准缩写
TEAM_ALIASES = {
    "雄鹿": "MIL", "bucks": "MIL", "密尔沃基": "MIL",
    "湖人": "LAL", "lakers": "LAL", "洛杉矶湖人": "LAL",
    "凯尔特人": "BOS", "celtics": "BOS", "波士顿": "BOS",
    "勇士": "GSW", "warriors": "GSW", "金州": "GSW",
    "掘金": "DEN", "nuggets": "DEN", "丹佛": "DEN",
    "太阳": "PHX", "suns": "PHX", "菲尼克斯": "PHX",
    "独行侠": "DAL", "mavericks": "DAL", "达拉斯": "DAL", "小牛": "DAL",
    "骑士": "CLE", "cavaliers": "CLE", "克利夫兰": "CLE",
    "尼克斯": "NYK", "knicks": "NYK", "纽约": "NYK",
    "热火": "MIA", "heat": "MIA", "迈阿密": "MIA",
    "雷霆": "OKC", "thunder": "OKC", "俄克拉荷马": "OKC",
    "76人": "PHI", "sixers": "PHI", "费城": "PHI",
    "快船": "LAC", "clippers": "LAC",
    "步行者": "IND", "pacers": "IND", "印第安纳": "IND",
    "灰熊": "MEM", "grizzlies": "MEM", "孟菲斯": "MEM",
    "国王": "SAC", "kings": "SAC", "萨克拉门托": "SAC",
    "鹈鹕": "NOP", "pelicans": "NOP", "新奥尔良": "NOP",
    "猛龙": "TOR", "raptors": "TOR", "多伦多": "TOR",
    "森林狼": "MIN", "timberwolves": "MIN", "明尼苏达": "MIN",
    "老鹰": "ATL", "hawks": "ATL", "亚特兰大": "ATL",
    "公牛": "CHI", "bulls": "CHI", "芝加哥": "CHI",
    "篮网": "BKN", "nets": "BKN", "布鲁克林": "BKN",
    "魔术": "ORL", "magic": "ORL", "奥兰多": "ORL",
    "马刺": "SAS", "spurs": "SAS", "圣安东尼奥": "SAS",
    "开拓者": "POR", "blazers": "POR", "波特兰": "POR",
    "火箭": "HOU", "rockets": "HOU", "休斯顿": "HOU",
    "爵士": "UTA", "jazz": "UTA", "犹他": "UTA",
    "活塞": "DET", "pistons": "DET", "底特律": "DET",
    "黄蜂": "CHA", "hornets": "CHA", "夏洛特": "CHA",
    "奇才": "WAS", "wizards": "WAS", "华盛顿": "WAS",
}


class TelegramAIBot:
    """交互式Telegram AI交易顾问。"""

    def __init__(self, scanner):
        """
        Args:
            scanner: main.py 的 Scanner 实例（共享数据源）
        """
        self.scanner = scanner
        self.db = scanner.db
        self.sb = scanner.sb
        self.espn = scanner.espn
        self.engine = scanner.engine
        self.ml = scanner.ml
        self.learner = scanner.learner
        self.committee = scanner.committee
        self.memory = scanner.memory
        self.features = scanner.features

        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        self.minimax_key = os.getenv("MINIMAX_API_KEY", "")
        self.minimax_url = os.getenv("MINIMAX_API_URL", "https://api.minimax.io/v1")
        self.minimax_model = os.getenv("MINIMAX_MODEL", "MiniMax-M2.5-highspeed")

    def start(self):
        """在后台线程启动Bot轮询。"""
        if not self.bot_token:
            logger.warning("[TGBot] 未配置TELEGRAM_BOT_TOKEN，交互式Bot未启动")
            return

        thread = threading.Thread(target=self._run, daemon=True)
        thread.start()
        logger.info("[TGBot] 交互式AI顾问已启动")

    def _run(self):
        """运行telegram bot（在独立线程+事件循环中）。"""
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._async_run())

    async def _async_run(self):
        app = Application.builder().token(self.bot_token).build()

        # 命令
        app.add_handler(CommandHandler("scan", self._cmd_scan))
        app.add_handler(CommandHandler("stats", self._cmd_stats))
        app.add_handler(CommandHandler("learn", self._cmd_learn))
        app.add_handler(CommandHandler("help", self._cmd_help))
        app.add_handler(CommandHandler("balance", self._cmd_balance))
        app.add_handler(CommandHandler("watchlist", self._cmd_watchlist))

        # 下单确认按钮回调
        app.add_handler(CallbackQueryHandler(self._handle_trade_callback))

        # 自由文本 → AI回答
        app.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            self._handle_message,
        ))

        logger.info("[TGBot] 开始轮询...")
        await app.initialize()
        await app.start()
        await app.updater.start_polling(drop_pending_updates=True)

        # 保持运行
        import asyncio
        while True:
            await asyncio.sleep(3600)

    # ── 命令处理 ────────────────────────────────────────────────────
    async def _cmd_help(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        text = (
            "🏀 <b>NBA交易顾问</b>\n\n"
            "直接发消息问我任何NBA交易问题：\n"
            "• \"雄鹿今晚怎么看？\"\n"
            "• \"我想买凯尔特人，挂单挂多少？\"\n"
            "• \"今天有什么好机会？\"\n"
            "• \"ATL vs ORL 分析一下\"\n\n"
            "<b>命令:</b>\n"
            "/scan — 立即扫描今日比赛\n"
            "/watchlist — 价格监控名单\n"
            "/balance — 交易账户+持仓\n"
            "/stats — 系统表现统计\n"
            "/learn — 自学习进度\n"
            "/help — 帮助"
        )
        await update.message.reply_text(text, parse_mode="HTML")

    async def _cmd_scan(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("🔄 正在扫描今日比赛...")
        try:
            from main import fetch_nba_games
            games = fetch_nba_games()
            if not games:
                await update.message.reply_text("暂无NBA比赛数据")
                return

            signals = []
            for game in games[:15]:  # 限制前15场
                result = self.engine.score_game(game)
                if result is None:
                    continue
                buy_side = result.get("buy_side", "away")
                buy_team = game["away_team"] if buy_side == "away" else game["home_team"]
                signals.append({
                    "matchup": f"{game['away_team']}@{game['home_team']}",
                    "buy_team": buy_team,
                    "score": result["score"],
                    "edge": result["edge"],
                    "price": result.get("poly_price", 0),
                    "kelly": result.get("kelly", 0),
                })

            signals.sort(key=lambda x: x["score"], reverse=True)
            msg = "📊 <b>今日比赛信号</b>\n━━━━━━━━━━━━━━━\n"
            for s in signals[:10]:
                icon = "🟢" if s["score"] >= 30 else "⚪"
                msg += (
                    f"{icon} {s['matchup']} 买{s['buy_team']}\n"
                    f"   评分:{s['score']} edge:{s['edge']:+.2%} "
                    f"价格:{s['price']:.0%} Kelly:{s['kelly']:.1%}\n"
                )
            await update.message.reply_text(msg, parse_mode="HTML")
        except Exception as e:
            await update.message.reply_text(f"扫描失败: {e}")

    async def _cmd_stats(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        try:
            stats = self.db.execute_one("""
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN prediction_correct=1 THEN 1 ELSE 0 END) as wins,
                       AVG(actual_roi) as avg_roi,
                       SUM(hypo_pnl) as total_pnl
                FROM signal_log WHERE prediction_correct IS NOT NULL
            """) or {}
            total = stats.get("total", 0) or 0
            wins = stats.get("wins", 0) or 0
            avg_roi = stats.get("avg_roi", 0) or 0

            pending = self.db.execute_one(
                "SELECT COUNT(*) as c FROM signal_log WHERE prediction_correct IS NULL"
            ) or {"c": 0}

            # 学习状态
            lp = self.learner.params.get("_meta", {})

            msg = (
                f"📈 <b>系统表现</b>\n━━━━━━━━━━━━━━━\n"
                f"已结算: {total}场\n"
                f"胜率: {wins}/{total} ({wins/total:.0%})\n" if total > 0 else
                f"📈 <b>系统表现</b>\n━━━━━━━━━━━━━━━\n"
                f"已结算: 0场\n"
            )
            if total > 0:
                msg += f"平均ROI: {avg_roi:+.1%}\n"
            msg += (
                f"待结算: {pending['c']}场\n"
                f"━━━━━━━━━━━━━━━\n"
                f"🧠 自学习 v{lp.get('version', 1)}\n"
                f"metric: {lp.get('baseline_metric', 'N/A')}\n"
                f"ML启用: {'✅' if self.ml.game_model else '❌'}"
            )
            await update.message.reply_text(msg, parse_mode="HTML")
        except Exception as e:
            await update.message.reply_text(f"统计失败: {e}")

    async def _cmd_learn(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        try:
            report = self.learner.get_learning_report()
            await update.message.reply_text(report, parse_mode="HTML")
        except Exception as e:
            await update.message.reply_text(f"学习报告失败: {e}")

    async def _cmd_watchlist(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """查看价格监控名单。"""
        watcher = getattr(self.scanner, 'watcher', None)
        if not watcher:
            await update.message.reply_text("价格监控未启用")
            return
        summary = watcher.get_watchlist_summary()
        try:
            await update.message.reply_text(summary, parse_mode="HTML")
        except Exception:
            await update.message.reply_text(summary)

    async def _cmd_balance(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """查看交易账户状态。"""
        trader = getattr(self.scanner, 'trader', None)
        if not trader:
            await update.message.reply_text("交易模块未启用")
            return
        s = trader.get_status()
        msg = (
            f"💰 <b>交易账户</b>\n"
            f"钱包: <code>{s.get('wallet', 'N/A')}</code>\n"
            f"设定资金: ${s['bankroll']:.0f}\n"
            f"今日已投: ${s['daily_invested']:.0f}\n"
            f"今日剩余: ${s['daily_remaining']:.0f}\n"
            f"单笔上限: ${s['max_single']:.0f}\n"
            f"CLOB连接: {'✅' if s['client_ready'] else '❌'}"
        )
        orders = s.get("open_orders", [])
        if orders:
            msg += f"\n\n📋 <b>挂单</b>\n"
            for o in orders:
                msg += f"  {o['side']} {o['market']} @{o['price']} x{o['size']} [{o['status']}]\n"
        await update.message.reply_text(msg, parse_mode="HTML")

    async def _handle_trade_callback(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """处理下单确认/取消按钮。"""
        query = update.callback_query
        await query.answer()

        data = query.data or ""
        trader = getattr(self.scanner, 'trader', None)

        if data.startswith("trade_confirm:"):
            # 格式: trade_confirm:token_id:price:amount:game_id:buy_team
            parts = data.split(":")
            if len(parts) < 6 or not trader:
                await query.edit_message_text("交易信息不完整或交易模块未启用")
                return

            token_id = parts[1]
            price = float(parts[2])
            amount = float(parts[3])
            game_id = parts[4]
            buy_team = parts[5]

            await query.edit_message_text(f"⏳ 正在下单: {buy_team} ${amount:.0f}@{price:.0%}...")

            result = trader.place_limit_order(
                token_id=token_id,
                side="BUY",
                price=price,
                amount=amount,
                game_id=game_id,
            )

            if result["success"]:
                msg = (
                    f"✅ <b>下单成功</b>\n"
                    f"买入: {buy_team} ${amount:.0f}@{price:.0%}\n"
                    f"订单: {result.get('order_id', '')[:20]}\n"
                    f"{result['message']}"
                )
            else:
                msg = f"❌ 下单失败: {result['message']}"

            await query.edit_message_text(msg, parse_mode="HTML")

        elif data.startswith("trade_skip:"):
            await query.edit_message_text("⏭ 已跳过此交易")

        elif data.startswith("trade_custom:"):
            await query.edit_message_text("📝 请直接发送金额，如: 买20")

    # ── 自由文本 → AI回答 ──────────────────────────────────────────
    async def _handle_message(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """处理用户自由文本消息，用LLM生成回答。"""
        if not update.message:
            return

        # 只响应目标chat
        if self.chat_id and str(update.message.chat_id) != str(self.chat_id):
            return

        # 提取所有文本内容（正文 + caption + 引用 + URL实体）
        parts = []

        msg_text = update.message.text or update.message.caption or ""
        if msg_text:
            parts.append(msg_text)

        # 提取消息中的URL（Polymarket链接等）
        entities = update.message.entities or update.message.caption_entities or []
        for ent in entities:
            if ent.type == "url":
                url_text = msg_text[ent.offset:ent.offset + ent.length]
                if url_text not in msg_text:
                    parts.append(url_text)
            elif ent.type == "text_link" and ent.url:
                parts.append(ent.url)

        # 如果是回复某条消息，把被回复的内容也加上
        if update.message.reply_to_message:
            reply_msg = update.message.reply_to_message
            reply_text = reply_msg.text or reply_msg.caption or ""
            if reply_text:
                parts.append(f"[引用消息]: {reply_text[:800]}")
            # 也提取引用消息中的URL
            reply_entities = reply_msg.entities or reply_msg.caption_entities or []
            for ent in reply_entities:
                if ent.type == "text_link" and ent.url:
                    parts.append(ent.url)

        user_msg = "\n".join(parts).strip()
        if not user_msg:
            return

        logger.info(f"[TGBot] 收到消息: {user_msg[:120]}")

        # 发送"正在输入"状态
        await update.message.chat.send_action("typing")

        # 1. 识别用户提到的球队
        teams = self._extract_teams(user_msg)
        logger.info(f"[TGBot] 识别球队: {teams}")

        # 2. 收集实时上下文
        context = self._build_context(teams, user_msg)

        # 3. 多Agent或单Agent回答
        await update.message.chat.send_action("typing")
        if teams and self.committee and self.committee.api_key:
            # 有球队 → 用专家委员会（2次LLM调用的快速模式）
            memories = self.memory.get_relevant_memories(teams) if self.memory else []
            reply = self.committee.quick_analysis(user_msg, context, memories)
        else:
            # 通用问题 → 单次LLM
            reply = self._ask_llm(user_msg, context)

        # 尝试HTML格式，失败则纯文本
        try:
            await update.message.reply_text(reply, parse_mode="HTML")
        except Exception:
            await update.message.reply_text(reply)

    def _extract_teams(self, text: str) -> list:
        """从用户消息中提取球队缩写。支持中文名、英文名、缩写、Polymarket链接。"""
        found = []
        lower = text.lower()
        upper = text.upper()

        # 1. 匹配3字母缩写 (DAL, NOP, MIL 等)
        import re
        all_abbrs = set(TEAM_ALIASES.values())
        for abbr in all_abbrs:
            if re.search(r'\b' + abbr + r'\b', upper) and abbr not in found:
                found.append(abbr)
            # 也匹配小写 (dal, nop)
            if re.search(r'\b' + abbr.lower() + r'\b', lower) and abbr not in found:
                found.append(abbr)

        # 2. 从Polymarket链接提取 (如 "pacers-vs-bucks")
        slug_match = re.findall(r'([a-z]+)-vs-([a-z]+)', lower)
        for away, home in slug_match:
            for alias, abbr in TEAM_ALIASES.items():
                if alias in away and abbr not in found:
                    found.append(abbr)
                if alias in home and abbr not in found:
                    found.append(abbr)

        # 3. 常规中英文球队名匹配
        for alias, abbr in TEAM_ALIASES.items():
            if alias in lower and abbr not in found:
                found.append(abbr)

        return found

    def _build_context(self, teams: list, user_msg: str) -> str:
        """根据用户提到的球队，收集实时数据作为LLM上下文。"""
        parts = []

        # 1. 今日比赛信号（如果提到了具体球队）
        if teams:
            try:
                from main import fetch_nba_games
                games = fetch_nba_games()
                for game in games:
                    if game["home_team"] in teams or game["away_team"] in teams:
                        result = self.engine.score_game(game)
                        if result:
                            buy_side = result.get("buy_side", "away")
                            buy_team = game["away_team"] if buy_side == "away" else game["home_team"]
                            bd = result.get("breakdown", {})
                            clob = bd.get("clob", {})
                            inj = bd.get("injury_impact", {})
                            lm = bd.get("line_movement", {})

                            parts.append(
                                f"【实时数据】{game['away_team']}@{game['home_team']}\n"
                                f"  Polymarket买入价: {result.get('poly_price',0):.1%}\n"
                                f"  Pinnacle公平值: {result.get('fair_prob',0):.1%} (来源:{result.get('source','')})\n"
                                f"  Edge: {result['edge']:+.2%} (有效:{result.get('effective_edge',0):+.2%})\n"
                                f"  建议买入方: {buy_team} ({buy_side})\n"
                                f"  评分: {result['score']}/100\n"
                                f"  Kelly仓位: {result.get('kelly',0):.2%}\n"
                                f"  CLOB: 买价{clob.get('buy_price','N/A')} 卖价{clob.get('sell_price','N/A')} 价差{clob.get('spread','N/A')}\n"
                                f"  伤病: 主队Elo-{inj.get('home_elo_penalty',0):.0f} 客队Elo-{inj.get('away_elo_penalty',0):.0f}\n"
                                f"  盘口移动: poly变化{lm.get('poly_shift',0):+.2%} 响应差{lm.get('discordance',0):+.2%}"
                            )
            except Exception as e:
                parts.append(f"[比赛数据获取失败: {e}]")

        # 2. 球队伤病详情
        for team in teams:
            try:
                impact = self.espn.get_injury_impact(team)
                if impact > 0:
                    parts.append(f"【{team}伤病】Elo惩罚: -{impact:.0f}")
            except Exception:
                pass

        # 3. 历史表现统计
        try:
            stats = self.db.execute_one("""
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN prediction_correct=1 THEN 1 ELSE 0 END) as wins,
                       AVG(actual_roi) as avg_roi
                FROM signal_log WHERE prediction_correct IS NOT NULL
            """) or {}
            total = stats.get("total", 0) or 0
            wins = stats.get("wins", 0) or 0
            avg_roi = stats.get("avg_roi", 0) or 0
            if total > 0:
                parts.append(
                    f"【系统历史表现】{total}场已结算 胜率{wins/total:.0%} 平均ROI{avg_roi:+.1%}"
                )

            # 特定球队的历史
            for team in teams:
                team_stats = self.db.execute_one("""
                    SELECT COUNT(*) as c,
                           SUM(CASE WHEN prediction_correct=1 THEN 1 ELSE 0 END) as w,
                           AVG(actual_roi) as r
                    FROM signal_log
                    WHERE (buy_team=? OR home_team=? OR away_team=?)
                      AND prediction_correct IS NOT NULL
                """, (team, team, team)) or {}
                tc = team_stats.get("c", 0) or 0
                if tc > 0:
                    tw = team_stats.get("w", 0) or 0
                    tr = team_stats.get("r", 0) or 0
                    parts.append(f"【{team}历史】{tc}场 胜率{tw/tc:.0%} ROI{tr:+.1%}")
        except Exception:
            pass

        # 4. 历史教训（长期记忆）
        if teams and self.memory:
            try:
                memories = self.memory.get_relevant_memories(teams, limit=3)
                for m in memories:
                    icon = "✅" if m.get("prediction_correct") else "❌"
                    parts.append(
                        f"【历史教训】{icon} {m.get('game_date','')} "
                        f"{m.get('away_team','')}@{m.get('home_team','')} "
                        f"{m.get('insight', '')[:120]}"
                    )
            except Exception:
                pass

        # 5. 自学习参数
        try:
            lp = self.learner.params
            pred = lp.get("prediction", {})
            parts.append(
                f"【系统参数】模型权重:{pred.get('model_weight','?')} "
                f"信心门槛:{pred.get('min_model_confidence','?')} "
                f"价值门槛:{pred.get('min_value_edge','?')}"
            )
        except Exception:
            pass

        return "\n\n".join(parts) if parts else "暂无实时数据"

    def _ask_llm(self, user_msg: str, context: str) -> str:
        """调用LLM生成回答，带重试。"""
        if not self.minimax_key:
            return f"⚠️ 未配置AI模型\n\n以下是收集到的数据:\n{context}"

        system_prompt = """你是一个专业的Polymarket NBA交易顾问AI。你有以下能力：
1. 根据实时数据给出交易建议（买入/卖出/观望）
2. 根据CLOB价差建议最佳挂单价格
3. 分析伤病、盘口移动对比赛的影响
4. 基于Kelly公式建议仓位大小

回复规则：
- 简洁直接，不超过200字
- 如果用户问挂单价格，给出具体数字（基于CLOB买卖价和fair value）
- 如果用户问某场比赛，结合评分、edge、伤病给出明确建议
- 如果数据不足以做判断，诚实说明
- 用中文回复
- 不要输出思考过程，直接给结论"""

        # 截断context避免过长
        if len(context) > 2000:
            context = context[:2000] + "\n...(数据已截断)"

        for attempt in range(2):
            try:
                resp = requests.post(
                    f"{self.minimax_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.minimax_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.minimax_model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"用户问题: {user_msg}\n\n实时数据:\n{context}"},
                        ],
                        "temperature": 0.3,
                        "max_tokens": 2000,  # 推理模型需要更多token（思考+回答）
                    },
                    timeout=90,
                )
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                return content[:1000]
            except requests.exceptions.Timeout:
                if attempt == 0:
                    logger.warning("[TGBot] LLM超时，重试中...")
                    continue
                logger.warning("[TGBot] LLM两次超时")
                return f"⚠️ AI响应超时，以下是原始数据:\n\n{context[:800]}"
            except Exception as e:
                logger.warning(f"[TGBot] LLM调用失败: {e}")
                return f"⚠️ AI回复失败: {e}\n\n原始数据:\n{context[:800]}"
        return context[:800]
