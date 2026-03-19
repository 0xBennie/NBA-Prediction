"""
main.py — 主扫描器 + Telegram推送

环境变量（.env 或直接设置）:
  TELEGRAM_BOT_TOKEN   必须
  TELEGRAM_CHAT_ID     必须
  ODDS_API_KEY         可选（有则用Pinnacle，无则走下级fallback）
  MINIMAX_API_KEY      可选（有则用AI分析）

运行:
  python main.py              # 启动持续扫描
  python main.py --once       # 只扫描一次
  python main.py --health     # 数据源健康检查
  python main.py --report     # ML表现报告
"""

import os
import json
import re
import time
import logging
import argparse
import requests
import schedule
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from core.database import Database
from core.sportsbook_client import SportsbookClient
from core.espn_client import ESPNClient
from core.clob_client import get_market_prices
from core.mismatch_engine import MismatchEngine
from ml.ml_predictor import MLPredictor
from ml.result_resolver import ResultResolver
from ml.auto_learner import AutoLearner
from ml.game_predictor import GamePredictor
from ml.nba_features import NBAFeatureBuilder
from ml.game_memory import GameMemory
from core.committee import ExpertCommittee
from core.telegram_bot import TelegramAIBot
from core.notion_sync import NotionSync
from core.auto_trader import AutoTrader
from core.price_watcher import PriceWatcher
from core.injury_checker import InjuryChecker
from ml.player_ratings import update_player_ratings
from ml.team_ratings import TeamSkillRatings
from ml.backtester import Backtester

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── 配置 ──────────────────────────────────────────────────────────
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
ODDS_KEY = os.getenv("ODDS_API_KEY", "")
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY", "")
MINIMAX_API_URL = os.getenv("MINIMAX_API_URL", "https://api.minimax.io/v1")
MINIMAX_MODEL = os.getenv("MINIMAX_MODEL", "MiniMax-M2.5-highspeed")
NOTION_TOKEN = os.getenv("NOTION_TOKEN", "")
NOTION_DB_ID = os.getenv("NOTION_DB_ID", "")
POLY_PRIVATE_KEY = os.getenv("POLY_PRIVATE_KEY", "")

POLYMARKET_GAMMA = "https://gamma-api.polymarket.com"
NBA_TAG_ID = 745  # Polymarket NBA sport tag
POLY_FEE_RATE = 0.01  # 交易费率（与mismatch_engine一致）


# ── Telegram推送 ───────────────────────────────────────────────────
def send_telegram(text: str):
    if not BOT_TOKEN or not CHAT_ID:
        logger.warning("[TG] 未配置TOKEN/CHAT_ID，跳过推送")
        print(f"\n{'='*50}\n{text}\n{'='*50}")
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception as e:
        logger.error(f"[TG] 发送失败: {e}")


def format_game_message(game: dict, result: dict, slug: str = "") -> str:
    score = result["score"]
    buy_side = result.get("buy_side", "away")
    poly = result.get("poly_price", 0)
    model_prob = result.get("model_prob", 0)
    blended = result.get("blended_prob", result.get("fair_prob", 0))
    value_edge = result.get("effective_edge", 0)
    source = result["source"]
    kelly = result["kelly"]
    home_b2b = "🔴B2B" if result.get("home_b2b") else ""
    away_b2b = "🔴B2B" if result.get("away_b2b") else ""

    if buy_side == "away":
        buy_team = game["away_team"]
        buy_label = "客队"
    else:
        buy_team = game["home_team"]
        buy_label = "主队"

    bd = result["breakdown"]
    inj = bd.get("injury_impact", {})
    clob = bd.get("clob", {})
    has_clob = bd.get("has_clob", False)
    agreement = bd.get("pinnacle_agreement", {})

    # 盈亏比
    win_return = (1.0 / poly - 1.0) if poly > 0 else 0

    msg = (
        f"🏀 <b>NBA比赛预测信号</b>\n"
        f"━━━━━━━━━━━━━━━\n"
        f"📍 {game['away_team']} {away_b2b} @ {game['home_team']} {home_b2b}\n"
        f"🔮 模型预测: <b>{buy_team}胜 {model_prob:.1%}</b>\n"
        f"💰 市场价格: {poly:.1%}"
    )
    if has_clob:
        msg += f" (买:{clob.get('buy_price',0):.1%} 卖:{clob.get('sell_price',0):.1%})"
    msg += (
        f"\n📈 价值边际: <b>+{value_edge:.1%}</b>\n"
        f"📊 Pinnacle: {result.get('fair_prob',0):.1%}"
        f" {'✅一致' if agreement.get('agrees') else '⚠️不一致'}\n"
        f"━━━━━━━━━━━━━━━\n"
        f"🏆 评分: <b>{score}/100</b>\n"
        f"⚖️ 赔率: 赢<b>+{win_return:.0%}</b> / 输-100%\n"
        f"💡 Kelly仓位: <b>{kelly:.1%}</b>\n"
    )
    if inj.get("delta", 0) != 0:
        msg += f"🏥 伤病: 主队-{inj.get('home_elo_penalty',0):.0f} / 客队-{inj.get('away_elo_penalty',0):.0f}Elo\n"

    if slug:
        msg += f"\n🔗 <a href=\"https://polymarket.com/event/{slug}\">Polymarket</a>\n"

    msg += f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    return msg


def format_futures_message(market: dict, result: dict) -> str:
    score = result["score"]
    bd = result["breakdown"]
    model_prob = result.get("model_prob", 0)
    value_edge = result.get("value_edge", 0)
    momentum = bd.get("momentum", {})
    price = market.get("price", 0)
    win_return = (1.0 / price - 1.0) if price > 0 else 0
    days = market.get("days_to_resolution", 0)
    slug = market.get("slug", "")

    TYPE_CN = {"champion": "总冠军", "mvp": "MVP", "division": "赛区冠军",
               "playoff": "季后赛", "roy": "最佳新秀"}
    type_label = TYPE_CN.get(market.get("type", ""), market.get("type", "").upper())

    msg = (
        f"📅 <b>NBA期货预测信号</b>\n"
        f"━━━━━━━━━━━━━━━\n"
        f"📍 {market.get('team', 'N/A')} — {type_label}\n"
        f"🔮 预测概率: <b>{model_prob:.1%}</b>\n"
        f"💰 市场价格: <b>{price:.1%}</b>\n"
        f"📈 价值边际: <b>+{value_edge:.1%}</b>\n"
        f"━━━━━━━━━━━━━━━\n"
        f"🏆 评分: <b>{score}/100</b>\n"
        f"⚖️ 赔率: 赢<b>+{win_return:.0%}</b> / 输-100%\n"
        f"💡 Kelly仓位: <b>{result['kelly']:.1%}</b>\n"
        f"📊 动量: 日{momentum.get('daily_change', 0):+.1%} 周{momentum.get('weekly_change', 0):+.1%}\n"
        f"⏳ 结算: {days}天后\n"
    )

    if slug:
        msg += f"\n🔗 <a href=\"https://polymarket.com/event/{slug}\">Polymarket</a>\n"

    msg += f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    return msg


# ── Polymarket数据获取 ─────────────────────────────────────────────
TEAM_NAME_MAP = {
    "jazz": "UTA", "trail blazers": "POR", "blazers": "POR",
    "lakers": "LAL", "warriors": "GSW", "celtics": "BOS",
    "nets": "BKN", "knicks": "NYK", "heat": "MIA",
    "bucks": "MIL", "nuggets": "DEN", "suns": "PHX",
    "clippers": "LAC", "thunder": "OKC", "rockets": "HOU",
    "mavericks": "DAL", "spurs": "SAS", "grizzlies": "MEM",
    "pelicans": "NOP", "hawks": "ATL", "hornets": "CHA",
    "bulls": "CHI", "cavaliers": "CLE", "pistons": "DET",
    "pacers": "IND", "magic": "ORL", "76ers": "PHI", "sixers": "PHI",
    "raptors": "TOR", "wizards": "WAS", "kings": "SAC",
    "timberwolves": "MIN",
}


def _abbr(team_name: str) -> str:
    lower = team_name.lower().strip()
    for k, v in TEAM_NAME_MAP.items():
        if k in lower:
            return v
    return team_name.upper()[:3]


def _extract_team(question: str) -> str:
    for team_key in TEAM_NAME_MAP:
        if team_key in question.lower():
            return TEAM_NAME_MAP[team_key]
    return ""


def _days_to_close(end_date: str) -> int:
    if not end_date:
        return 999
    try:
        ed = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
        return max(0, (ed.date() - datetime.now().date()).days)
    except Exception:
        return 999


def _is_moneyline(question: str, title: str) -> bool:
    """判断market是否是胜负线(moneyline)。
    Polymarket NBA胜负线的question要么等于event title，
    要么包含 "will X win" / "moneyline"。
    排除spread/total/props等。
    """
    q = question.lower().strip()
    t = title.lower().strip()

    # 排除非胜负线
    skip_kw = ["spread", "total", "o/u", "points", "assists", "rebounds",
               "steals", "blocks", "1h ", "1h:", "player", "prop", "+"]
    if any(kw in q for kw in skip_kw):
        return False

    # 胜负线: question和title相同（如 "Cavaliers vs. Mavericks"）
    if q == t:
        return True
    # 或包含 "moneyline"
    if "moneyline" in q:
        return True
    # 或包含 "will X win"
    if "will " in q and " win" in q:
        return True

    return False


def fetch_nba_games() -> list:
    """从Polymarket拉取今日NBA比赛盘（胜负线）"""
    try:
        r = requests.get(
            f"{POLYMARKET_GAMMA}/events",
            params={"tag_id": NBA_TAG_ID, "limit": 200, "active": "true",
                    "closed": "false"},
            timeout=15,
        )
        r.raise_for_status()
        events = r.json()
    except Exception as e:
        logger.error(f"[Polymarket] 比赛拉取失败: {e}")
        return []

    games = []
    for event in events:
        title = event.get("title", "")
        slug = event.get("slug", "")
        if " vs " not in title.lower() and " vs." not in title.lower():
            continue

        for market in event.get("markets", []):
            q = market.get("question", "")
            if not _is_moneyline(q, title):
                continue

            outcomes = market.get("outcomePrices", "[]")
            try:
                prices = json.loads(outcomes) if isinstance(outcomes, str) else outcomes
                prices = [float(p) for p in prices]
            except Exception:
                continue

            if len(prices) < 2:
                continue

            # 跳过已结算（价格0/1）或无成交量的市场
            if prices[0] <= 0.01 or prices[0] >= 0.99:
                continue

            # 解析clobTokenIds
            try:
                clob_tokens = json.loads(market.get("clobTokenIds", "[]"))
            except Exception:
                clob_tokens = []

            # Polymarket NBA ordering=away: 第一个队是客队
            t = title.lower()
            if " vs." in t:
                parts = title.split(" vs.")
            elif " vs " in t:
                parts = title.split(" vs ")
            else:
                continue

            if len(parts) < 2:
                continue

            away_raw = parts[0].strip()
            home_raw = parts[1].strip()

            game_data = {
                "game_id": f"{_abbr(away_raw)}_{_abbr(home_raw)}_{datetime.now().strftime('%Y%m%d')}",
                "home_team": _abbr(home_raw),
                "away_team": _abbr(away_raw),
                "polymarket_price_away": prices[0],
                "volume_24h": float(market.get("volume24hr", 0) or 0),
                "game_date": datetime.now().strftime("%Y-%m-%d"),
                "condition_id": market.get("conditionId", ""),
                "slug": slug,
                "clob_token_ids": clob_tokens,
            }

            # 获取CLOB真实买入/卖出价
            if clob_tokens and len(clob_tokens) >= 2:
                clob_prices = get_market_prices(clob_tokens)
                if clob_prices["valid"]:
                    game_data["clob"] = clob_prices
                    # 用真实buy价格(ask)替代Gamma mid-price
                    if clob_prices.get("away_buy") is not None:
                        game_data["polymarket_price_away"] = clob_prices["away_buy"]
                    logger.info(
                        f"[CLOB] {_abbr(away_raw)}@{_abbr(home_raw)} "
                        f"away buy={clob_prices.get('away_buy', 'N/A')} "
                        f"sell={clob_prices.get('away_sell', 'N/A')} "
                        f"spread={clob_prices.get('away_spread', 'N/A')}"
                    )

            games.append(game_data)

    return games


def fetch_nba_futures(db: Database) -> list:
    """从Polymarket拉取NBA期货盘"""
    try:
        r = requests.get(
            f"{POLYMARKET_GAMMA}/events",
            params={"tag_id": NBA_TAG_ID, "limit": 200, "active": "true",
                    "closed": "false"},
            timeout=15,
        )
        r.raise_for_status()
        events = r.json()
    except Exception as e:
        logger.error(f"[Polymarket] 期货拉取失败: {e}")
        return []

    futures = []
    for event in events:
        for market in event.get("markets", []):
            q = market.get("question", "").lower()
            if "champion" in q or "nba finals" in q:
                mtype = "champion"
            elif "mvp" in q:
                mtype = "mvp"
            elif "division" in q:
                mtype = "division"
            elif "playoff" in q:
                mtype = "playoff"
            elif "rookie of the year" in q or "roy" in q:
                mtype = "roy"
            else:
                continue

            try:
                prices = json.loads(market.get("outcomePrices", "[]"))
                price = float(prices[0]) if prices else 0
            except Exception:
                continue

            cid = market.get("conditionId", "")
            hist = _get_price_history(db, cid)

            # 获取CLOB真实价格
            bid_ask_spread = None
            try:
                clob_tokens = json.loads(market.get("clobTokenIds", "[]"))
                if clob_tokens:
                    from core.clob_client import get_price
                    buy_p = get_price(clob_tokens[0], "buy")
                    sell_p = get_price(clob_tokens[0], "sell")
                    if buy_p is not None and sell_p is not None:
                        bid_ask_spread = round(buy_p - sell_p, 4)
                        price = buy_p  # 用真实买入价
            except Exception:
                pass

            futures.append({
                "condition_id": cid,
                "type": mtype,
                "team": _extract_team(market.get("question", "")),
                "question": market.get("question", ""),
                "price": price,
                "volume": float(market.get("volume", 0) or 0),
                "volume_24h": float(market.get("volume24hr", 0) or 0),
                "days_to_resolution": _days_to_close(market.get("endDate", "")),
                "bid_ask_spread": bid_ask_spread,
                "price_24h_ago": hist.get("price_24h_ago"),
                "price_7d_ago": hist.get("price_7d_ago"),
                "slug": event.get("slug", ""),
            })

    return futures


def _get_price_history(db: Database, condition_id: str) -> dict:
    now = int(time.time())
    row_24h = db.execute_one("""
        SELECT poly_price_away FROM price_history
        WHERE condition_id=? AND timestamp BETWEEN ? AND ?
        ORDER BY ABS(timestamp - ?) LIMIT 1
    """, (condition_id, now - 90000, now - 72000, now - 86400))

    row_7d = db.execute_one("""
        SELECT poly_price_away FROM price_history
        WHERE condition_id=? AND timestamp BETWEEN ? AND ?
        ORDER BY ABS(timestamp - ?) LIMIT 1
    """, (condition_id, now - 648000, now - 518400, now - 604800))

    return {
        "price_24h_ago": dict(row_24h).get("poly_price_away") if row_24h else None,
        "price_7d_ago": dict(row_7d).get("poly_price_away") if row_7d else None,
    }


# ── 主扫描逻辑 ─────────────────────────────────────────────────────
class Scanner:

    def __init__(self):
        self.db = Database()
        self.sb = SportsbookClient(odds_api_key=ODDS_KEY)
        self.espn = ESPNClient(self.db)
        self.ml = MLPredictor(db_path=self.db.db_path)
        self.predictor = GamePredictor(db_path=self.db.db_path)
        self.features = NBAFeatureBuilder(self.db, self.espn)
        self.engine = MismatchEngine(self.db, self.sb, self.espn, self.ml,
                                     predictor=self.predictor, features=self.features)
        self.resolver = ResultResolver(self.db, self.ml, odds_api_key=ODDS_KEY)
        self.learner = AutoLearner(db_path=self.db.db_path)
        self.committee = ExpertCommittee(MINIMAX_API_KEY, MINIMAX_API_URL, MINIMAX_MODEL)
        self.memory = GameMemory(self.db, MINIMAX_API_KEY, MINIMAX_API_URL, MINIMAX_MODEL)
        self.notion = NotionSync(NOTION_TOKEN, NOTION_DB_ID) if NOTION_TOKEN else None
        self.trader = AutoTrader(POLY_PRIVATE_KEY, "", "",
                                 bankroll=1000.0) if POLY_PRIVATE_KEY else None
        self.watcher = PriceWatcher(self.db)
        self.injury_checker = InjuryChecker(self.espn)
        self.daily_push_count = 0
        self.DAILY_LIMIT = 10
        # 候选池: {game_id: {"game": ..., "result": ..., "best_score": ..., "scans": 0}}
        self._candidates = {}

    def scan_games(self):
        """扫描比赛盘 — 积累候选，不立即推送。

        逻辑:
          每次扫描 → 更新候选池（保留最优评分）
          推送由 _push_best_candidates() 在最佳时机触发
        """
        logger.info("[Scanner] 开始扫描比赛盘...")
        games = fetch_nba_games()
        logger.info(f"[Scanner] 获取到 {len(games)} 场比赛")

        for game in games:
            gid = game["game_id"]

            # 已推送过的跳过
            exists = self.db.execute_one(
                "SELECT 1 FROM alerted_games WHERE game_id=?", (gid,)
            )
            if exists:
                continue

            result = self.engine.score_game(game)
            if result is None:
                continue

            buy_side = result.get("buy_side", "away")
            buy_team = game["away_team"] if buy_side == "away" else game["home_team"]
            logger.info(
                f"[Scanner] {game['away_team']}@{game['home_team']} "
                f"买{buy_side}({buy_team}) 评分:{result['score']} edge:{result['edge']:+.2%}"
            )

            # 记录到回测日志
            self._log_signal(game, result)

            # 所有有模型预测的比赛都加入价格监控
            model_prob = result.get("model_prob", 0)
            if model_prob >= 0.55:
                self.watcher.update_watchlist(
                    game, model_prob, buy_side,
                    result.get("poly_price", 0),
                )

            # 更新候选池: 所有有潜力的信号都追踪（不只是当前达到push条件的）
            # 这样如果一场比赛第一次扫描没达标，第二次达标了，也能被推送
            if result["push"] or result["score"] >= 20:
                prev = self._candidates.get(gid)
                if prev is None or result["score"] > prev["best_score"]:
                    self._candidates[gid] = {
                        "game": game,
                        "result": result,
                        "best_score": result["score"],
                        "buy_team": buy_team,
                        "buy_side": buy_side,
                        "scans": (prev["scans"] + 1) if prev else 1,
                    }
                    if prev:
                        logger.info(
                            f"[Scanner] {game['away_team']}@{game['home_team']} "
                            f"候选更新: {prev['best_score']} → {result['score']}"
                        )
                else:
                    self._candidates[gid]["scans"] = prev["scans"] + 1

        # 扫描完后，推送成熟的候选（扫描≥2次 = 至少观察了2小时）
        self._push_best_candidates()

        # 清理过期监控
        self.watcher.cleanup_old()

    def _push_best_candidates(self):
        """推送最优候选 — 一场比赛只推一次。

        条件:
          - 扫描≥2次（至少观察2小时，确认信号稳定）
          - 按评分排序，推最好的
          - 每场比赛只推一次
        """
        # 筛选成熟候选（扫描≥2次 + 最终结果达到push条件）
        ready = [
            (gid, c) for gid, c in self._candidates.items()
            if c["scans"] >= 2 and c["result"].get("push", False)
        ]
        if not ready:
            return

        # 按评分排序
        ready.sort(key=lambda x: x[1]["best_score"], reverse=True)

        for gid, candidate in ready:
            if self.daily_push_count >= self.DAILY_LIMIT:
                break

            game = candidate["game"]
            result = candidate["result"]
            buy_side = candidate["buy_side"]
            buy_team = candidate["buy_team"]

            # 伤病检查 — 推送前联网查最新伤病
            try:
                inj_check = self.injury_checker.check_team(buy_team)
                if inj_check["has_star_out"]:
                    penalty = inj_check["injury_penalty"]
                    original_prob = result.get("model_prob", 0)
                    adjusted_prob = original_prob - penalty
                    logger.info(
                        f"⚠️ 伤病警告: {buy_team} {inj_check['details']} "
                        f"模型{original_prob:.0%}→{adjusted_prob:.0%}"
                    )
                    # 如果调整后信心不够，取消推送
                    if adjusted_prob < 0.55:
                        logger.info(f"🚫 伤病否决: {game['away_team']}@{game['home_team']} 核心缺阵")
                        try:
                            self.db.insert("""
                                INSERT INTO committee_rejections
                                    (game_id, away_team, home_team, buy_side, buy_team,
                                     score, model_prob, value_edge, poly_price,
                                     verdict, confidence, reasoning, rejection_source)
                                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                            """, (
                                gid, game['away_team'], game['home_team'],
                                buy_side, buy_team,
                                result['score'], adjusted_prob,
                                result.get('effective_edge', 0), result.get('poly_price', 0),
                                'reject', adjusted_prob,
                                f"核心缺阵: {inj_check['details']}",
                                'injury',
                            ))
                        except Exception:
                            pass
                        del self._candidates[gid]
                        continue
                    # 更新result中的model_prob
                    result["model_prob"] = adjusted_prob
                    result["injury_warning"] = inj_check["details"]
            except Exception as e:
                logger.warning(f"[Injury] {buy_team} 伤病检查失败: {e}，按无伤病处理")

            # 获取双方伤病详情（传给委员会做分析）
            try:
                home_injuries = self.injury_checker.check_team(game["home_team"])
                away_injuries = self.injury_checker.check_team(game["away_team"])
            except Exception:
                home_injuries = {}
                away_injuries = {}

            # 多Agent专家委员会审议
            if MINIMAX_API_KEY:
                game_ctx = {
                    "home_team": game["home_team"],
                    "away_team": game["away_team"],
                    "buy_side": buy_side,
                    "buy_team": buy_team,
                    "model_prob": result.get("model_prob", 0),
                    "blended_prob": result.get("blended_prob", 0),
                    "pinnacle_prob": result.get("fair_prob", 0),
                    "buy_price": result.get("poly_price", 0),
                    "value_edge": result.get("effective_edge", 0),
                    "score": result["score"],
                    "kelly": result.get("kelly", 0),
                    "breakdown": result["breakdown"],
                    "home_injuries": home_injuries,
                    "away_injuries": away_injuries,
                    "features": self.features.get_game_features(
                        game["home_team"], game["away_team"],
                        game.get("game_date", "")
                    ) or {},
                }
                memories = self.memory.get_relevant_memories(
                    [game["home_team"], game["away_team"]]
                )
                verdict = self.committee.deliberate(game_ctx, memories)

                if verdict["verdict"] != "buy" or verdict["confidence"] < 0.6:
                    logger.info(
                        f"🚫 委员会否决: {game['away_team']}@{game['home_team']} "
                        f"评分:{result['score']} → {verdict['verdict']} "
                        f"信心:{verdict['confidence']:.0%}"
                    )
                    # 记录否决到数据库（用于假阴性分析）
                    try:
                        self.db.insert("""
                            INSERT INTO committee_rejections
                                (game_id, away_team, home_team, buy_side, buy_team,
                                 score, model_prob, value_edge, poly_price,
                                 verdict, confidence, reasoning, rejection_source)
                            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                        """, (
                            gid, game['away_team'], game['home_team'],
                            buy_side, buy_team,
                            result['score'], result.get('model_prob', 0),
                            result.get('effective_edge', 0), result.get('poly_price', 0),
                            verdict['verdict'], verdict['confidence'],
                            verdict.get('reasoning', ''),
                            'committee',
                        ))
                    except Exception as e:
                        logger.debug(f"[DB] 否决记录失败: {e}")
                    # 从候选池移除（已审议）
                    del self._candidates[gid]
                    continue

                msg = format_game_message(game, result, slug=game.get("slug", ""))
                analysis = verdict.get("full_analysis", "")
                if analysis:
                    msg += f"\n\n🧠 <b>专家委员会:</b>\n{analysis}"
                if verdict.get("reasoning"):
                    msg += f"\n\n✅ <b>决策:</b> {verdict['reasoning']}"
                if verdict.get("entry_price"):
                    msg += f"\n📌 入场:{verdict['entry_price']:.0%}"
                if verdict.get("stop_loss"):
                    msg += f" 止损:{verdict['stop_loss']:.0%}"
                if verdict.get("take_profit"):
                    msg += f" 止盈:{verdict['take_profit']:.0%}"
            else:
                msg = format_game_message(game, result, slug=game.get("slug", ""))

            msg += f"\n📊 观察{candidate['scans']}次扫描，最优评分"
            # 伤病提醒
            if result.get("injury_warning"):
                msg += f"\n⚠️ <b>伤病:</b> {result['injury_warning']}"

            # 计算下单参数（附在消息里，按钮通过Telegram Bot发）
            if self.trader:
                bet_size = self.trader.calculate_bet_size(result.get("kelly", 0.02))
                if bet_size > 0:
                    # 获取token_id
                    clob_tokens = game.get("clob_token_ids", [])
                    token_id = ""
                    if clob_tokens and len(clob_tokens) >= 2:
                        token_id = clob_tokens[0] if buy_side == "away" else clob_tokens[1]
                    if token_id:
                        msg += (
                            f"\n\n💰 <b>一键下单:</b> ${bet_size:.0f} "
                            f"(Kelly {result.get('kelly', 0):.1%})"
                        )
                        # 通过Telegram Bot发带按钮的消息
                        self._send_trade_buttons(
                            msg, token_id, result.get("poly_price", 0),
                            bet_size, gid, buy_team
                        )
                        # 已通过按钮消息发送，跳过普通发送
                        self.db.insert(
                            "INSERT OR IGNORE INTO alerted_games (game_id, score, edge, source) VALUES (?,?,?,?)",
                            (gid, result["score"], result["edge"], result.get("source", ""))
                        )
                        self.daily_push_count += 1
                        if self.notion:
                            try:
                                self.notion.add_push(
                                    game_id=gid,
                                    matchup=f"{game['away_team']}@{game['home_team']}",
                                    date=game.get("game_date", ""),
                                    buy_team=buy_team,
                                    buy_price=result.get("poly_price", 0),
                                    model_prob=result.get("model_prob", 0),
                                    score=result["score"],
                                    kelly=result.get("kelly", 0),
                                    value_edge=result.get("effective_edge", 0),
                                )
                            except Exception:
                                pass
                        del self._candidates[gid]
                        logger.info(f"✅ 推送+下单按钮: {game['away_team']}@{game['home_team']} 评分:{result['score']}")
                        continue

            send_telegram(msg)
            self.db.insert(
                "INSERT OR IGNORE INTO alerted_games (game_id, score, edge, source) VALUES (?,?,?,?)",
                (gid, result["score"], result["edge"], result.get("source", ""))
            )
            self.daily_push_count += 1

            # Notion同步
            if self.notion:
                try:
                    self.notion.add_push(
                        game_id=gid,
                        matchup=f"{game['away_team']}@{game['home_team']}",
                        date=game.get("game_date", ""),
                        buy_team=buy_team,
                        buy_price=result.get("poly_price", 0),
                        model_prob=result.get("model_prob", 0),
                        score=result["score"],
                        kelly=result.get("kelly", 0),
                        value_edge=result.get("effective_edge", 0),
                    )
                except Exception as e:
                    logger.warning(f"[Notion] 同步失败: {e}")

            del self._candidates[gid]
            logger.info(
                f"✅ 推送: {game['away_team']}@{game['home_team']} "
                f"评分:{result['score']} (观察{candidate['scans']}次)"
            )

    def scan_futures(self):
        logger.info("[Scanner] 开始扫描期货盘...")
        markets = fetch_nba_futures(self.db)
        logger.info(f"[Scanner] 获取到 {len(markets)} 个期货盘")

        # 先评分所有期货，然后只推最高确定性的（按评分排序，最多推2个）
        scored = []
        for market in markets:
            cid = market["condition_id"]
            exists = self.db.execute_one(
                "SELECT 1 FROM alerted_futures WHERE condition_id=? AND pushed_at > datetime('now', '-24 hours')",
                (cid,)
            )
            if exists:
                continue

            result = self.engine.score_futures(market)
            if result is None or result.get("skip"):
                continue

            if result["push"]:
                scored.append((market, result))

        # 按评分排序，只推前2个最高确定性的
        scored.sort(key=lambda x: x[1]["score"], reverse=True)
        max_futures_push = 2

        for market, result in scored[:max_futures_push]:
            if self.daily_push_count >= self.DAILY_LIMIT:
                break

            cid = market["condition_id"]
            team = market.get("team", "")
            mtype = market.get("type", "")
            logger.info(f"[Scanner] 期货候选 {team} {mtype} 评分:{result['score']}")

            # 专家委员会审议（与比赛盘同等严格）
            if MINIMAX_API_KEY:
                TYPE_CN = {"champion": "总冠军", "mvp": "MVP", "division": "赛区冠军",
                           "playoff": "季后赛", "roy": "最佳新秀"}
                futures_ctx = {
                    "home_team": team,
                    "away_team": f"{TYPE_CN.get(mtype, mtype)}期货",
                    "buy_side": "futures",
                    "buy_team": team,
                    "model_prob": result.get("model_prob", 0),
                    "blended_prob": result.get("model_prob", 0),
                    "pinnacle_prob": 0,
                    "buy_price": market.get("price", 0),
                    "value_edge": result.get("value_edge", 0),
                    "score": result["score"],
                    "kelly": result.get("kelly", 0),
                    "breakdown": result.get("breakdown", {}),
                    "home_injuries": {},
                    "away_injuries": {},
                    "features": {
                        "market_type": "futures",
                        "futures_type": mtype,
                        "days_to_resolution": market.get("days_to_resolution", 0),
                        "question": market.get("question", ""),
                    },
                }
                memories = self.memory.get_relevant_memories([team]) if team else []
                verdict = self.committee.deliberate(futures_ctx, memories)

                if verdict["verdict"] != "buy" or verdict["confidence"] < 0.6:
                    logger.info(
                        f"🚫 委员会否决期货: {team} {mtype} "
                        f"→ {verdict['verdict']} 信心:{verdict['confidence']:.0%}"
                    )
                    continue

                msg = format_futures_message(market, result)
                analysis = verdict.get("full_analysis", "")
                if analysis:
                    msg += f"\n\n🧠 <b>专家委员会:</b>\n{analysis}"
                if verdict.get("reasoning"):
                    msg += f"\n\n✅ <b>决策:</b> {verdict['reasoning']}"
            else:
                msg = format_futures_message(market, result)

            send_telegram(msg)
            self.db.insert(
                "INSERT OR IGNORE INTO alerted_futures (condition_id, score) VALUES (?,?)",
                (cid, result["score"])
            )
            self.daily_push_count += 1
            logger.info(f"✅ 期货推送: {team} {mtype} 评分:{result['score']}")

    def _log_signal(self, game: dict, result: dict):
        """记录每个信号到signal_log，用于回测分析。

        每场比赛+buy_side只保留一条记录（更新为最新评分）。
        避免同一场比赛被记录20+次导致统计失真。
        """
        try:
            breakdown = dict(result["breakdown"])
            breakdown["away_edge"] = result.get("away_edge", 0)
            breakdown["home_edge"] = result.get("home_edge", 0)

            buy_side = result.get("buy_side", "away")
            buy_team = game["away_team"] if buy_side == "away" else game["home_team"]

            # 检查是否已有该比赛+买入方向的记录
            existing = self.db.execute_one(
                "SELECT id, score FROM signal_log WHERE game_id=? AND buy_side=?",
                (game["game_id"], buy_side)
            )

            if existing:
                # 已有记录：只在评分更高时更新
                if result["score"] > (existing["score"] or 0):
                    self.db.insert("""
                        UPDATE signal_log SET
                            poly_price=?, pinnacle_prob=?, raw_edge=?, effective_edge=?,
                            score=?, kelly=?, was_pushed=MAX(was_pushed, ?),
                            breakdown_json=?, source=?, scanned_at=CURRENT_TIMESTAMP
                        WHERE id=?
                    """, (
                        result.get("poly_price", result.get("poly_price_away", 0)),
                        result.get("fair_prob", result.get("fair_prob_away", 0)),
                        round(result["edge"], 4),
                        round(result.get("effective_edge", result["edge"]), 4),
                        result["score"],
                        round(result["kelly"], 4),
                        1 if result["push"] else 0,
                        json.dumps(breakdown, ensure_ascii=False),
                        result.get("source", ""),
                        existing["id"],
                    ))
            else:
                # 新记录
                self.db.insert("""
                    INSERT INTO signal_log
                        (game_id, away_team, home_team, market_type,
                         buy_side, buy_team, source,
                         poly_price, pinnacle_prob, raw_edge, effective_edge,
                         score, kelly, was_pushed, breakdown_json)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    game["game_id"],
                    game["away_team"],
                    game["home_team"],
                    "game",
                    buy_side,
                    buy_team,
                    result.get("source", ""),
                    result.get("poly_price", result.get("poly_price_away", 0)),
                    result.get("fair_prob", result.get("fair_prob_away", 0)),
                    round(result["edge"], 4),
                    round(result.get("effective_edge", result["edge"]), 4),
                    result["score"],
                    round(result["kelly"], 4),
                    1 if result["push"] else 0,
                    json.dumps(breakdown, ensure_ascii=False),
                ))
        except Exception as e:
            logger.warning(f"[SignalLog] 记录失败: {e} | game_id={game['game_id']}")

    def resolve_results(self):
        """自动回填比赛结果（通过The Odds API /scores，免费端点）"""
        resolved_list = []  # 初始化防NameError
        try:
            # 1. 回填push_results（ML训练用）
            result = self.resolver.resolve_all()
            if result["resolved"] > 0:
                logger.info(f"[Resolver] 回填{result['resolved']}条推送结果")

            # 2. 回填signal_log（回测用）
            resolved_signals, resolved_list = self._resolve_signal_log()
            if resolved_signals > 0:
                logger.info(f"[Resolver] 回填{resolved_signals}条回测信号")

            # 3. 赛后复盘 — 分析输赢原因，存入长期记忆
            if resolved_list:
                try:
                    mem_count = self.memory.run_post_mortem(resolved_list)
                    if mem_count > 0:
                        logger.info(f"[Memory] 新增{mem_count}条赛后复盘记忆")
                except Exception as e:
                    logger.warning(f"[Memory] 复盘失败: {e}")

            # 4. 自学习循环 — 每次有新结果时运行参数优化（autoresearch风格）
            if result["resolved"] > 0 or resolved_signals > 0:
                self._run_learning_cycle()
        except Exception as e:
            logger.warning(f"[Resolver] 回填失败: {e}")

    def _run_learning_cycle(self):
        """运行一轮autoresearch风格的自学习循环。"""
        try:
            learn_result = self.learner.run_learning_cycle(n_experiments=5)
            if learn_result.get("improved"):
                logger.info(
                    f"[AutoLearn] 参数优化成功! "
                    f"metric: {learn_result['baseline_metric']} → {learn_result['best_metric']}"
                )
                # 通知mismatch_engine重载参数
                self.engine.reload_params()
            elif learn_result.get("experiments_run", 0) > 0:
                logger.info(
                    f"[AutoLearn] 跑了{learn_result['experiments_run']}个实验，未发现改进 "
                    f"(当前metric: {learn_result.get('baseline_metric', 'N/A')})"
                )
        except Exception as e:
            logger.warning(f"[AutoLearn] 学习循环失败: {e}")

    def _resolve_signal_log(self) -> tuple:
        """回填signal_log中未结算的信号：判断预测对错 + 计算实际ROI。

        Returns:
            (resolved_count, resolved_signals_list)
        """
        from core.sportsbook_client import get_nba_scores, _match

        pending = self.db.execute("""
            SELECT id, game_id, away_team, home_team, buy_side, buy_team,
                   poly_price, kelly, effective_edge, breakdown_json, was_pushed
            FROM signal_log
            WHERE prediction_correct IS NULL
              AND scanned_at < datetime('now', '-3 hours')
        """)
        if not pending:
            return 0, []

        scores = get_nba_scores(ODDS_KEY, days_from=3)
        completed = [s for s in scores if s.get("completed")]
        resolved = 0
        resolved_list = []

        for row in pending:
            r = dict(row)
            for s in completed:
                if not (_match(s["home_team"], r["home_team"]) and
                        _match(s["away_team"], r["away_team"])):
                    continue
                hs = s.get("home_score")
                aws = s.get("away_score")
                if hs is None or aws is None:
                    continue

                away_won = aws > hs
                actual_outcome = 1 if away_won else 0

                buy_side = r.get("buy_side") or "away"
                if buy_side == "away":
                    prediction_correct = 1 if away_won else 0
                else:
                    prediction_correct = 1 if not away_won else 0

                poly_price = r["poly_price"] or 0.5
                if prediction_correct == 1 and poly_price > 0:
                    actual_roi = round((1.0 / poly_price) - 1.0 - POLY_FEE_RATE, 4)
                else:
                    actual_roi = -1.0

                kelly = r["kelly"] or 0
                hypo_pnl = round(actual_roi * kelly, 4) if kelly > 0 else 0

                self.db.insert("""
                    UPDATE signal_log
                    SET actual_outcome=?, actual_score_home=?, actual_score_away=?,
                        prediction_correct=?, actual_roi=?, hypo_pnl=?,
                        resolved_at=CURRENT_TIMESTAMP
                    WHERE id=?
                """, (actual_outcome, hs, aws, prediction_correct, actual_roi,
                      hypo_pnl, r["id"]))
                resolved += 1

                # 记录已结算信号（用于赛后复盘）
                r["prediction_correct"] = prediction_correct
                r["actual_score_home"] = hs
                r["actual_score_away"] = aws
                r["actual_roi"] = actual_roi
                # 从scanned_at提取日期（比从game_id切片更可靠）
                scanned = r.get("scanned_at") or ""
                r["game_date"] = scanned[:10] if len(scanned) >= 10 else ""
                resolved_list.append(r)

                # Notion同步结果
                if r.get("was_pushed") and self.notion:
                    try:
                        matchup = f"{r['away_team']}@{r['home_team']}"
                        self.notion.update_result(
                            game_id=r["game_id"],
                            matchup=matchup,
                            score_str=f"{aws}-{hs}",
                            prediction_correct=prediction_correct,
                            actual_roi=actual_roi,
                        )
                    except Exception:
                        pass
                break

        # 回填否决记录的比赛结果（假阴性分析）
        self._resolve_rejections(completed)

        return resolved, resolved_list

    def _resolve_rejections(self, completed_scores: list):
        """回填委员会否决记录的比赛结果，用于分析假阴性。"""
        from core.sportsbook_client import _match
        try:
            pending = self.db.execute("""
                SELECT id, away_team, home_team, buy_side
                FROM committee_rejections
                WHERE actual_outcome IS NULL
                  AND rejected_at < datetime('now', '-3 hours')
            """)
            if not pending:
                return
            resolved = 0
            for row in pending:
                r = dict(row)
                for s in completed_scores:
                    if not (_match(s["home_team"], r["home_team"]) and
                            _match(s["away_team"], r["away_team"])):
                        continue
                    hs, aws = s.get("home_score"), s.get("away_score")
                    if hs is None or aws is None:
                        continue
                    away_won = aws > hs
                    buy_side = r.get("buy_side") or "away"
                    if buy_side == "away":
                        correct = 1 if away_won else 0
                    else:
                        correct = 1 if not away_won else 0
                    self.db.insert(
                        "UPDATE committee_rejections SET actual_outcome=? WHERE id=?",
                        (correct, r["id"])
                    )
                    resolved += 1
                    break
            if resolved:
                # 统计假阴性率
                stats = self.db.execute_one("""
                    SELECT COUNT(*) as total,
                           SUM(CASE WHEN actual_outcome=1 THEN 1 ELSE 0 END) as false_negatives
                    FROM committee_rejections WHERE actual_outcome IS NOT NULL
                """)
                if stats and stats["total"] > 0:
                    fn_rate = (stats["false_negatives"] or 0) / stats["total"]
                    logger.info(
                        f"[Rejection] 回填{resolved}条否决结果 | "
                        f"假阴性率: {fn_rate:.0%} ({stats['false_negatives']}/{stats['total']}场被否决但实际会赢)"
                    )
        except Exception as e:
            logger.debug(f"[Rejection] 回填失败: {e}")

    def refresh_injury_data(self):
        """每日伤病数据刷新 — 重新拉取ESPN，自动清除康复球员。"""
        try:
            result = self.espn.cleanup_recovered_players()
            if result["recovered"] > 0:
                logger.info(
                    f"[Injury] 每日清理完成: 清除{result['recovered']}名康复球员, "
                    f"当前{result['after']}名伤病"
                )
            if result["new_injuries"] > 0:
                logger.info(
                    f"[Injury] 新增{result['new_injuries']}名伤病球员, "
                    f"当前{result['after']}名伤病"
                )
        except Exception as e:
            logger.warning(f"[Injury] 伤病刷新失败: {e}")

    def send_daily_summary(self):
        """每日复盘报表 — 推送到Telegram + 分析点位质量 + 自我进化"""
        try:
            # ── 1. 昨日推送信号复盘 ──
            pushed_resolved = self.db.execute("""
                SELECT buy_team, score, poly_price, raw_edge, effective_edge,
                       prediction_correct, actual_roi, actual_score_away, actual_score_home,
                       away_team, home_team, buy_side, pinnacle_prob
                FROM signal_log
                WHERE was_pushed=1 AND prediction_correct IS NOT NULL
                  AND resolved_at > datetime('now', '-1 day')
                ORDER BY resolved_at DESC
            """)

            # ── 2. 7天累计 ──
            push_stats = self.db.execute_one("""
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN prediction_correct=1 THEN 1 ELSE 0 END) as wins,
                       AVG(actual_roi) as avg_roi
                FROM signal_log
                WHERE was_pushed=1 AND prediction_correct IS NOT NULL
                  AND resolved_at > datetime('now', '-7 days')
            """) or {}

            # ── 3. 点位质量分析（核心：推送价格 vs 当天最优价格）──
            price_analysis = self.db.execute("""
                SELECT game_id,
                       MIN(poly_price) as best_price,
                       MAX(poly_price) as worst_price,
                       AVG(poly_price) as avg_price
                FROM signal_log
                WHERE was_pushed=1
                  AND resolved_at > datetime('now', '-1 day')
                GROUP BY game_id
            """)

            pending = self.db.execute_one("""
                SELECT COUNT(*) as c FROM signal_log WHERE prediction_correct IS NULL
            """) or {"c": 0}

            # ── 构建报表 ──
            push_total = push_stats.get("total", 0) or 0
            push_wins = push_stats.get("wins", 0) or 0
            push_roi = push_stats.get("avg_roi", 0) or 0

            msg = f"📊 <b>每日复盘报表</b>\n━━━━━━━━━━━━━━━\n"

            # 昨日结果
            yesterday_results = [dict(r) for r in pushed_resolved]
            if yesterday_results:
                y_wins = sum(1 for r in yesterday_results if r["prediction_correct"])
                y_total = len(yesterday_results)
                msg += f"📅 <b>昨日推送: {y_wins}胜{y_total - y_wins}负 ({y_wins}/{y_total})</b>\n"
                for r in yesterday_results:
                    icon = "✅" if r["prediction_correct"] else "❌"
                    roi = r["actual_roi"] or 0
                    score_str = f"{r['actual_score_away']}-{r['actual_score_home']}"
                    msg += f"  {icon} {r['buy_team']} 入场{r['poly_price']:.0%} {score_str} ROI{roi:+.0%}\n"
            else:
                msg += f"📅 昨日无已结算推送\n"

            msg += f"━━━━━━━━━━━━━━━\n"

            # 7天累计
            if push_total > 0:
                msg += f"📈 <b>7天累计: {push_wins}/{push_total} ({push_wins/push_total:.0%})</b> ROI{push_roi:+.1%}\n"

            # 点位质量分析
            if price_analysis:
                price_data = [dict(r) for r in price_analysis]
                if price_data:
                    msg += f"━━━━━━━━━━━━━━━\n"
                    msg += f"📌 <b>点位质量</b>\n"
                    for p in price_data:
                        best = p["best_price"] or 0
                        worst = p["worst_price"] or 0
                        if best > 0 and worst > 0 and best != worst:
                            savings = worst - best
                            msg += f"  {p['game_id'][:12]}: 最优{best:.0%} 最差{worst:.0%} 差{savings:+.0%}\n"

            msg += f"━━━━━━━━━━━━━━━\n"
            msg += f"⏳ 待结算: {pending['c']}场\n"

            # 自学习+模型状态
            try:
                lp = self.learner.params.get("_meta", {})
                msg += f"🧠 自学习v{lp.get('version', 1)}"
                pred_report = self.predictor.get_report()
                msg += f" | 模型样本:{pred_report.get('historical_games', 0)}\n"
            except Exception:
                pass

            msg += f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}"

            send_telegram(msg)
            logger.info("[Scanner] 每日复盘已推送")

            # ── 4. 自我进化：分析点位质量，调整参数 ──
            self._daily_evolution(yesterday_results)

        except Exception as e:
            logger.warning(f"[Scanner] 每日复盘失败: {e}")

    def _daily_evolution(self, yesterday_results: list):
        """每日自我进化 — 分析昨天哪些推送点位好/差，调整策略。"""
        if not yesterday_results:
            return

        try:
            wins = [r for r in yesterday_results if r.get("prediction_correct")]
            losses = [r for r in yesterday_results if not r.get("prediction_correct")]

            # 分析赢的共同特征
            if wins:
                avg_win_price = sum(r["poly_price"] for r in wins) / len(wins)
                avg_win_edge = sum(r.get("effective_edge", 0) or 0 for r in wins) / len(wins)
                avg_win_score = sum(r["score"] for r in wins) / len(wins)
            else:
                avg_win_price = avg_win_edge = avg_win_score = 0

            # 分析输的共同特征
            if losses:
                avg_loss_price = sum(r["poly_price"] for r in losses) / len(losses)
                avg_loss_edge = sum(r.get("effective_edge", 0) or 0 for r in losses) / len(losses)
                avg_loss_score = sum(r["score"] for r in losses) / len(losses)
            else:
                avg_loss_price = avg_loss_edge = avg_loss_score = 0

            logger.info(
                f"[Evolution] 昨日复盘: "
                f"赢{len(wins)}场(均价{avg_win_price:.0%} edge{avg_win_edge:+.1%} 分{avg_win_score:.0f}) "
                f"输{len(losses)}场(均价{avg_loss_price:.0%} edge{avg_loss_edge:+.1%} 分{avg_loss_score:.0f})"
            )

            # 如果输的平均价格明显高于赢的 → 说明高价买入容易亏
            if losses and wins and avg_loss_price > avg_win_price + 0.05:
                # 自动降低max_buy_price
                import json
                params = json.loads(open("scoring_params.json").read())
                current_max = params.get("prediction", {}).get("max_buy_price", 0.65)
                new_max = max(0.45, current_max - 0.02)  # 每天最多降2%
                if new_max < current_max:
                    params["prediction"]["max_buy_price"] = round(new_max, 2)
                    params["_meta"]["updated_by"] = "daily_evolution"
                    params["_meta"]["updated_at"] = datetime.now().isoformat()
                    with open("scoring_params.json", "w") as f:
                        json.dump(params, f, indent=4, ensure_ascii=False)
                    self.engine.reload_params()
                    logger.info(f"[Evolution] 降低max_buy_price: {current_max:.0%} → {new_max:.0%} (输的均价偏高)")

            # 如果整体胜率>70% → 可以稍微放宽门槛
            total = len(yesterday_results)
            wr = len(wins) / total if total > 0 else 0
            if wr >= 0.75 and total >= 3:
                import json
                params = json.loads(open("scoring_params.json").read())
                current_conf = params.get("prediction", {}).get("min_model_confidence", 0.60)
                new_conf = max(0.55, current_conf - 0.01)  # 微调
                if new_conf < current_conf:
                    params["prediction"]["min_model_confidence"] = round(new_conf, 2)
                    params["_meta"]["updated_by"] = "daily_evolution"
                    params["_meta"]["updated_at"] = datetime.now().isoformat()
                    with open("scoring_params.json", "w") as f:
                        json.dump(params, f, indent=4, ensure_ascii=False)
                    self.engine.reload_params()
                    logger.info(f"[Evolution] 降低信心门槛: {current_conf:.0%} → {new_conf:.0%} (胜率优秀)")

            # 如果整体胜率<40% → 收紧门槛
            if wr < 0.40 and total >= 3:
                import json
                params = json.loads(open("scoring_params.json").read())
                current_conf = params.get("prediction", {}).get("min_model_confidence", 0.60)
                new_conf = min(0.75, current_conf + 0.02)
                if new_conf > current_conf:
                    params["prediction"]["min_model_confidence"] = round(new_conf, 2)
                    params["_meta"]["updated_by"] = "daily_evolution"
                    params["_meta"]["updated_at"] = datetime.now().isoformat()
                    with open("scoring_params.json", "w") as f:
                        json.dump(params, f, indent=4, ensure_ascii=False)
                    self.engine.reload_params()
                    logger.info(f"[Evolution] 收紧信心门槛: {current_conf:.0%} → {new_conf:.0%} (胜率太低)")

            # 运行自学习实验
            self._run_learning_cycle()

        except Exception as e:
            logger.warning(f"[Evolution] 自我进化失败: {e}")

    def _send_trade_buttons(self, msg: str, token_id: str, price: float,
                             amount: float, game_id: str, buy_team: str):
        """发送带交易确认按钮的Telegram消息。"""
        try:
            # 截断callback_data到64字节限制
            tid_short = token_id[:20]
            gid_short = game_id[:15]
            confirm_data = f"trade_confirm:{tid_short}:{price:.2f}:{amount:.0f}:{gid_short}:{buy_team}"
            skip_data = f"trade_skip:{gid_short}"

            keyboard = {
                "inline_keyboard": [
                    [
                        {"text": f"✅ 买入 ${amount:.0f}", "callback_data": confirm_data[:64]},
                        {"text": "❌ 跳过", "callback_data": skip_data[:64]},
                    ]
                ]
            }

            requests.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                json={
                    "chat_id": CHAT_ID,
                    "text": msg,
                    "parse_mode": "HTML",
                    "reply_markup": keyboard,
                },
                timeout=10,
            )
        except Exception as e:
            logger.warning(f"[TG] 发送交易按钮失败: {e}")
            send_telegram(msg)  # 降级为普通消息

    def send_nightly_review(self):
        """每晚深度复盘 — 比赛结束后AI全面分析当天所有比赛。

        与 send_daily_summary 的区别:
          - daily_summary = 早上推数字报表（胜负、ROI）
          - nightly_review = 晚上AI深度分析（每场比赛为什么对/错，模式总结，策略调整）

        流程:
          1. 先回填今日比赛结果
          2. 拉取今日所有已结算的信号（推送+未推送）
          3. AI分析每场比赛：预测vs实际，为什么对/错
          4. 总结模式：今天什么类型的比赛赢了/输了
          5. 提出改进建议
          6. 推送复盘报告到Telegram
          7. 刷新伤病数据（清除康复球员）
        """
        logger.info("[Nightly] 开始每晚深度复盘...")

        try:
            # ── 1. 先回填结果 ──
            self.resolve_results()

            # ── 2. 拉取今日已结算信号 ──
            today_signals = self.db.execute("""
                SELECT game_id, away_team, home_team, buy_side, buy_team,
                       score, poly_price, pinnacle_prob, raw_edge, effective_edge,
                       prediction_correct, actual_roi, actual_score_away, actual_score_home,
                       was_pushed, kelly, breakdown_json, source
                FROM signal_log
                WHERE prediction_correct IS NOT NULL
                  AND resolved_at > datetime('now', '-18 hours')
                ORDER BY score DESC
            """)

            if not today_signals:
                logger.info("[Nightly] 今日无已结算比赛")
                # 仍然刷新伤病
                self.refresh_injury_data()
                return

            signals = [dict(r) for r in today_signals]
            wins = [s for s in signals if s["prediction_correct"]]
            losses = [s for s in signals if not s["prediction_correct"]]
            pushed = [s for s in signals if s["was_pushed"]]
            pushed_wins = [s for s in pushed if s["prediction_correct"]]

            # ── 3. 构建每场比赛摘要 ──
            game_summaries = []
            for s in signals:
                icon = "✅" if s["prediction_correct"] else "❌"
                push_mark = "📢推送" if s["was_pushed"] else "📋未推送"
                score_str = f"{s['actual_score_away']}-{s['actual_score_home']}"
                roi = s["actual_roi"] or 0

                summary = (
                    f"{icon} {s['away_team']}@{s['home_team']} {score_str}\n"
                    f"  买{s['buy_side']}({s['buy_team']}) 入场{s['poly_price']:.0%} "
                    f"模型评分:{s['score']} edge:{s['effective_edge']:+.2%} "
                    f"ROI:{roi:+.0%} [{push_mark}]"
                )
                game_summaries.append(summary)

            # ── 4. AI深度分析 ──
            ai_review = ""
            if MINIMAX_API_KEY:
                all_games_text = "\n".join(game_summaries)

                # 统计数据
                total = len(signals)
                win_rate = len(wins) / total if total > 0 else 0
                pushed_total = len(pushed)
                pushed_wr = len(pushed_wins) / pushed_total if pushed_total > 0 else 0
                avg_win_score = sum(s["score"] for s in wins) / len(wins) if wins else 0
                avg_loss_score = sum(s["score"] for s in losses) / len(losses) if losses else 0
                avg_win_price = sum(s["poly_price"] for s in wins) / len(wins) if wins else 0
                avg_loss_price = sum(s["poly_price"] for s in losses) / len(losses) if losses else 0

                stats_text = (
                    f"全部信号: {len(wins)}胜{len(losses)}负 ({win_rate:.0%})\n"
                    f"推送信号: {len(pushed_wins)}胜{pushed_total - len(pushed_wins)}负 ({pushed_wr:.0%})\n"
                    f"赢的均分:{avg_win_score:.0f} 输的均分:{avg_loss_score:.0f}\n"
                    f"赢的均价:{avg_win_price:.0%} 输的均价:{avg_loss_price:.0%}"
                )

                review_prompt = f"""你是NBA预测市场的首席分析师。请对今天的比赛预测进行深度复盘。

今日比赛结果:
{all_games_text}

统计数据:
{stats_text}

请分析以下内容（总共不超过500字）：

1. 【今日总结】一句话概括今日表现
2. 【亮点】哪些预测最准确？为什么？（提到具体比赛）
3. 【失误】哪些预测失败了？分析失败原因（提到具体比赛、比分）
4. 【模式发现】今天的结果揭示了什么规律？（如：高评分信号是否更准？主队/客队偏差？大edge是否可信？）
5. 【明日建议】基于今天的复盘，明天选比赛应该注意什么？
6. 【策略调整】是否需要调整参数？（如门槛太高/太低、某类信号应该加权/减权）"""

                try:
                    resp = requests.post(
                        f"{MINIMAX_API_URL}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {MINIMAX_API_KEY}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": MINIMAX_MODEL,
                            "messages": [{"role": "user", "content": review_prompt}],
                            "temperature": 0.3,
                            "max_tokens": 2000,
                        },
                        timeout=90,
                    )
                    resp.raise_for_status()
                    content = resp.json()["choices"][0]["message"]["content"]
                    ai_review = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                    ai_review = ai_review[:1500]
                except Exception as e:
                    logger.warning(f"[Nightly] AI复盘失败: {e}")
                    ai_review = ""

            # ── 5. 构建Telegram消息 ──
            total = len(signals)
            msg = f"🌙 <b>每晚深度复盘</b>\n━━━━━━━━━━━━━━━\n"

            # 数字概览
            msg += f"📊 <b>今日: {len(wins)}胜{len(losses)}负"
            if total > 0:
                msg += f" ({len(wins)/total:.0%})"
            msg += "</b>\n"

            if pushed:
                msg += f"📢 推送信号: {len(pushed_wins)}胜{len(pushed) - len(pushed_wins)}负"
                if pushed:
                    msg += f" ({len(pushed_wins)/len(pushed):.0%})"
                msg += "\n"

            msg += "━━━━━━━━━━━━━━━\n"

            # 每场比赛结果
            for s in signals[:12]:  # 最多显示12场
                icon = "✅" if s["prediction_correct"] else "❌"
                push_mark = "📢" if s["was_pushed"] else ""
                score_str = f"{s['actual_score_away']}-{s['actual_score_home']}"
                roi = s["actual_roi"] or 0
                msg += (
                    f"{icon}{push_mark} {s['away_team']}@{s['home_team']} "
                    f"{score_str} 买{s['buy_team']} "
                    f"{s['poly_price']:.0%} 评分:{s['score']} "
                    f"ROI:{roi:+.0%}\n"
                )

            if len(signals) > 12:
                msg += f"  ... 还有{len(signals) - 12}场\n"

            # AI深度分析
            if ai_review:
                msg += f"\n━━━━━━━━━━━━━━━\n"
                msg += f"🧠 <b>AI深度分析:</b>\n{ai_review}\n"

            msg += f"\n━━━━━━━━━━━━━━━\n"
            msg += f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}"

            send_telegram(msg)
            logger.info(
                f"[Nightly] 深度复盘已推送: "
                f"{len(wins)}胜{len(losses)}负 推送{len(pushed_wins)}/{len(pushed)}"
            )

            # ── 6. 自我进化（基于全量信号，不只是推送）──
            self._daily_evolution(signals)

            # ── 7. 刷新伤病数据 ──
            self.refresh_injury_data()

            # ── 8. 赛后复盘记忆（存入game_memory） ──
            try:
                mem_count = self.memory.run_post_mortem(signals)
                if mem_count > 0:
                    logger.info(f"[Nightly] 新增{mem_count}条赛后记忆")
            except Exception as e:
                logger.warning(f"[Nightly] 赛后记忆失败: {e}")

        except Exception as e:
            logger.warning(f"[Nightly] 深度复盘失败: {e}")

    def send_evening_picks(self):
        """每晚精选推送 — 用更高门槛筛选当晚最佳机会，只推1-2个精选。

        与常规扫描的区别:
          - 更高评分门槛 (evening_min_score)
          - 更高信心门槛 (evening_min_confidence)
          - 更高edge门槛 (evening_min_value_edge)
          - 经过一天的价格观察，信号更稳定
          - 最多推2个精选（少而精）
        """
        logger.info("[Scanner] 开始每晚精选扫描...")
        lp = self.engine.learned_params
        pred_cfg = lp.get("prediction", {})
        evening_min_score = pred_cfg.get("evening_min_score", 55)
        evening_min_confidence = pred_cfg.get("evening_min_confidence", 0.68)
        evening_min_value_edge = pred_cfg.get("evening_min_value_edge", 0.025)

        games = fetch_nba_games()
        if not games:
            logger.info("[Evening] 无比赛数据")
            return

        evening_candidates = []
        for game in games:
            gid = game["game_id"]

            # 已推送过的跳过
            exists = self.db.execute_one(
                "SELECT 1 FROM alerted_games WHERE game_id=?", (gid,)
            )
            if exists:
                continue

            result = self.engine.score_game(game)
            if result is None:
                continue

            buy_side = result.get("buy_side", "away")
            buy_team = game["away_team"] if buy_side == "away" else game["home_team"]
            model_prob = result.get("model_prob", 0)
            effective_edge = result.get("effective_edge", 0)
            bd = result.get("breakdown", {})
            pinnacle_agrees = bd.get("pinnacle_agreement", {}).get("agrees", False)

            # 精选门槛：比常规更严格
            if (result["score"] >= evening_min_score
                    and model_prob >= evening_min_confidence
                    and effective_edge >= evening_min_value_edge
                    and pinnacle_agrees
                    and result.get("poly_price", 1) <= 0.65
                    and result.get("poly_price", 0) >= 0.30):
                evening_candidates.append({
                    "game": game,
                    "result": result,
                    "buy_team": buy_team,
                    "buy_side": buy_side,
                    "score": result["score"],
                })

        if not evening_candidates:
            logger.info("[Evening] 无满足精选门槛的比赛")
            return

        # 按评分排序，只推前2个
        evening_candidates.sort(key=lambda x: x["score"], reverse=True)
        max_evening = 2

        pushed = 0
        for candidate in evening_candidates[:max_evening]:
            if self.daily_push_count >= self.DAILY_LIMIT:
                break

            game = candidate["game"]
            result = candidate["result"]
            buy_team = candidate["buy_team"]
            buy_side = candidate["buy_side"]
            gid = game["game_id"]

            # 伤病检查
            try:
                inj_check = self.injury_checker.check_team(buy_team)
                if inj_check["has_star_out"]:
                    adjusted_prob = result.get("model_prob", 0) - inj_check["injury_penalty"]
                    if adjusted_prob < evening_min_confidence:
                        logger.info(f"🚫 精选被伤病否决: {buy_team} {inj_check['details']}")
                        continue
                    result["model_prob"] = adjusted_prob
                    result["injury_warning"] = inj_check["details"]
            except Exception as e:
                logger.warning(f"[Evening] {buy_team} 伤病检查失败: {e}")

            # 获取双方伤病详情（传给委员会做分析）
            try:
                home_injuries = self.injury_checker.check_team(game["home_team"])
                away_injuries = self.injury_checker.check_team(game["away_team"])
            except Exception:
                home_injuries = {}
                away_injuries = {}

            # 专家委员会审议
            if MINIMAX_API_KEY:
                game_ctx = {
                    "home_team": game["home_team"],
                    "away_team": game["away_team"],
                    "buy_side": buy_side,
                    "buy_team": buy_team,
                    "model_prob": result.get("model_prob", 0),
                    "blended_prob": result.get("blended_prob", 0),
                    "pinnacle_prob": result.get("fair_prob", 0),
                    "buy_price": result.get("poly_price", 0),
                    "value_edge": result.get("effective_edge", 0),
                    "score": result["score"],
                    "kelly": result.get("kelly", 0),
                    "breakdown": result["breakdown"],
                    "home_injuries": home_injuries,
                    "away_injuries": away_injuries,
                    "features": self.features.get_game_features(
                        game["home_team"], game["away_team"],
                        game.get("game_date", "")
                    ) or {},
                }
                memories = self.memory.get_relevant_memories(
                    [game["home_team"], game["away_team"]]
                )
                verdict = self.committee.deliberate(game_ctx, memories)

                if verdict["verdict"] != "buy" or verdict["confidence"] < 0.65:
                    logger.info(
                        f"🚫 精选委员会否决: {game['away_team']}@{game['home_team']} "
                        f"→ {verdict['verdict']} 信心:{verdict['confidence']:.0%}"
                    )
                    continue

                msg = f"🌟 <b>今晚精选推荐</b>\n━━━━━━━━━━━━━━━\n"
                msg += format_game_message(game, result, slug=game.get("slug", ""))
                analysis = verdict.get("full_analysis", "")
                if analysis:
                    msg += f"\n\n🧠 <b>专家委员会:</b>\n{analysis}"
                if verdict.get("reasoning"):
                    msg += f"\n\n✅ <b>决策:</b> {verdict['reasoning']}"
                if verdict.get("entry_price"):
                    msg += f"\n📌 入场:{verdict['entry_price']:.0%}"
                if verdict.get("stop_loss"):
                    msg += f" 止损:{verdict['stop_loss']:.0%}"
                if verdict.get("take_profit"):
                    msg += f" 止盈:{verdict['take_profit']:.0%}"
            else:
                msg = f"🌟 <b>今晚精选推荐</b>\n━━━━━━━━━━━━━━━\n"
                msg += format_game_message(game, result, slug=game.get("slug", ""))

            if result.get("injury_warning"):
                msg += f"\n⚠️ <b>伤病:</b> {result['injury_warning']}"

            # 带交易按钮
            clob_tokens = game.get("clob_token_ids", [])
            token_id = ""
            if clob_tokens and len(clob_tokens) >= 2:
                token_id = clob_tokens[0] if buy_side == "away" else clob_tokens[1]
            if self.trader and token_id:
                bet_size = self.trader.calculate_bet_size(result.get("kelly", 0.02))
                if bet_size > 0:
                    self._send_trade_buttons(
                        msg, token_id, result.get("poly_price", 0),
                        bet_size, gid, buy_team
                    )
                    self.db.insert(
                        "INSERT OR IGNORE INTO alerted_games (game_id, score, edge, source) VALUES (?,?,?,?)",
                        (gid, result["score"], result["edge"], result.get("source", ""))
                    )
                    self.daily_push_count += 1
                    pushed += 1
                    logger.info(f"🌟 精选推送+下单: {game['away_team']}@{game['home_team']} 评分:{result['score']}")
                    continue

            send_telegram(msg)
            self.db.insert(
                "INSERT OR IGNORE INTO alerted_games (game_id, score, edge, source) VALUES (?,?,?,?)",
                (gid, result["score"], result["edge"], result.get("source", ""))
            )
            self.daily_push_count += 1
            pushed += 1
            logger.info(f"🌟 精选推送: {game['away_team']}@{game['home_team']} 评分:{result['score']}")

        if pushed > 0:
            logger.info(f"[Evening] 精选推送完成: {pushed}场")
        else:
            logger.info("[Evening] 所有候选被否决，无精选推送")

    def reset_daily_count(self):
        self.daily_push_count = 0
        self._candidates = {}
        if self.trader:
            self.trader.reset_daily()
        logger.info("[Scanner] 每日推送计数+候选池+交易额度重置")


# ── 回测报告 ─────────────────────────────────────────────────────
def print_backtest_report():
    db = Database()

    total = db.execute_one("SELECT COUNT(*) as c FROM signal_log")["c"]
    resolved = db.execute_one("SELECT COUNT(*) as c FROM signal_log WHERE actual_outcome IS NOT NULL")["c"]
    pending = total - resolved
    pushed = db.execute_one("SELECT COUNT(*) as c FROM signal_log WHERE was_pushed=1")["c"]

    print("\n━━━ 回测数据报告 ━━━")
    print(f"  总信号数:      {total}")
    print(f"  已结算:        {resolved}")
    print(f"  待结算:        {pending}")
    print(f"  达到推送阈值:  {pushed}")

    if resolved == 0:
        print("\n  暂无结算数据，等比赛结束后自动回填。")
        # 显示今天的信号预览
        today_signals = db.execute("""
            SELECT away_team, home_team, score, effective_edge, poly_price, pinnacle_prob, kelly
            FROM signal_log ORDER BY scanned_at DESC LIMIT 20
        """)
        if today_signals:
            print(f"\n━━━ 最近{len(today_signals)}条信号 ━━━")
            print(f"  {'比赛':12s} {'评分':>4s} {'edge':>7s} {'Poly价':>6s} {'Pin概率':>7s} {'Kelly':>6s}")
            print(f"  {'─'*50}")
            for s in today_signals:
                r = dict(s)
                matchup = f"{r['away_team']}@{r['home_team']}"
                print(f"  {matchup:12s} {r['score']:4d} {r['effective_edge']:+6.2%} {r['poly_price']:6.1%} {r['pinnacle_prob']:6.1%} {r['kelly']:6.2%}")
        return

    # 全量统计
    wins = db.execute_one("SELECT COUNT(*) as c FROM signal_log WHERE actual_outcome=1")["c"]
    win_rate = wins / resolved
    total_hypo_pnl = db.execute_one("SELECT COALESCE(SUM(hypo_pnl),0) as s FROM signal_log WHERE actual_outcome IS NOT NULL")["s"]
    avg_hypo_pnl = db.execute_one("SELECT COALESCE(AVG(hypo_pnl),0) as s FROM signal_log WHERE actual_outcome IS NOT NULL")["s"]

    print(f"\n━━━ 全量信号表现 ━━━")
    print(f"  总胜率:        {win_rate:.1%} ({wins}/{resolved})")
    print(f"  假设总ROI:     {total_hypo_pnl:+.2%}")
    print(f"  假设平均ROI:   {avg_hypo_pnl:+.4f}/信号")

    # 按评分区间统计
    score_buckets = db.execute("""
        SELECT
            CASE
                WHEN score >= 70 THEN '70+ (推送级)'
                WHEN score >= 50 THEN '50-69'
                WHEN score >= 30 THEN '30-49'
                ELSE '0-29'
            END as bucket,
            COUNT(*) as cnt,
            SUM(actual_outcome) as wins,
            AVG(actual_outcome) as win_rate,
            SUM(hypo_pnl) as total_pnl,
            AVG(hypo_pnl) as avg_pnl
        FROM signal_log WHERE actual_outcome IS NOT NULL
        GROUP BY bucket ORDER BY bucket DESC
    """)
    print(f"\n━━━ 按评分区间 ━━━")
    print(f"  {'区间':14s} {'场次':>4s} {'胜率':>6s} {'总ROI':>8s} {'均ROI':>8s}")
    print(f"  {'─'*46}")
    for b in score_buckets:
        r = dict(b)
        wr = r['win_rate'] or 0
        tp = r['total_pnl'] or 0
        ap = r['avg_pnl'] or 0
        print(f"  {r['bucket']:14s} {r['cnt']:4d} {wr:6.1%} {tp:+7.2%} {ap:+7.4f}")

    # 按edge区间统计
    edge_buckets = db.execute("""
        SELECT
            CASE
                WHEN effective_edge >= 0.05 THEN 'edge 5%+'
                WHEN effective_edge >= 0.03 THEN 'edge 3-5%'
                WHEN effective_edge >= 0.01 THEN 'edge 1-3%'
                WHEN effective_edge >= 0 THEN 'edge 0-1%'
                ELSE 'edge <0%'
            END as bucket,
            COUNT(*) as cnt,
            AVG(actual_outcome) as win_rate,
            SUM(hypo_pnl) as total_pnl
        FROM signal_log WHERE actual_outcome IS NOT NULL
        GROUP BY bucket ORDER BY bucket DESC
    """)
    print(f"\n━━━ 按Edge区间 ━━━")
    print(f"  {'区间':14s} {'场次':>4s} {'胜率':>6s} {'总ROI':>8s}")
    print(f"  {'─'*36}")
    for b in edge_buckets:
        r = dict(b)
        wr = r['win_rate'] or 0
        tp = r['total_pnl'] or 0
        print(f"  {r['bucket']:14s} {r['cnt']:4d} {wr:6.1%} {tp:+7.2%}")

    # 最近的已结算信号
    recent = db.execute("""
        SELECT away_team, home_team, score, effective_edge, poly_price,
               actual_outcome, actual_score_away, actual_score_home, hypo_pnl
        FROM signal_log WHERE actual_outcome IS NOT NULL
        ORDER BY resolved_at DESC LIMIT 10
    """)
    if recent:
        print(f"\n━━━ 最近结算 ━━━")
        print(f"  {'比赛':12s} {'评分':>4s} {'edge':>7s} {'结果':4s} {'比分':>9s} {'ROI':>8s}")
        print(f"  {'─'*50}")
        for s in recent:
            r = dict(s)
            matchup = f"{r['away_team']}@{r['home_team']}"
            result_str = "✅" if r['actual_outcome'] == 1 else "❌"
            score_str = f"{r['actual_score_away']}-{r['actual_score_home']}"
            pnl = r['hypo_pnl'] or 0
            print(f"  {matchup:12s} {r['score']:4d} {r['effective_edge']:+6.2%} {result_str:4s} {score_str:>9s} {pnl:+7.2%}")


# ── 详细统计报告 ──────────────────────────────────────────────────
def print_stats_report():
    db = Database()

    # 全量统计
    total = db.execute_one("SELECT COUNT(*) as c FROM signal_log WHERE prediction_correct IS NOT NULL")
    total = total["c"] if total else 0
    if total == 0:
        print("\n暂无已结算数据。")
        return

    wins = db.execute_one("SELECT COUNT(*) as c FROM signal_log WHERE prediction_correct=1")["c"]
    avg_roi = db.execute_one("SELECT AVG(actual_roi) as v FROM signal_log WHERE prediction_correct IS NOT NULL")["v"] or 0
    pending = db.execute_one("SELECT COUNT(*) as c FROM signal_log WHERE prediction_correct IS NULL")["c"]

    print(f"\n━━━ 累计统计报告 ━━━")
    print(f"  已结算: {total}场 | 待结算: {pending}场")
    print(f"  预测胜率: {wins}/{total} ({wins/total:.1%})")
    print(f"  平均ROI: {avg_roi:+.2%}")

    # 推送 vs 未推送对比
    push_data = db.execute_one("""
        SELECT COUNT(*) as c, SUM(prediction_correct) as w, AVG(actual_roi) as r
        FROM signal_log WHERE was_pushed=1 AND prediction_correct IS NOT NULL
    """)
    nopush_data = db.execute_one("""
        SELECT COUNT(*) as c, SUM(prediction_correct) as w, AVG(actual_roi) as r
        FROM signal_log WHERE was_pushed=0 AND prediction_correct IS NOT NULL
    """)
    print(f"\n━━━ 推送 vs 未推送 ━━━")
    pc, pw, pr = push_data["c"] or 0, push_data["w"] or 0, push_data["r"] or 0
    nc, nw, nr = nopush_data["c"] or 0, nopush_data["w"] or 0, nopush_data["r"] or 0
    if pc > 0:
        print(f"  推送:   {pc}场 胜率{pw/pc:.1%} ROI{pr:+.2%}")
    if nc > 0:
        print(f"  未推送: {nc}场 胜率{nw/nc:.1%} ROI{nr:+.2%}")

    # 按评分区间
    buckets = db.execute("""
        SELECT
            CASE WHEN score >= 50 THEN '50+分(推送级)'
                 WHEN score >= 30 THEN '30-49分'
                 WHEN score >= 15 THEN '15-29分'
                 ELSE '0-14分' END as bucket,
            COUNT(*) as cnt,
            SUM(prediction_correct) as wins,
            AVG(actual_roi) as avg_roi
        FROM signal_log WHERE prediction_correct IS NOT NULL
        GROUP BY bucket ORDER BY bucket DESC
    """)
    print(f"\n━━━ 按评分区间 ━━━")
    print(f"  {'区间':14s} {'场次':>4s} {'胜率':>6s} {'平均ROI':>8s}")
    print(f"  {'─'*36}")
    for b in buckets:
        b = dict(b)
        cnt, bw = b["cnt"], b["wins"] or 0
        br = b["avg_roi"] or 0
        print(f"  {b['bucket']:14s} {cnt:4d} {bw/cnt:6.1%} {br:+7.2%}")

    # 按Edge区间
    edge_buckets = db.execute("""
        SELECT
            CASE WHEN raw_edge >= 0.03 THEN 'edge 3%+'
                 WHEN raw_edge >= 0.02 THEN 'edge 2-3%'
                 WHEN raw_edge >= 0.01 THEN 'edge 1-2%'
                 WHEN raw_edge >= 0.005 THEN 'edge 0.5-1%'
                 ELSE 'edge <0.5%' END as bucket,
            COUNT(*) as cnt,
            SUM(prediction_correct) as wins,
            AVG(actual_roi) as avg_roi
        FROM signal_log WHERE prediction_correct IS NOT NULL
        GROUP BY bucket ORDER BY bucket DESC
    """)
    print(f"\n━━━ 按Edge区间 ━━━")
    print(f"  {'区间':14s} {'场次':>4s} {'胜率':>6s} {'平均ROI':>8s}")
    print(f"  {'─'*36}")
    for b in edge_buckets:
        b = dict(b)
        cnt, bw = b["cnt"], b["wins"] or 0
        br = b["avg_roi"] or 0
        print(f"  {b['bucket']:14s} {cnt:4d} {bw/cnt:6.1%} {br:+7.2%}")

    # 按赔率来源
    source_buckets = db.execute("""
        SELECT COALESCE(source, 'unknown') as src,
            COUNT(*) as cnt,
            SUM(prediction_correct) as wins,
            AVG(actual_roi) as avg_roi
        FROM signal_log WHERE prediction_correct IS NOT NULL AND source IS NOT NULL
        GROUP BY src ORDER BY cnt DESC
    """)
    if source_buckets:
        print(f"\n━━━ 按赔率来源 ━━━")
        for b in source_buckets:
            b = dict(b)
            cnt, bw = b["cnt"], b["wins"] or 0
            br = b["avg_roi"] or 0
            print(f"  {b['src']:14s} {cnt:4d}场 胜率{bw/cnt:.1%} ROI{br:+.2%}")

    # 最近10条已结算
    recent = db.execute("""
        SELECT buy_team, away_team, home_team, score, raw_edge,
               prediction_correct, actual_roi, actual_score_away, actual_score_home
        FROM signal_log WHERE prediction_correct IS NOT NULL
        ORDER BY resolved_at DESC LIMIT 10
    """)
    if recent:
        print(f"\n━━━ 最近结算 ━━━")
        print(f"  {'买入':5s} {'比赛':12s} {'评分':>4s} {'edge':>7s} {'结果':4s} {'比分':>7s} {'ROI':>7s}")
        print(f"  {'─'*52}")
        for s in recent:
            r = dict(s)
            matchup = f"{r['away_team']}@{r['home_team']}"
            icon = "✅" if r["prediction_correct"] else "❌"
            score_str = f"{r['actual_score_away']}-{r['actual_score_home']}"
            roi = r["actual_roi"] or 0
            print(f"  {r['buy_team'] or '?':5s} {matchup:12s} {r['score']:4d} {r['raw_edge']:+6.2%} {icon:4s} {score_str:>7s} {roi:+6.0%}")


def _bootstrap_features_and_train(predictor):
    """为历史比赛构建特征并训练模型。"""
    import sqlite3
    con = sqlite3.connect("nba_predictor.db", timeout=30)
    con.row_factory = sqlite3.Row

    rows = con.execute("""
        SELECT game_id, home_team, away_team, game_date, home_won
        FROM historical_games WHERE features_json IS NULL
        ORDER BY game_date DESC LIMIT 1000
    """).fetchall()

    db = Database()
    espn = ESPNClient(db)
    fb = NBAFeatureBuilder(db, espn)

    updated = 0
    for r in rows:
        d = dict(r)
        try:
            # 用standings兜底（历史数据无法回溯nba_api game log）
            features = fb._fallback_features(d["home_team"], d["away_team"], d["game_date"])
            if features:
                con.execute(
                    "UPDATE historical_games SET features_json=? WHERE game_id=?",
                    (json.dumps(features), d["game_id"])
                )
                updated += 1
        except Exception:
            continue

    con.commit()
    con.close()
    print(f"  特征构建: {updated}场")

    result = predictor.train(force=True)
    print(f"  模型训练: {result['method']} | 样本:{result['samples']} | 准确率:{result['accuracy']:.1%}")


# ── 入口 ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="只运行一次")
    parser.add_argument("--health", action="store_true", help="数据源健康检查")
    parser.add_argument("--report", action="store_true", help="ML表现报告")
    parser.add_argument("--backtest", action="store_true", help="回测数据报告")
    parser.add_argument("--summary", action="store_true", help="立即发送每日报表")
    parser.add_argument("--review", action="store_true", help="立即运行每晚深度复盘")
    parser.add_argument("--stats", action="store_true", help="详细统计报告")
    parser.add_argument("--learn", action="store_true", help="手动运行一轮自学习")
    parser.add_argument("--learn-report", action="store_true", help="查看自学习进度报告")
    parser.add_argument("--bootstrap", action="store_true", help="导入历史比赛数据+训练预测模型")
    parser.add_argument("--notion-sync", action="store_true", help="同步历史推送到Notion")
    args = parser.parse_args()

    if args.health:
        sb = SportsbookClient(odds_api_key=ODDS_KEY)
        status = sb.health_check()
        print("\n━━━ 数据源健康检查 ━━━")
        for k, v in status.items():
            print(f"  {k:25s} {v}")
        return

    if args.report:
        ml = MLPredictor()
        report = ml.get_performance_report()
        print("\n━━━ ML推送表现报告 ━━━")
        print(f"  已结算推送:    {report['total_resolved']}")
        print(f"  胜率:          {report['win_rate']:.1%}" if report['win_rate'] else "  胜率: N/A（数据不足）")
        print(f"  平均PnL/单位:  {report['avg_pnl']:.4f}" if report['avg_pnl'] else "  平均PnL: N/A")
        print(f"  待结算:        {report['pending']}")
        ml_active = report['ml_active']
        samples_needed = report.get('samples_needed', 0)
        print(f"  ML启用:        {'✅' if ml_active else f'❌ 还需{samples_needed}条'}")
        if report.get("edge_breakdown"):
            print("\n  各Edge区间胜率:")
            for b in report["edge_breakdown"]:
                print(f"    {b['bucket']:12s}  {b['cnt']:3d}场  胜率:{b['win_rate']:.1%}")
        return

    if args.backtest:
        print_backtest_report()
        print()  # 分隔
        bt = Backtester()
        bt.print_report("kelly")
        return

    if args.summary:
        scanner = Scanner()
        scanner.resolve_results()
        scanner.send_daily_summary()
        return

    if args.review:
        scanner = Scanner()
        scanner.send_nightly_review()
        return

    if args.stats:
        print_stats_report()
        return

    if args.learn:
        learner = AutoLearner()
        print("\n━━━ 手动自学习循环 ━━━")
        result = learner.run_learning_cycle(n_experiments=50)
        print(f"  实验数: {result.get('experiments_run', 0)}")
        print(f"  改进:   {'✅ 是' if result.get('improved') else '❌ 否'}")
        print(f"  Baseline: {result.get('baseline_metric', 'N/A')}")
        print(f"  Best:     {result.get('best_metric', 'N/A')}")
        print(f"  样本数:   {result.get('samples', 0)}")
        if result.get("reason"):
            print(f"  跳过原因: {result['reason']}")
        return

    if args.learn_report:
        learner = AutoLearner()
        print(learner.get_learning_report())
        return

    if args.notion_sync:
        print("\n━━━ 同步历史推送到Notion ━━━")
        notion = NotionSync(NOTION_TOKEN, NOTION_DB_ID)
        db = Database()
        count = notion.sync_historical(db)
        print(f"  同步完成: {count}条推送记录")
        return

    if args.bootstrap:
        print("\n━━━ 导入历史比赛数据 ━━━")
        predictor = GamePredictor()
        count = predictor.bootstrap_historical()
        print(f"  导入: {count}场比赛")
        if count > 0:
            # 更新 OpenSkill 球队评分
            print("  更新OpenSkill球队评分...")
            db = Database()
            skill_ratings = TeamSkillRatings(db)
            skill_result = skill_ratings.update_from_historical()
            print(f"  OpenSkill: {skill_result.get('games_processed', 0)}场比赛处理")
            print(skill_ratings.get_report())
            print()
            print("  正在用历史数据构建特征并训练模型...")
            _bootstrap_features_and_train(predictor)
        return

    scanner = Scanner()

    if args.once:
        scanner.scan_games()
        scanner.scan_futures()
        return

    # 持续运行模式
    logger.info("🏀 NBA预测市场扫描器启动")

    # ── 启动初始化：强制刷新数据 ──
    logger.info("[启动] 强制刷新战绩...")
    scanner.espn.force_refresh_standings()

    logger.info("[启动] 更新球员评分...")
    update_player_ratings(scanner.db)

    logger.info("[启动] 更新OpenSkill球队评分...")
    try:
        skill_ratings = TeamSkillRatings(scanner.db)
        skill_ratings.update_from_historical()
    except Exception as e:
        logger.warning(f"[启动] OpenSkill更新失败: {e}")

    logger.info("[启动] 补全历史比赛特征...")
    _bootstrap_features_and_train(scanner.predictor)

    # 启动交互式AI Bot（后台线程）
    ai_bot = TelegramAIBot(scanner)
    ai_bot.start()

    schedule.every(1).hours.do(scanner.scan_games)
    schedule.every(1).hours.do(scanner.resolve_results)    # 每小时回填结果（免费端点）
    schedule.every().day.at("07:00").do(scanner.refresh_injury_data)  # 每日伤病刷新+清除康复球员
    schedule.every().day.at("08:00").do(scanner.scan_futures)
    schedule.every().day.at("09:00").do(scanner.send_daily_summary)  # 每日报表
    schedule.every().day.at("17:00").do(scanner.refresh_injury_data)  # 赛前再刷新一次伤病
    schedule.every().day.at("19:00").do(scanner.send_evening_picks)  # 每晚精选推送（赛前1-2h）
    schedule.every().day.at("23:30").do(scanner.send_nightly_review) # 每晚深度复盘（比赛结束后）
    schedule.every().day.at("00:00").do(scanner.reset_daily_count)
    schedule.every().day.at("06:00").do(lambda: update_player_ratings(scanner.db))

    # 启动时立即跑一次
    scanner.refresh_injury_data()
    scanner.scan_games()
    scanner.scan_futures()
    scanner.resolve_results()

    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    main()
