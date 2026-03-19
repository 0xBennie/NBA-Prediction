"""
mismatch_engine.py — 核心评分引擎（研究校准版）

研究依据:
  - B2B影响: ~2.26分分差, 胜算乘数0.62 (Small 2007, NESSIS)
  - 盘口移动: 连续量化而非布尔，越接近开赛越"硬" (赔率研究)
  - Edge计算: 用有效买入价(含点差)而非中间价 (Polymarket微观结构)
  - 成交量: 必须绑定深度/价差联合评估，防刷量假信号 (Polymarket刷量风险)
  - Kelly: 1/4 Kelly作为建议上限 (Thorp 2007)

单场比赛盘评分维度（0-100）:
  Pinnacle边际(edge)  40分  — 核心α来源：信息传播滞后
  盘口移动            25分  — 连续Δp + 跨市场响应差
  伤病影响            20分  — RAPTOR Elo惩罚差值
  价格位置            10分  — 低位=高赔率=更多上涨空间
  背靠背(B2B)          5分  — 量化2.26分影响→胜率修正

期货盘评分维度（0-100）:
  价格位置  25分
  基本面    25分
  动量      20分
  成交量+深度 20分  — 合并评分，量与深度必须同时满足
  结算时间  -10~0   — 远期风险敞口
"""

import json
import math
import time
import logging
from pathlib import Path
from typing import Optional

from core.database import Database
from core.sportsbook_client import SportsbookClient
from core.espn_client import ESPNClient
from core.clob_client import get_market_prices
from ml.ml_predictor import MLPredictor

logger = logging.getLogger(__name__)

GAME_THRESHOLD = 50
FUTURES_THRESHOLD = 50
FUTURES_MAX_PRICE = 0.40
MIN_VOLUME = 10000

# 研究校准的B2B参数 (Small 2007: 2.26分分差, 胜算乘数0.62)
B2B_SPREAD_IMPACT = 2.26   # 分差影响（点）
B2B_ODDS_MULTIPLIER = 0.62  # 胜算乘数（B2B vs 3+天休息）

# Polymarket交易成本估算（Limit order ~0.5% 手续费 + ~0.5% 滑点）
POLY_FEE_RATE = 0.01  # ~1%交易费+滑点（limit order场景）

PARAMS_PATH = Path("scoring_params.json")


def _load_scoring_params() -> dict:
    """从scoring_params.json加载自学习优化的参数。"""
    if PARAMS_PATH.exists():
        try:
            return json.loads(PARAMS_PATH.read_text())
        except Exception:
            pass
    return {}


class MismatchEngine:

    def __init__(self, db: Database, sb: SportsbookClient,
                 espn: ESPNClient, ml: MLPredictor,
                 predictor=None, features=None):
        self.db = db
        self.sb = sb
        self.espn = espn
        self.ml = ml
        self.predictor = predictor   # GamePredictor 实例
        self.features = features     # NBAFeatureBuilder 实例
        self.learned_params = _load_scoring_params()

    def reload_params(self):
        """重新加载自学习参数（每轮学习后调用）。"""
        self.learned_params = _load_scoring_params()
        logger.info(f"[Engine] 重载自学习参数 v{self.learned_params.get('_meta', {}).get('version', '?')}")

    # ── 比赛盘评分（预测优先版）──────────────────────────────────────
    def score_game(self, game: dict) -> Optional[dict]:
        """
        预测优先评分：先预测谁赢，再找市场低估的赢家，在低价时买入。

        逻辑:
          1. 模型预测 P(主队赢) → 确定买入方向（买预测的赢家）
          2. 混合 Pinnacle 赔率做第二意见
          3. 比较 混合概率 vs 市场价格 → 价值边际(value_edge)
          4. 只推送：模型>60%信心 + 市场低估5%+ + 价格在低位（利润空间大）
        """
        home = game["home_team"]
        away = game["away_team"]
        poly_away = game["polymarket_price_away"]
        poly_home = 1 - poly_away
        game_date = game.get("game_date", "")

        # ── 1. 获取 Pinnacle 公平概率 ──
        fair = self.sb.get_fair_prob(home, away)
        if fair is None:
            return None
        fair_home = fair["home"]
        fair_away = fair["away"]
        source = fair["source"]

        # ── 2. 模型预测：P(主队赢) ──
        model_home_prob = 0.5
        has_model = False
        game_features = None
        if self.predictor and self.features:
            try:
                game_features = self.features.get_game_features(home, away, game_date)
                if game_features:
                    model_home_prob = self.predictor.predict(game_features)
                    has_model = True
                else:
                    logger.warning(f"[Engine] {away}@{home} 无法构建特征，模型预测跳过")
            except Exception as e:
                logger.warning(f"[Engine] {away}@{home} 模型预测异常: {e}")
        model_away_prob = 1 - model_home_prob

        # ── 3. 混合概率 = log-odds 空间混合 + 市场锚定 ──
        #
        # 研究依据:
        #   - 线性混合 p1*w + p2*(1-w) 在极端概率时偏差大
        #   - log-odds 混合在数学上等价于独立信息源的贝叶斯更新
        #   - 当模型与市场分歧过大时，市场通常更准确（有效市场假说）
        #   - 数据验证：edge>5% 信号全输 → 大分歧时必须向市场收缩
        lp = self.learned_params
        pred_cfg = lp.get("prediction", {})
        base_model_weight = pred_cfg.get("model_weight", 0.15) if has_model else 0.0

        # 自适应模型权重: 模型需要积累 track record 才给权重
        # 研究(Wheatcroft 2024): 校准差的模型 ROI -35%, 校准好的 +35%
        # 数据验证: 当前模型最优权重=0%, 说明模型还需要更多训练
        # 公式: effective_w = base_w * min(1, resolved_signals / 200)
        resolved = self.db.execute_one(
            "SELECT COUNT(*) as c FROM signal_log WHERE prediction_correct IS NOT NULL"
        )
        n_resolved = resolved["c"] if resolved else 0
        model_weight = base_model_weight * min(1.0, n_resolved / 200.0)

        blended_home = self._bayesian_blend(
            model_home_prob, fair_home, poly_home, model_weight
        )
        blended_away = 1.0 - blended_home

        # ── 4. CLOB真实买入价 ──
        clob = game.get("clob")
        has_clob = clob is not None and clob.get("valid", False)
        if has_clob:
            away_buy_price = clob["away_buy"]
            home_buy_price = clob["home_buy"]
        else:
            away_buy_price = poly_away
            home_buy_price = poly_home
            logger.warning(
                f"[Engine] {away}@{home} CLOB不可用，回退到Gamma中间价 "
                f"(away={poly_away:.4f} home={poly_home:.4f})，边际计算可能偏高"
            )

        # ── 5. 确定买入方向：买模型认为会赢且市场低估的一方 ──
        home_value = blended_home - home_buy_price
        away_value = blended_away - away_buy_price

        if home_value >= away_value:
            buy_side = "home"
            buy_price = home_buy_price
            model_prob = model_home_prob
            blended_prob = blended_home
            pinnacle_prob = fair_home
            value_edge = home_value
        else:
            buy_side = "away"
            buy_price = away_buy_price
            model_prob = model_away_prob
            blended_prob = blended_away
            pinnacle_prob = fair_away
            value_edge = away_value

        raw_edge = pinnacle_prob - buy_price

        # edge 可信度衰减 — 大 edge 不代表大机会，代表大分歧（通常是模型错）
        # 数据验证: edge 0.5-3% 甜区(正ROI), edge>5% 全输
        edge_confidence = 1.0 / (1.0 + (value_edge / 0.03) ** 2)
        effective_value_edge = value_edge * edge_confidence

        # ── 6. 伤病 + B2B ──
        home_injury = self.espn.get_injury_impact(home)
        away_injury = self.espn.get_injury_impact(away)
        injury_delta = home_injury - away_injury
        home_b2b = self.espn.is_back_to_back(home, game_date) if game_date else False
        away_b2b = self.espn.is_back_to_back(away, game_date) if game_date else False

        # ── 7. 盘口移动 ──
        line_movement = self._measure_line_movement(game, fair)

        # ── 8. 评分（预测优先，0-100）──
        breakdown = {}

        # 8.1 模型信心 (0-35): 模型预测赢家的概率
        if model_prob >= 0.80:
            confidence_score = 35
        elif model_prob >= 0.70:
            confidence_score = 25 + (model_prob - 0.70) / 0.10 * 10
        elif model_prob >= 0.60:
            confidence_score = 15 + (model_prob - 0.60) / 0.10 * 10
        elif model_prob >= 0.55:
            confidence_score = 8 + (model_prob - 0.55) / 0.05 * 7
        else:
            confidence_score = max(0, model_prob - 0.45) / 0.10 * 8
        breakdown["model_confidence"] = {
            "score": round(confidence_score, 1),
            "model_prob": round(model_prob, 4),
            "has_model": has_model,
        }

        # 8.2 价值边际 (0-30): 用衰减后的 effective edge 评分
        # 这确保大 edge 不会得高分（大 edge = 模型错误的概率更高）
        ee = effective_value_edge
        if ee >= 0.04:
            value_score = 30
        elif ee >= 0.03:
            value_score = 20 + (ee - 0.03) / 0.01 * 10
        elif ee >= 0.02:
            value_score = 10 + (ee - 0.02) / 0.01 * 10
        elif ee >= 0.01:
            value_score = (ee - 0.01) / 0.01 * 10
        else:
            value_score = 0
        breakdown["value_edge"] = {
            "score": round(value_score, 1),
            "edge": round(value_edge, 4),
            "effective_edge": round(effective_value_edge, 4),
            "edge_confidence": round(edge_confidence, 4),
            "blended_prob": round(blended_prob, 4),
            "market_price": round(buy_price, 4),
        }

        # 8.3 Pinnacle一致性 (0-20): 模型和Pinnacle是否同意赢家
        pinnacle_agrees = (
            (buy_side == "home" and fair_home > 0.5) or
            (buy_side == "away" and fair_away > 0.5)
        )
        agree_score = 20 if pinnacle_agrees else 0
        breakdown["pinnacle_agreement"] = {
            "score": agree_score,
            "agrees": pinnacle_agrees,
            "pinnacle_prob": round(pinnacle_prob, 4),
        }

        # 8.4 伤病优势 (0-10)
        effective_injury = injury_delta if buy_side == "away" else -injury_delta
        inj_score = min(10, max(0, effective_injury / 5))
        breakdown["injury_impact"] = {
            "score": round(inj_score, 1),
            "home_elo_penalty": round(home_injury, 1),
            "away_elo_penalty": round(away_injury, 1),
            "delta": round(injury_delta, 1),
        }

        # 8.5 休息优势 (0-5)
        b2b_score = self._score_b2b(home_b2b, away_b2b, pinnacle_prob, buy_side)
        breakdown["b2b"] = {
            "score": round(b2b_score, 1),
            "home_b2b": home_b2b,
            "away_b2b": away_b2b,
        }

        # CLOB信息
        breakdown["has_clob"] = has_clob
        if has_clob:
            side_prefix = buy_side
            breakdown["clob"] = {
                "buy_price": clob[f"{side_prefix}_buy"],
                "sell_price": clob[f"{side_prefix}_sell"],
                "spread": clob[f"{side_prefix}_spread"],
            }
        breakdown["raw_edge"] = round(raw_edge, 4)
        breakdown["effective_edge"] = round(value_edge, 4)
        breakdown["data_quality"] = {
            "has_model": has_model,
            "has_clob": has_clob,
            "price_source": "clob" if has_clob else "gamma_mid",
        }

        # 盘口移动（保留记录）
        breakdown["line_movement"] = {
            "score": round(self._score_line_movement(line_movement), 1),
            "discordance": round(line_movement.get("discordance", 0), 4),
        }

        final_score = int(round(min(100, max(0,
            confidence_score + value_score + agree_score + inj_score + b2b_score
        ))))

        # ── 9. Kelly仓位 ──
        kp = lp.get("kelly", {})
        kelly_frac = kp.get("fraction", 0.25)
        kelly_max = kp.get("max_size", 0.05)
        if effective_value_edge > 0 and buy_price > 0 and buy_price < 1:
            b_win = (1 - buy_price) / buy_price
            kelly = blended_prob / 1.0 - (1 - blended_prob) / b_win
            # Kelly 也乘以 edge_confidence，大分歧时自动缩小仓位
            kelly = max(0, min(kelly * kelly_frac * edge_confidence, kelly_max))
        else:
            kelly = 0

        # ── 10. 推送条件：根据主客队分别设定门槛（研究: 买客队32%胜率远低于主队70%）──
        min_confidence = pred_cfg.get("min_model_confidence", 0.60)
        min_value = pred_cfg.get("min_value_edge", 0.05)
        max_buy_price = pred_cfg.get("max_buy_price", 0.65)

        # 客队需要更高门槛（基于历史数据：客队胜率仅32%）
        # 但不是一棒子打死：强客队(如客场胜率>55%+高eFG%)可以放宽
        if buy_side == "away":
            away_venue_wp = game_features.get("away_venue_win_pct", 0.5) if game_features else 0.5
            away_penalty = game_features.get("away_confidence_penalty", 0) if game_features else 0
            # 客队客场胜率>55% → 强客队，正常门槛
            # 客队客场胜率<40% → 弱客队，大幅提高门槛
            if away_venue_wp >= 0.55:
                adj_confidence = min_confidence  # 强客队不惩罚
            elif away_venue_wp >= 0.45:
                adj_confidence = min_confidence + 0.05  # 一般客队+5%
            else:
                adj_confidence = min_confidence + 0.12  # 弱客队+12%
            adj_confidence += away_penalty
        else:
            adj_confidence = min_confidence

        # 数据驱动的过滤规则（基于506场回测，去重后89场）:
        #   edge 0.5-3% = 甜区（37.7%胜率，正ROI）
        #   edge 3-5% = 可以（36.8%胜率）
        #   edge >5% = 纯噪声（客队0%胜率，主队28%） → 严格过滤
        #   <28¢ 只有20%胜率 → 提高下限
        #   客队买入整体21.7%胜率 → 需要更严格的门槛
        #   推送(去重后) 61.5%胜率 → 评分高的信号质量好

        # 推送条件（使用 effective_value_edge，已经包含了可信度衰减）
        # edge_confidence 已经自动处理了"大edge=不可信"的问题：
        #   raw_edge=2% → effective=1.7% (可信度85%)
        #   raw_edge=5% → effective=1.1% (可信度22%)  → 自然达不到min_value
        #   raw_edge=10% → effective=0.9% (可信度9%) → 完全不推
        should_push = (
            model_prob >= adj_confidence          # 根据主客场动态调整的信心门槛
            and effective_value_edge >= min_value  # 衰减后的edge仍有足够价值
            and buy_price <= max_buy_price         # 价格上限
            and buy_price >= 0.30                  # 不买<30¢冷门
            and final_score >= 45                  # 最低评分门槛（提高到45）
            and pinnacle_agrees                    # Pinnacle必须与模型一致
        )
        breakdown["adj_confidence_threshold"] = round(adj_confidence, 4)

        # 11. 保存价格历史
        volume_24h = game.get("volume_24h", 0)
        self._save_price_history(game, poly_home, poly_away, fair_home, fair_away, volume_24h)

        # 12. 如果推送，保存特征
        if should_push:
            push_id = f"game_{game['game_id']}"
            ml_features = {"model_prob": model_prob, "value_edge": value_edge,
                           "buy_price": buy_price, "blended_prob": blended_prob}
            self.ml.save_push(push_id, "game", ml_features, final_score, 0)
            self.db.insert("""
                INSERT OR IGNORE INTO push_results
                    (push_id, market_type, game_id, away_team, home_team,
                     poly_price_at_push, pinnacle_prob, edge_at_push)
                VALUES (?,?,?,?,?,?,?,?)
            """, (push_id, "game", game["game_id"], away, home,
                  buy_price, blended_prob, value_edge))

        return {
            "score": final_score,
            "push": should_push,
            "buy_side": buy_side,
            "edge": raw_edge,
            "effective_edge": effective_value_edge,
            "raw_value_edge": value_edge,
            "model_prob": model_prob,
            "blended_prob": blended_prob,
            "fair_prob": pinnacle_prob,
            "poly_price": buy_price,
            "fair_prob_away": fair_away,
            "poly_price_away": poly_away,
            "away_edge": blended_away - away_buy_price,
            "home_edge": blended_home - home_buy_price,
            "source": source,
            "kelly": kelly,
            "home_b2b": home_b2b,
            "away_b2b": away_b2b,
            "breakdown": breakdown,
        }

    # ── 贝叶斯概率混合 ──────────────────────────────────────────────
    @staticmethod
    def _bayesian_blend(model_prob: float, pinnacle_prob: float,
                        market_price: float, model_weight: float) -> float:
        """在 log-odds 空间混合概率，并向市场价格做贝叶斯收缩。

        数学原理:
          1. log-odds 混合 = 独立信息源的贝叶斯后验
             logit(blended) = w * logit(model) + (1-w) * logit(pinnacle)
          2. 分歧收缩: 当 blended 与 market 差距过大时，
             用 shrinkage 函数把 blended 拉向 market。
             原因: edge>5% 历史数据全输 → 大分歧 = 模型错误

        公式:
          raw_blend = logit_mix(model, pinnacle, w=0.3)
          divergence = |raw_blend - market|
          shrinkage = 1 / (1 + (divergence/0.03)^2)  # 3%以内几乎不收缩，5%+大幅收缩
          final = market + shrinkage * (raw_blend - market)
        """
        # 边界保护
        eps = 0.02
        model_prob = max(eps, min(1 - eps, model_prob))
        pinnacle_prob = max(eps, min(1 - eps, pinnacle_prob))
        market_price = max(eps, min(1 - eps, market_price))

        if model_weight <= 0:
            raw_blend = pinnacle_prob
        else:
            # log-odds 空间混合
            def logit(p):
                return math.log(p / (1 - p))
            def inv_logit(x):
                return 1.0 / (1.0 + math.exp(-x))

            log_model = logit(model_prob)
            log_pinnacle = logit(pinnacle_prob)
            log_blend = model_weight * log_model + (1 - model_weight) * log_pinnacle
            raw_blend = inv_logit(log_blend)

        # 分歧收缩: edge 越大越不可信
        divergence = abs(raw_blend - market_price)
        # Cauchy 收缩: 3% 以内几乎无衰减，5%+ 快速衰减到接近 0
        shrinkage = 1.0 / (1.0 + (divergence / 0.03) ** 2)

        final = market_price + shrinkage * (raw_blend - market_price)
        return max(eps, min(1 - eps, final))

    # ── B2B量化 ────────────────────────────────────────────────────
    def _b2b_prob_adjustment(self, home_b2b: bool, away_b2b: bool, base_away_prob: float) -> float:
        """将B2B的分差影响(2.26分)转为胜率修正。

        研究: B2B vs 3+天休息 → 分差-2.26分, 胜算×0.62
        用logistic映射: 2.26分 ≈ 3.3%胜率变动 (D=17的logistic模型)
        """
        if home_b2b == away_b2b:
            return 0.0  # 双方都B2B或都不B2B

        # 2.26分分差 → 胜率变动（logistic模型D=17）
        prob_shift = B2B_SPREAD_IMPACT / 17 * 0.25  # ≈ 0.033 (3.3%)

        if home_b2b and not away_b2b:
            return prob_shift   # 主队B2B → 客队利好+3.3%
        else:
            return -prob_shift  # 客队B2B → 客队不利-3.3%

    def _score_b2b(self, home_b2b: bool, away_b2b: bool, buy_fair: float, buy_side: str) -> float:
        """B2B评分 (0-5)"""
        adj = self._b2b_prob_adjustment(home_b2b, away_b2b, buy_fair)
        # adj > 0 表示客队受益; 根据买入方向判断是否加分
        if buy_side == "away":
            return 5 if adj > 0.02 else 0
        else:  # home
            return 5 if adj < -0.02 else 0

    # ── 盘口移动量化 ──────────────────────────────────────────────
    def _measure_line_movement(self, game: dict, current_fair: dict) -> dict:
        """量化盘口移动，返回连续特征。

        研究依据: 盘口变动包含信息，越接近开赛越"硬"。
        核心α = discordance (跨市场响应差) = Pinnacle动了但Poly没跟上。
        """
        history = self._get_price_history_with_fair(game["game_id"])
        if not history or len(history) < 2:
            return {"poly_shift": 0, "fair_shift": 0, "discordance": 0, "speed": 0}

        latest = history[0]   # 最新
        earliest = history[-1]  # 最早

        poly_shift = latest["poly"] - earliest["poly"] if latest["poly"] and earliest["poly"] else 0
        fair_shift = (current_fair["away"] - earliest["fair"]) if earliest["fair"] else 0

        # 跨市场响应差 = Pinnacle概率变化 - Polymarket价格变化
        # 正值 = Pinnacle向客队方向动了但Poly没跟上 = 我们的α
        discordance = fair_shift - poly_shift

        # 变化速度 = |Δp| / 时间间隔(小时)
        time_diff = max(latest["ts"] - earliest["ts"], 1) / 3600
        speed = abs(poly_shift) / time_diff if time_diff > 0 else 0

        return {
            "poly_shift": poly_shift,
            "fair_shift": fair_shift,
            "discordance": discordance,
            "speed": speed,
        }

    def _score_line_movement(self, lm: dict) -> float:
        """盘口移动评分 (0-25)

        优先看discordance（跨市场响应差），这是核心α来源。
        """
        disc = abs(lm.get("discordance", 0))
        poly_shift = abs(lm.get("poly_shift", 0))
        speed = lm.get("speed", 0)

        score = 0

        # 跨市场响应差（核心信号）
        if disc >= 0.05:
            score = 20
        elif disc >= 0.03:
            score = 15
        elif disc >= 0.02:
            score = 10
        elif disc >= 0.01:
            score = 5

        # Polymarket自身价格变动（补充信号）
        if poly_shift >= 0.05:
            score = max(score, 15)
        elif poly_shift >= 0.03:
            score = max(score, 10)

        # 快速变动加分（可能是突发伤病/轮休消息）
        if speed >= 0.05:  # 每小时>5%变动
            score = min(25, score + 5)

        return min(25, score)

    # ── 期货盘评分（预测优先版）──────────────────────────────────────
    def score_futures(self, market: dict) -> Optional[dict]:
        """
        期货盘预测优先评分。

        逻辑：用球队赛季表现预测期货结果概率，与市场价格比较找低估值。
        只推送：基本面强(预测概率高) + 市场低估 + 价格低(利润空间大)
        """
        price = market.get("price", 0)
        volume = market.get("volume", 0)
        volume_24h = market.get("volume_24h", 0)
        days = market.get("days_to_resolution", 999)
        team = market.get("team", "")
        mtype = market.get("type", "")

        # 基础过滤
        if price <= 0.01 or price > FUTURES_MAX_PRICE:
            return {"score": 0, "push": False, "skip": True}
        if volume < MIN_VOLUME:
            return {"score": 0, "push": False, "skip": True}
        if days > 120:
            return {"score": 0, "push": False, "skip": True}

        breakdown = {}

        # ── 1. 基本面预测概率 ──
        # 用球队赛季数据估算期货实现概率
        model_prob = self._estimate_futures_prob(team, mtype, price)
        value_edge = model_prob - price  # 我们认为的概率 - 市场价格

        breakdown["model_prob"] = round(model_prob, 4)
        breakdown["value_edge"] = round(value_edge, 4)
        breakdown["market_price"] = round(price, 4)

        # ── 2. 评分（预测优先，0-100）──

        # 2.1 基本面信心 (0-35): 我们预测这个结果多大概率发生
        if model_prob >= 0.40:
            confidence_score = 35
        elif model_prob >= 0.30:
            confidence_score = 25 + (model_prob - 0.30) / 0.10 * 10
        elif model_prob >= 0.20:
            confidence_score = 15 + (model_prob - 0.20) / 0.10 * 10
        elif model_prob >= 0.10:
            confidence_score = 5 + (model_prob - 0.10) / 0.10 * 10
        else:
            confidence_score = 0
        breakdown["confidence"] = round(confidence_score, 1)

        # 2.2 价值边际 (0-30): model_prob - market_price
        if value_edge >= 0.15:
            value_score = 30
        elif value_edge >= 0.10:
            value_score = 20 + (value_edge - 0.10) / 0.05 * 10
        elif value_edge >= 0.05:
            value_score = 10 + (value_edge - 0.05) / 0.05 * 10
        elif value_edge >= 0.02:
            value_score = (value_edge - 0.02) / 0.03 * 10
        else:
            value_score = 0
        breakdown["value_score"] = round(value_score, 1)

        # 2.3 价格位置 (0-20): 低价=高赔率=大利润空间
        if price <= 0.10:
            price_score = 20  # 10x赔率
        elif price <= 0.15:
            price_score = 18
        elif price <= 0.20:
            price_score = 15
        elif price <= 0.25:
            price_score = 10
        elif price <= 0.30:
            price_score = 5
        else:
            price_score = 0
        breakdown["price_position"] = price_score

        # 2.4 动量 (0-10): 价格趋势
        daily_change = 0
        weekly_change = 0
        p24 = market.get("price_24h_ago")
        p7d = market.get("price_7d_ago")
        if p24 and p24 > 0:
            daily_change = (price - p24) / p24
        if p7d and p7d > 0:
            weekly_change = (price - p7d) / p7d
        momentum_score = min(10, max(0, int(abs(daily_change) * 100)))
        breakdown["momentum"] = {
            "score": momentum_score,
            "daily_change": round(daily_change, 4),
            "weekly_change": round(weekly_change, 4),
        }

        # 2.5 结算时间惩罚 (-5~0)
        settle_penalty = -5 if days > 90 else (-3 if days > 60 else 0)
        breakdown["settle_penalty"] = settle_penalty

        final_score = int(round(min(100, max(0,
            confidence_score + value_score + price_score + momentum_score + settle_penalty
        ))))

        # ── 3. Kelly仓位 ──
        lp = self.learned_params
        kp = lp.get("kelly", {})
        kelly_frac = kp.get("fraction", 0.25)
        kelly_max = kp.get("max_size", 0.05)
        if value_edge > 0 and price > 0 and price < 1:
            b_win = (1 - price) / price
            kelly = model_prob / 1.0 - (1 - model_prob) / b_win
            kelly = max(0, min(kelly * kelly_frac, kelly_max))
        else:
            kelly = 0

        # ── 4. 推送条件（只推最高确定性）──
        pred_cfg = lp.get("prediction", {})
        should_push = (
            model_prob >= 0.20            # 基本面概率至少20%（强队才推）
            and value_edge >= 0.08        # 市场低估8%+（比比赛盘更严格）
            and price <= 0.25             # 低价买入（4x+赔率）
            and final_score >= 50         # 高评分门槛
        )

        # ── 5. 保存 ──
        if should_push:
            push_id = f"futures_{market['condition_id']}"
            features = {"model_prob": model_prob, "value_edge": value_edge,
                        "price": price, "team": team, "type": mtype}
            self.ml.save_push(push_id, "futures", features, final_score, 0)
            self.db.insert("""
                INSERT OR IGNORE INTO push_results
                    (push_id, market_type, game_id, poly_price_at_push, edge_at_push)
                VALUES (?,?,?,?,?)
            """, (push_id, "futures", market["condition_id"], price, value_edge))

        return {
            "score": final_score,
            "push": should_push,
            "kelly": kelly,
            "model_prob": model_prob,
            "value_edge": value_edge,
            "breakdown": breakdown,
        }

    def _estimate_futures_prob(self, team: str, mtype: str, market_price: float) -> float:
        """用球队赛季数据估算期货实现概率。

        champion: 冠军概率 ≈ f(胜率, 净胜分) — 强队高、弱队低
        mvp/roy: 没有好的预测方法，用市场价格打折
        division: 赛区第一概率 ≈ f(胜率 vs 赛区对手)
        playoff: 进季后赛概率 ≈ f(胜率)
        """
        if not team:
            return market_price * 0.9  # 无球队信息，信任市场

        standings = self.db.execute_one(
            "SELECT * FROM standings WHERE team_abbr=?", (team,)
        )
        if not standings:
            return market_price * 0.9

        wins = standings.get("wins", 0)
        losses = standings.get("losses", 0)
        total = wins + losses
        if total == 0:
            return market_price * 0.9

        win_pct = wins / total
        ppg_diff = standings.get("ppg_diff", 0)

        if mtype == "champion":
            # 冠军概率：强相关于胜率和净胜分
            # NBA历史：60%+胜率队约15-25%冠军概率，70%+约30-40%
            if win_pct >= 0.70:
                base = 0.25 + (win_pct - 0.70) * 2.0
            elif win_pct >= 0.60:
                base = 0.10 + (win_pct - 0.60) * 1.5
            elif win_pct >= 0.50:
                base = 0.03 + (win_pct - 0.50) * 0.7
            else:
                base = max(0.005, win_pct * 0.05)
            # 净胜分修正
            net_adj = max(-0.05, min(0.05, ppg_diff / 100))
            return max(0.01, min(0.50, base + net_adj))

        elif mtype == "playoff":
            # 进季后赛：50%+胜率基本稳进，40-50%有风险
            if win_pct >= 0.55:
                return min(0.95, 0.80 + (win_pct - 0.55) * 1.0)
            elif win_pct >= 0.45:
                return 0.40 + (win_pct - 0.45) * 4.0
            else:
                return max(0.05, win_pct * 0.8)

        elif mtype == "division":
            # 赛区冠军：需要比赛区其他队强
            if win_pct >= 0.60:
                return min(0.60, 0.30 + (win_pct - 0.60) * 3.0)
            elif win_pct >= 0.50:
                return 0.15 + (win_pct - 0.50) * 1.5
            else:
                return max(0.03, win_pct * 0.25)

        else:
            # mvp/roy等：没有好的基本面预测，信任市场价格打小折
            return market_price * 0.95

    # ── 内部方法 ────────────────────────────────────────────────────

    def _score_futures_liquidity(self, market: dict) -> int:
        """期货盘 成交量+深度联合评分 (0-20)

        研究: 成交量单独使用不可靠（刷量风险），必须与深度/价差联合。
        量+窄价差 = 真流动性 = 高分
        量+无价差数据 = 可能刷量 = 打折
        量少+窄价差 = 流动性一般 = 中分
        """
        volume = market.get("volume", 0)
        volume_24h = market.get("volume_24h", 0)
        spread = market.get("bid_ask_spread")

        has_depth = spread is not None and spread <= 0.03
        has_volume = volume_24h > 0 and volume > 0

        if has_volume and has_depth:
            # 量+深度都有: 真流动性
            ratio = volume_24h / volume
            if ratio > 0.05 and spread <= 0.01:
                return 20  # 异常活跃+极窄价差
            elif ratio > 0.02 and spread <= 0.02:
                return 15
            elif ratio > 0.01:
                return 10
            return 7
        elif has_volume and not has_depth:
            # 有量但无深度数据: 打折（可能刷量）
            ratio = volume_24h / volume
            if ratio > 0.05:
                return 10  # 原本15→打折到10
            elif ratio > 0.02:
                return 7
            return 4
        elif has_depth and not has_volume:
            # 有深度但量少
            if spread <= 0.01:
                return 8
            return 5
        else:
            # 都没有
            if volume > 500000:
                return 5  # 高总量兜底
            return 0

    def _score_futures_fundamental(self, market: dict) -> int:
        """期货盘基本面评分 (0-25)"""
        team = market.get("team", "")
        mtype = market.get("type", "")
        price = market.get("price", 0)

        if not team:
            return 5

        standings = self.db.execute_one(
            "SELECT * FROM standings WHERE team_abbr=?", (team,)
        )
        if not standings:
            return 5

        wins = standings.get("wins", 0)
        losses = standings.get("losses", 0)
        total = wins + losses
        if total == 0:
            return 5

        win_pct = wins / total
        ppg_diff = standings.get("ppg_diff", 0)

        if mtype == "champion":
            if win_pct >= 0.65 and price <= 0.20:
                return 25
            elif win_pct >= 0.55 and price <= 0.15:
                return 20
            elif win_pct >= 0.50 and price <= 0.10:
                return 15
            elif win_pct >= 0.45:
                return 5
        else:
            if ppg_diff > 5 and price <= 0.25:
                return 20
            elif ppg_diff > 0 and price <= 0.20:
                return 15
            elif win_pct >= 0.50:
                return 10

        return 5

    def _get_price_history_with_fair(self, game_id: str) -> list:
        """从price_history取最近的价格+公平概率"""
        rows = self.db.execute("""
            SELECT poly_price_away, pinnacle_fair_away, timestamp
            FROM price_history
            WHERE game_id=? ORDER BY timestamp DESC LIMIT 10
        """, (game_id,))
        result = []
        for r in rows:
            d = dict(r)
            result.append({
                "poly": d.get("poly_price_away"),
                "fair": d.get("pinnacle_fair_away"),
                "ts": d.get("timestamp", 0),
            })
        return result

    def _save_price_history(self, game: dict, poly_home: float, poly_away: float,
                            fair_home: float, fair_away: float, volume_24h):
        """保存价格快照到时序表"""
        try:
            self.db.insert("""
                INSERT INTO price_history
                    (game_id, timestamp, poly_price_home, poly_price_away,
                     pinnacle_fair_home, pinnacle_fair_away, volume_24h, market_type)
                VALUES (?,?,?,?,?,?,?,?)
            """, (game["game_id"], int(time.time()), poly_home, poly_away,
                  fair_home, fair_away, float(volume_24h) if volume_24h else 0, "game"))
        except Exception as e:
            logger.debug(f"[Engine] price_history insert: {e}")
