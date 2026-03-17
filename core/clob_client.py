"""
clob_client.py — Polymarket CLOB 价格数据

使用 /price 和 /midpoint 端点获取真实可交易价格。
注: /book 端点对sports市场返回stale数据(0.01/0.99)，不可用。

端点:
  GET /price?token_id=X&side=buy   → 真实买入价（你需要支付的价格）
  GET /price?token_id=X&side=sell  → 真实卖出价（你能收到的价格）
  GET /midpoint?token_id=X         → 中间价
无需认证（只读）。
"""

import logging
import requests
from typing import Optional

logger = logging.getLogger(__name__)

CLOB_BASE = "https://clob.polymarket.com"


def get_price(token_id: str, side: str = "buy", timeout: int = 8) -> Optional[float]:
    """获取token的真实价格。

    Args:
        token_id: CLOB token ID
        side: "buy" = 你买入需付的价格(ask), "sell" = 你卖出能得的价格(bid)

    Returns:
        价格(float)，或None
    """
    try:
        r = requests.get(
            f"{CLOB_BASE}/price",
            params={"token_id": token_id, "side": side},
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
        price = float(data.get("price", 0))
        return price if 0 < price < 1 else None
    except Exception as e:
        logger.debug(f"[CLOB] price获取失败 {token_id[:16]}...: {e}")
        return None


def get_midpoint(token_id: str, timeout: int = 8) -> Optional[float]:
    """获取token的中间价。"""
    try:
        r = requests.get(
            f"{CLOB_BASE}/midpoint",
            params={"token_id": token_id},
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
        mid = float(data.get("mid", 0))
        return mid if 0 < mid < 1 else None
    except Exception as e:
        logger.debug(f"[CLOB] midpoint获取失败 {token_id[:16]}...: {e}")
        return None


def get_market_prices(token_ids: list) -> dict:
    """获取双向token的真实交易价格。

    Args:
        token_ids: [away_token_id, home_token_id]

    Returns:
        {
            "away_buy": float,   # 买入客队token的价格(ask)
            "away_sell": float,  # 卖出客队token的价格(bid)
            "home_buy": float,   # 买入主队token的价格(ask)
            "home_sell": float,  # 卖出主队token的价格(bid)
            "away_mid": float,
            "home_mid": float,
            "away_spread": float,  # ask - bid
            "home_spread": float,
            "valid": bool,
        }
    """
    result = {
        "away_buy": None, "away_sell": None,
        "home_buy": None, "home_sell": None,
        "away_mid": None, "home_mid": None,
        "away_spread": None, "home_spread": None,
        "valid": False,
    }

    if not token_ids or len(token_ids) < 2:
        return result

    sides = [("away", token_ids[0]), ("home", token_ids[1])]
    for side, tid in sides:
        buy = get_price(tid, "buy")
        sell = get_price(tid, "sell")
        mid = get_midpoint(tid)

        if buy is not None:
            result[f"{side}_buy"] = buy
        if sell is not None:
            result[f"{side}_sell"] = sell
        if mid is not None:
            result[f"{side}_mid"] = mid
        if buy is not None and sell is not None:
            result[f"{side}_spread"] = round(buy - sell, 4)

    # 至少有away的buy价格才算有效
    result["valid"] = result["away_buy"] is not None and result["home_buy"] is not None
    return result
