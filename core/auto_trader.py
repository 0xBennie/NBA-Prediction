"""
auto_trader.py — Polymarket自动下单模块

流程:
  1. 委员会审议通过 → 发送确认按钮到Telegram
  2. 用户点 ✅确认 → 下Limit Order
  3. 记录到数据库 + Notion

安全机制:
  - 所有订单必须用户在Telegram确认后才执行
  - 只下Limit Order（挂单），不吃市价
  - 单笔上限: 总资金5%
  - 日上限: 总资金15%
  - 全部交易记录可审计
"""

import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

CLOB_BASE = "https://clob.polymarket.com"


class AutoTrader:
    """Polymarket CLOB自动交易器。"""

    def __init__(self, api_key: str, api_secret: str, passphrase: str,
                 bankroll: float = 1000.0):
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.bankroll = bankroll
        self.daily_invested = 0.0
        self.max_single_pct = 0.05   # 单笔上限5%
        self.max_daily_pct = 0.15    # 日上限15%
        self.client = None
        self._init_client()

    def _init_client(self):
        """初始化CLOB客户端（使用钱包私钥签名）。"""
        try:
            from py_clob_client.client import ClobClient
            # key = 钱包私钥（用于签名交易）
            self.client = ClobClient(
                CLOB_BASE,
                key=self.api_key,  # 这里传私钥
                chain_id=137,  # Polygon
            )
            creds = self.client.create_or_derive_api_creds()
            self.client.set_api_creds(creds)
            addr = self.client.get_address()
            logger.info(f"[Trader] CLOB客户端初始化成功 钱包:{addr[:10]}...")
        except Exception as e:
            logger.warning(f"[Trader] CLOB客户端初始化失败: {e}")
            self.client = None

    def place_limit_order(self, token_id: str, side: str, price: float,
                          amount: float, game_id: str = "") -> dict:
        """下Limit Order。

        Args:
            token_id: CLOB token ID
            side: "BUY" or "SELL"
            price: 挂单价格 (0-1)
            amount: USDC金额
            game_id: 比赛ID（记录用）

        Returns:
            {"success": bool, "order_id": str, "message": str}
        """
        # 风控检查
        check = self._risk_check(amount)
        if not check["pass"]:
            return {"success": False, "order_id": "", "message": check["reason"]}

        if self.client is None:
            return {"success": False, "order_id": "", "message": "CLOB客户端未初始化"}

        try:
            from py_clob_client.order_builder.constants import BUY, SELL

            order_side = BUY if side.upper() == "BUY" else SELL
            # 计算size: amount / price = 买多少份token
            size = round(amount / price, 2)

            order = self.client.create_and_post_order(
                order_args={
                    "token_id": token_id,
                    "price": round(price, 2),
                    "size": size,
                    "side": order_side,
                }
            )

            order_id = order.get("orderID", "") if isinstance(order, dict) else str(order)
            self.daily_invested += amount

            logger.info(
                f"[Trader] 下单成功: {side} {size}份@{price:.2f} "
                f"金额${amount:.1f} 订单{order_id}"
            )

            return {
                "success": True,
                "order_id": order_id,
                "message": f"Limit {side} {size}份@{price:.0%} = ${amount:.0f}",
                "size": size,
                "price": price,
                "amount": amount,
            }

        except Exception as e:
            logger.warning(f"[Trader] 下单失败: {e}")
            return {"success": False, "order_id": "", "message": str(e)}

    def calculate_bet_size(self, kelly: float) -> float:
        """根据Kelly建议计算下注金额。"""
        kelly = min(kelly, self.max_single_pct)  # 上限5%
        amount = self.bankroll * kelly
        # 日上限检查
        remaining_daily = self.bankroll * self.max_daily_pct - self.daily_invested
        amount = min(amount, max(0, remaining_daily))
        return round(amount, 2)

    def get_status(self) -> dict:
        """获取交易状态（含真实余额和持仓）。"""
        wallet = "N/A"
        open_orders = []
        if self.client:
            try:
                wallet = self.client.get_address()
                # 获取未成交订单
                try:
                    orders = self.client.get_orders()
                    if orders:
                        open_orders = [
                            {"market": o.get("asset_id", "?")[:12],
                             "side": o.get("side", "?"),
                             "price": o.get("price", 0),
                             "size": o.get("original_size", 0),
                             "status": o.get("status", "?")}
                            for o in orders[:5]
                        ]
                except Exception:
                    pass
            except Exception:
                pass

        return {
            "bankroll": self.bankroll,
            "daily_invested": self.daily_invested,
            "daily_remaining": max(0, self.bankroll * self.max_daily_pct - self.daily_invested),
            "max_single": self.bankroll * self.max_single_pct,
            "client_ready": self.client is not None,
            "wallet": wallet,
            "open_orders": open_orders,
        }

    def reset_daily(self):
        """每日重置。"""
        self.daily_invested = 0.0
        logger.info("[Trader] 日投入重置")

    def _risk_check(self, amount: float) -> dict:
        """风控检查。"""
        max_single = self.bankroll * self.max_single_pct
        if amount > max_single:
            return {"pass": False, "reason": f"超过单笔上限${max_single:.0f} (5%)"}

        remaining = self.bankroll * self.max_daily_pct - self.daily_invested
        if amount > remaining:
            return {"pass": False, "reason": f"超过日上限，今日剩余${remaining:.0f}"}

        if amount < 1:
            return {"pass": False, "reason": "金额太小(<$1)"}

        return {"pass": True, "reason": ""}
