from datetime import datetime
from typing import Literal

from pydantic import BaseModel

OrderStatus = Literal["pending", "paid", "shipped", "delivered", "cancelled"]
# 付款状态与订单状态分开建模：取消只改订单状态，款项是否要退、退到哪一步由付款状态表达。
# refunding = 退款已受理、款项处理中（到账由支付渠道异步完成，本系统不写 refunded）。
PaymentStatus = Literal["unpaid", "paid", "refunding", "refunded"]
LogisticsStatus = Literal["pending", "in_transit", "delivered"]


class Order(BaseModel):
    order_id: str
    user_id: str
    product_name: str
    amount: float
    status: OrderStatus
    payment_status: PaymentStatus
    created_at: datetime
    # 签收时间：退款时效的唯一锚点；未签收（含未发货、在途、取消）为 None。
    delivered_at: datetime | None = None


class LogisticsInfo(BaseModel):
    order_id: str
    tracking_number: str
    carrier: str
    status: LogisticsStatus
    latest_location: str
    latest_update: datetime
