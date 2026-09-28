from app.data_loader import get_logistics_by_order_id
from app.models import LogisticsInfo


class LogisticsService:
    """物流查询业务逻辑。

    对上层（Tool / Agent）屏蔽数据来源细节：当前数据来自 Mock JSON，
    后续切换为真实快递公司接口时，只需修改本类实现。
    """

    def get_logistics(self, order_id: str) -> list[LogisticsInfo]:
        """按订单号查询物流记录，一个订单可能拆分为多个包裹。

        Args:
            order_id: 订单号，例如 ``ORD-2026-0001``；首尾空格与大小写会被归一化。

        Returns:
            命中的 :class:`LogisticsInfo` 列表；订单号为空或该订单暂无物流记录时返回空列表。
        """
        normalized = order_id.strip().upper()
        if not normalized:
            return []
        return get_logistics_by_order_id(normalized)
