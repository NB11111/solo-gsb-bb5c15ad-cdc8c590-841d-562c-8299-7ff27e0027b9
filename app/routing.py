"""路由试算引擎：共享连续补充令牌桶上的多候选链选择。

每个请求携带多条候选桶链，仅当某条候选链上**所有桶**余额都充足时该链
才可选；在可选链中选择"扣减后链上最大占用率最小"的一条（占用率 =
(capacity - 扣减后余额) / capacity，平局按候选顺序），仅扣选中链上的桶。
无可选链时所有桶原子保持不变，并给出每条候选链上首个余额不足的桶。
"""
from __future__ import annotations

from app.models import (
    EPS,
    BucketBalance,
    RouteCandidateRejection,
    RouteResult,
    RouteSimulateRequest,
    RouteSimulateResponse,
)
from app.simulation import _round


def route_simulate(payload: RouteSimulateRequest) -> RouteSimulateResponse:
    if not payload.buckets:
        raise ValueError("buckets 不能为空")
    if not payload.requests:
        raise ValueError("requests 不能为空")

    # 模拟开始时刻 = 最早请求时间戳：所有桶在该时刻持有初始余额，
    # 之后按 (now - last) * rate 连续补充，上限为容量（与 /simulate 一致）。
    t_start = min(request.timestamp for request in payload.requests)
    t_end = max(request.timestamp for request in payload.requests)

    capacity = {b.id: float(b.capacity) for b in payload.buckets}
    refill_rate = {b.id: float(b.refill_rate) for b in payload.buckets}
    balance = {
        b.id: float(b.capacity if b.initial is None else b.initial) for b in payload.buckets
    }
    last_time = {b.id: t_start for b in payload.buckets}

    def refill(bucket_id: str, now: float) -> float:
        """把某个桶补充到时刻 now，返回补充后的余额。"""
        elapsed = now - last_time[bucket_id]
        if elapsed > 0:
            balance[bucket_id] = min(
                capacity[bucket_id], balance[bucket_id] + refill_rate[bucket_id] * elapsed
            )
        last_time[bucket_id] = now
        return balance[bucket_id]

    # 内部按时间顺序处理；同一时刻按原始输入顺序处理。
    order = sorted(
        range(len(payload.requests)),
        key=lambda i: (payload.requests[i].timestamp, i),
    )

    results_by_index: dict[int, RouteResult] = {}

    for i in order:
        request = payload.requests[i]
        now = request.timestamp
        cost = float(request.cost)

        # 该请求涉及的所有桶（按候选顺序、链内顺序去重）；
        # 无论最终是否放行，被触及的桶时间都会推进
        involved: list[str] = []
        seen: set[str] = set()
        for chain in request.candidates:
            for bucket_id in chain:
                if bucket_id not in seen:
                    seen.add(bucket_id)
                    involved.append(bucket_id)
        refilled = {bucket_id: refill(bucket_id, now) for bucket_id in involved}

        # 逐候选评估：链上全部充足才可选；可选链计扣减后链上最大占用率
        best_index: int | None = None
        best_score: float | None = None
        rejections: list[RouteCandidateRejection] = []
        for c_index, chain in enumerate(request.candidates):
            rejected_by: str | None = None
            for bucket_id in chain:
                if refilled[bucket_id] + EPS < cost:
                    rejected_by = bucket_id
                    break
            if rejected_by is not None:
                rejections.append(
                    RouteCandidateRejection(
                        candidate_index=c_index,
                        chain=list(chain),
                        rejected_by=rejected_by,
                    )
                )
                continue
            score = max(
                (capacity[bucket_id] - (refilled[bucket_id] - cost))
                / capacity[bucket_id]
                for bucket_id in chain
            )
            # 平局（EPS 容差内）保留候选顺序靠前者
            if best_score is None or score < best_score - EPS:
                best_score = score
                best_index = c_index

        if best_index is not None:
            # 仅扣选中链上的桶
            selected_chain = list(request.candidates[best_index])
            balances_view: dict[str, BucketBalance] = {}
            for bucket_id in selected_chain:
                before = refilled[bucket_id]
                balance[bucket_id] = before - cost
                balances_view[bucket_id] = BucketBalance(
                    before=_round(before), after=_round(balance[bucket_id])
                )
            results_by_index[i] = RouteResult(
                request_id=request.id,
                index=i,
                timestamp=_round(now),
                allowed=True,
                selected_index=best_index,
                selected_chain=selected_chain,
                rejections=None,
                balances=balances_view,
            )
        else:
            # 原子性：拒绝时所有桶保持不变，说明各候选首个不足桶
            results_by_index[i] = RouteResult(
                request_id=request.id,
                index=i,
                timestamp=_round(now),
                allowed=False,
                selected_index=None,
                selected_chain=None,
                rejections=rejections,
                balances={
                    bucket_id: BucketBalance(
                        before=_round(refilled[bucket_id]),
                        after=_round(refilled[bucket_id]),
                    )
                    for bucket_id in involved
                },
            )

    # 把所有未再参与请求的桶补充到最后时刻，给出终态余额
    final_balances = {
        bucket_id: _round(
            min(
                capacity[bucket_id],
                balance[bucket_id]
                + refill_rate[bucket_id] * max(0.0, t_end - last_time[bucket_id]),
            )
        )
        for bucket_id in balance
    }

    results = [results_by_index[i] for i in range(len(payload.requests))]
    allowed_count = sum(1 for r in results if r.allowed)

    return RouteSimulateResponse(
        allowed_count=allowed_count,
        rejected_count=len(results) - allowed_count,
        results=results,
        final_balances=final_balances,
    )
