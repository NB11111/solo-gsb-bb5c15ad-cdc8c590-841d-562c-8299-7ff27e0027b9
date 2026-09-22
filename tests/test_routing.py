"""路由试算（/route-simulate）引擎与 HTTP 层测试：pytest 运行。"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.errors import RequestValidationErrors
from app.main import app
from app.models import RouteSimulateRequest
from app.routing import route_simulate

client = TestClient(app)


def _run(buckets, requests):
    return route_simulate(
        RouteSimulateRequest.model_validate({"buckets": buckets, "requests": requests})
    )


# ---------- 选路语义 ----------

def test_single_candidate_degenerates_to_simulate():
    # 只有一条候选链时行为与 /simulate 一致：充足放行、不足拒绝
    resp = _run(
        [{"id": "A", "capacity": 10, "refill_rate": 0}],
        [
            {"id": "r1", "timestamp": 0, "cost": 4, "candidates": [["A"]]},
            {"id": "r2", "timestamp": 0, "cost": 7, "candidates": [["A"]]},
        ],
    )
    r1, r2 = resp.results
    assert r1.allowed is True
    assert r1.selected_index == 0 and r1.selected_chain == ["A"]
    assert r1.rejections is None
    assert r1.balances["A"].before == 10 and r1.balances["A"].after == 6
    assert r2.allowed is False
    assert r2.selected_index is None and r2.selected_chain is None
    assert [rej.rejected_by for rej in r2.rejections] == ["A"]
    assert r2.balances["A"].after == 6


def test_picks_lowest_max_occupancy_after_deduction():
    # A 容量 10、B 容量 20（初始均为 10），同扣 4：A 扣后占用率 0.4，B 为 0.7 -> 选 A
    resp = _run(
        [
            {"id": "A", "capacity": 10, "refill_rate": 0},
            {"id": "B", "capacity": 20, "refill_rate": 0, "initial": 10},
        ],
        [{"id": "r1", "timestamp": 0, "cost": 4, "candidates": [["A"], ["B"]]}],
    )
    r = resp.results[0]
    assert r.allowed is True
    assert r.selected_chain == ["A"]
    assert r.balances["A"].after == 6
    assert resp.final_balances == {"A": 6, "B": 10}


def test_occupancy_uses_post_deduction_state():
    # A 已被上一笔压低：第二笔时 A 扣后占用率 0.8 > B 的 0.7 -> 改选 B
    resp = _run(
        [
            {"id": "A", "capacity": 10, "refill_rate": 0},
            {"id": "B", "capacity": 20, "refill_rate": 0, "initial": 10},
        ],
        [
            {"id": "r1", "timestamp": 0, "cost": 4, "candidates": [["A"], ["B"]]},
            {"id": "r2", "timestamp": 0, "cost": 4, "candidates": [["A"], ["B"]]},
        ],
    )
    assert resp.results[0].selected_chain == ["A"]
    assert resp.results[1].selected_chain == ["B"]
    assert resp.results[1].balances["B"].after == 6
    assert resp.final_balances == {"A": 6, "B": 6}


def test_tie_breaks_by_candidate_order():
    # 两条完全对称的候选链：占用率相同，按候选顺序选第一条
    resp = _run(
        [
            {"id": "A", "capacity": 10, "refill_rate": 0},
            {"id": "B", "capacity": 10, "refill_rate": 0},
        ],
        [{"id": "r1", "timestamp": 0, "cost": 5, "candidates": [["A"], ["B"]]}],
    )
    r = resp.results[0]
    assert r.selected_index == 0
    assert r.selected_chain == ["A"]
    assert resp.final_balances == {"A": 5, "B": 10}


def test_max_occupancy_is_per_chain_not_per_bucket():
    # 链 ["A", "C"]：A 扣后占用率 0.5、C 为 0.9 -> 链得分 0.9；
    # 链 ["B"]：扣后占用率 0.8 -> 选 ["B"]
    resp = _run(
        [
            {"id": "A", "capacity": 10, "refill_rate": 0},
            {"id": "B", "capacity": 10, "refill_rate": 0, "initial": 8},
            {"id": "C", "capacity": 10, "refill_rate": 0, "initial": 6},
        ],
        [{"id": "r1", "timestamp": 0, "cost": 5, "candidates": [["A", "C"], ["B"]]}],
    )
    r = resp.results[0]
    assert r.selected_index == 1
    assert r.selected_chain == ["B"]
    assert resp.final_balances == {"A": 10, "B": 3, "C": 6}


def test_only_selected_chain_is_deducted():
    resp = _run(
        [
            {"id": "A", "capacity": 10, "refill_rate": 0},
            {"id": "B", "capacity": 10, "refill_rate": 0},
            {"id": "C", "capacity": 10, "refill_rate": 0},
        ],
        [{"id": "r1", "timestamp": 0, "cost": 3, "candidates": [["A", "B"], ["C"]]}],
    )
    r = resp.results[0]
    assert r.allowed is True
    # 两条链扣后最大占用率都是 0.3，平局选前者 ["A", "B"]
    assert r.selected_chain == ["A", "B"]
    assert set(r.balances) == {"A", "B"}
    assert resp.final_balances == {"A": 7, "B": 7, "C": 10}


def test_infeasible_candidate_is_skipped():
    # 候选 0 的 B 不足 -> 跳过；候选 1 可行 -> 选候选 1
    resp = _run(
        [
            {"id": "A", "capacity": 10, "refill_rate": 0},
            {"id": "B", "capacity": 2, "refill_rate": 0},
            {"id": "C", "capacity": 10, "refill_rate": 0},
        ],
        [{"id": "r1", "timestamp": 0, "cost": 5, "candidates": [["A", "B"], ["C"]]}],
    )
    r = resp.results[0]
    assert r.allowed is True
    assert r.selected_index == 1
    assert r.selected_chain == ["C"]
    assert resp.final_balances == {"A": 10, "B": 2, "C": 5}


# ---------- 拒绝与原子性 ----------

def test_all_candidates_infeasible_rejects_atomically():
    # 候选 0：A 不足；候选 1：B 充足但 C 不足 -> 全部拒绝，所有桶不变
    resp = _run(
        [
            {"id": "A", "capacity": 4, "refill_rate": 0},
            {"id": "B", "capacity": 10, "refill_rate": 0},
            {"id": "C", "capacity": 3, "refill_rate": 0},
        ],
        [{"id": "r1", "timestamp": 0, "cost": 5, "candidates": [["A"], ["B", "C"]]}],
    )
    r = resp.results[0]
    assert r.allowed is False
    assert r.selected_index is None and r.selected_chain is None
    assert [(rej.candidate_index, rej.rejected_by) for rej in r.rejections] == [
        (0, "A"),
        (1, "C"),
    ]
    assert r.rejections[1].chain == ["B", "C"]
    # 原子性：涉及的所有桶 after == before
    assert r.balances["A"].before == 4 and r.balances["A"].after == 4
    assert r.balances["B"].before == 10 and r.balances["B"].after == 10
    assert r.balances["C"].before == 3 and r.balances["C"].after == 3
    assert resp.final_balances == {"A": 4, "B": 10, "C": 3}
    assert resp.allowed_count == 0 and resp.rejected_count == 1


def test_rejected_by_first_insufficient_in_candidate_chain_order():
    resp = _run(
        [
            {"id": "A", "capacity": 1, "refill_rate": 0},
            {"id": "B", "capacity": 1, "refill_rate": 0},
        ],
        [{"id": "r1", "timestamp": 0, "cost": 5, "candidates": [["B", "A"]]}],
    )
    # 链内按顺序第一个不足的是 B
    assert resp.results[0].rejections[0].rejected_by == "B"


def test_rejection_does_not_block_refill_and_later_requests():
    # t=0 拒绝（需要 10 只有 5）；t=5 补满后同一候选放行
    resp = _run(
        [{"id": "A", "capacity": 10, "refill_rate": 1, "initial": 5}],
        [
            {"id": "r1", "timestamp": 0, "cost": 10, "candidates": [["A"]]},
            {"id": "r2", "timestamp": 5, "cost": 10, "candidates": [["A"]]},
        ],
    )
    assert resp.results[0].allowed is False
    assert resp.results[0].balances["A"].after == 5
    assert resp.results[1].allowed is True
    assert resp.results[1].balances["A"].before == 10
    assert resp.results[1].balances["A"].after == 0


# ---------- 回放顺序与终态 ----------

def test_out_of_order_timestamps_simulated_chronologically():
    # 输入乱序，按时间戳处理；结果仍按原始下标返回
    resp = _run(
        [{"id": "A", "capacity": 10, "refill_rate": 1, "initial": 0}],
        [
            {"id": "r_late", "timestamp": 5, "cost": 5, "candidates": [["A"]]},
            {"id": "r_early", "timestamp": 0, "cost": 1, "candidates": [["A"]]},
        ],
    )
    assert resp.results[0].request_id == "r_late"
    assert resp.results[0].allowed is True
    assert resp.results[0].balances["A"].before == 5
    assert resp.results[1].request_id == "r_early"
    assert resp.results[1].allowed is False
    assert [r.index for r in resp.results] == [0, 1]


def test_same_timestamp_requests_keep_input_order():
    resp = _run(
        [{"id": "A", "capacity": 1, "refill_rate": 0}],
        [
            {"id": "r1", "timestamp": 0, "cost": 1, "candidates": [["A"]]},
            {"id": "r2", "timestamp": 0, "cost": 1, "candidates": [["A"]]},
        ],
    )
    assert resp.results[0].allowed is True
    assert resp.results[1].allowed is False


def test_final_balances_cover_untouched_buckets():
    resp = _run(
        [
            {"id": "A", "capacity": 10, "refill_rate": 1},
            {"id": "B", "capacity": 8, "refill_rate": 1, "initial": 2},
        ],
        [
            {"id": "r1", "timestamp": 0, "cost": 5, "candidates": [["A"]]},
            {"id": "r2", "timestamp": 3, "cost": 1, "candidates": [["A"]]},
        ],
    )
    # A：t=0 扣 5 -> 5；t=3 补 3 -> 8，再扣 1 -> 7
    # B 从未被任何请求触及：终态按 t_end 补到 min(8, 2 + 3*1) = 5
    assert resp.results[1].balances["A"].before == 8
    assert resp.final_balances == {"A": 7, "B": 5}


def test_shared_bucket_across_candidates_chains():
    # 两条候选链共享 tenant 桶：选中链扣减会影响后续请求的可行链
    resp = _run(
        [
            {"id": "A", "capacity": 10, "refill_rate": 0},
            {"id": "B", "capacity": 10, "refill_rate": 0},
            {"id": "T", "capacity": 6, "refill_rate": 0},
        ],
        [
            {"id": "r1", "timestamp": 0, "cost": 4, "candidates": [["A", "T"], ["B", "T"]]},
            {"id": "r2", "timestamp": 0, "cost": 4, "candidates": [["A", "T"], ["B", "T"]]},
        ],
    )
    # r1：两链得分相同（T 扣后占用率 4/6 最大），平局选候选 0；T 余 2
    assert resp.results[0].selected_index == 0
    # r2：T 余 2 不足 4，两条候选都不可行 -> 原子拒绝
    r2 = resp.results[1]
    assert r2.allowed is False
    assert [rej.rejected_by for rej in r2.rejections] == ["T", "T"]
    assert resp.final_balances == {"A": 6, "B": 10, "T": 2}


# ---------- 校验 ----------

def _validate_error(payload):
    with pytest.raises(RequestValidationErrors) as exc:
        RouteSimulateRequest.model_validate(payload)
    return exc.value.errors


def test_unknown_bucket_ref_in_any_candidate_rejected():
    errors = _validate_error(
        {
            "buckets": [{"id": "A", "capacity": 1, "refill_rate": 0}],
            "requests": [
                {"id": "r", "timestamp": 0, "cost": 1, "candidates": [["A"], ["GHOST"]]}
            ],
        }
    )
    assert any(
        e["type"] == "unknown_bucket_ref"
        and e["input"] == "GHOST"
        and e["loc"] == ("requests", 0, "candidates", 1, 0)
        for e in errors
    )


def test_duplicate_bucket_ids_rejected():
    errors = _validate_error(
        {
            "buckets": [
                {"id": "A", "capacity": 1, "refill_rate": 0},
                {"id": "A", "capacity": 2, "refill_rate": 0},
            ],
            "requests": [{"id": "r", "timestamp": 0, "cost": 1, "candidates": [["A"]]}],
        }
    )
    assert all(e["type"] == "duplicate_bucket" for e in errors)


def test_duplicate_request_ids_rejected():
    errors = _validate_error(
        {
            "buckets": [{"id": "A", "capacity": 1, "refill_rate": 0}],
            "requests": [
                {"id": "r", "timestamp": 0, "cost": 1, "candidates": [["A"]]},
                {"id": "r", "timestamp": 1, "cost": 1, "candidates": [["A"]]},
            ],
        }
    )
    assert any(e["type"] == "duplicate_request" for e in errors)


def test_empty_candidates_rejected():
    with pytest.raises(Exception):
        RouteSimulateRequest.model_validate(
            {
                "buckets": [{"id": "A", "capacity": 1, "refill_rate": 0}],
                "requests": [{"id": "r", "timestamp": 0, "cost": 1, "candidates": []}],
            }
        )


def test_empty_candidate_chain_rejected():
    with pytest.raises(Exception):
        RouteSimulateRequest.model_validate(
            {
                "buckets": [{"id": "A", "capacity": 1, "refill_rate": 0}],
                "requests": [{"id": "r", "timestamp": 0, "cost": 1, "candidates": [[]]}],
            }
        )


def test_duplicate_bucket_within_candidate_chain_rejected():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RouteSimulateRequest.model_validate(
            {
                "buckets": [
                    {"id": "A", "capacity": 1, "refill_rate": 0},
                    {"id": "B", "capacity": 1, "refill_rate": 0},
                ],
                "requests": [
                    {"id": "r", "timestamp": 0, "cost": 1, "candidates": [["A", "B", "A"]]}
                ],
            }
        )


def test_non_finite_timestamp_and_bad_cost_rejected():
    with pytest.raises(Exception):
        RouteSimulateRequest.model_validate(
            {
                "buckets": [{"id": "A", "capacity": 1, "refill_rate": 0}],
                "requests": [
                    {"id": "r", "timestamp": float("nan"), "cost": 1, "candidates": [["A"]]}
                ],
            }
        )
    with pytest.raises(Exception):
        RouteSimulateRequest.model_validate(
            {
                "buckets": [{"id": "A", "capacity": 1, "refill_rate": 0}],
                "requests": [{"id": "r", "timestamp": 0, "cost": 0, "candidates": [["A"]]}],
            }
        )


def test_extra_fields_rejected():
    with pytest.raises(Exception):
        RouteSimulateRequest.model_validate(
            {
                "buckets": [{"id": "A", "capacity": 1, "refill_rate": 0}],
                "requests": [
                    {
                        "id": "r",
                        "timestamp": 0,
                        "cost": 1,
                        "candidates": [["A"]],
                        "chain": ["A"],
                    }
                ],
            }
        )


# ---------- HTTP ----------

def test_http_route_simulate_ok():
    response = client.post(
        "/route-simulate",
        json={
            "buckets": [
                {"id": "A", "capacity": 10, "refill_rate": 0},
                {"id": "B", "capacity": 20, "refill_rate": 0, "initial": 10},
            ],
            "requests": [
                {"id": "r1", "timestamp": 0, "cost": 4, "candidates": [["A"], ["B"]]},
                {"id": "r2", "timestamp": 0, "cost": 20, "candidates": [["A"], ["B"]]},
            ],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["allowed_count"] == 1 and data["rejected_count"] == 1
    r1, r2 = data["results"]
    # r1：A 扣后占用率 0.4 < B 的 0.7 -> 选候选 0
    assert r1["selected_index"] == 0 and r1["selected_chain"] == ["A"]
    assert r1["rejections"] is None
    assert r1["balances"] == {"A": {"before": 10.0, "after": 6.0}}
    # r2：A 余 6、B 余 10 都不足 20 -> 原子拒绝，各候选给出首个不足桶
    assert r2["allowed"] is False
    assert r2["selected_chain"] is None
    assert [(rej["candidate_index"], rej["rejected_by"]) for rej in r2["rejections"]] == [
        (0, "A"),
        (1, "B"),
    ]
    assert r2["balances"]["A"]["after"] == 6.0
    assert r2["balances"]["B"]["after"] == 10.0
    assert data["final_balances"] == {"A": 6.0, "B": 10.0}


def test_http_validation_error_is_422():
    response = client.post(
        "/route-simulate",
        json={
            "buckets": [{"id": "A", "capacity": 1, "refill_rate": 0}],
            "requests": [
                {"id": "r", "timestamp": 0, "cost": 1, "candidates": [["A", "GHOST"]]}
            ],
        },
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert any(e["type"] == "unknown_bucket_ref" for e in detail)


def test_http_value_error_is_422_not_500():
    response = client.post(
        "/route-simulate",
        json={
            "buckets": [{"id": "A", "capacity": 1, "refill_rate": 0}],
            "requests": [{"id": "r", "timestamp": 0, "cost": 1, "candidates": [[]]}],
        },
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert isinstance(detail, list) and detail
