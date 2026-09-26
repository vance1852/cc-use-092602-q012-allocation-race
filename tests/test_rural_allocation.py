from __future__ import annotations

import json
import sqlite3
import tempfile
import threading
import unittest
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from rural_allocation.api import JsonApplication
from rural_allocation.clock import FrozenClock
from rural_allocation.errors import Conflict, DomainConflict, Forbidden
from rural_allocation.planning import AllocationRequest, PricePoint, allocate_capacity, digest, latest_streak
from rural_allocation.service import SupplyService
from rural_allocation.risk import DemandBucket, inventory_coverage, mark_to_market, supply_gap
from rural_allocation.storage import connect


class PlanningTests(unittest.TestCase):
    def test_latest_down_streak_uses_first_close_as_base(self) -> None:
        streak = latest_streak([
            PricePoint("2026-09-18", Decimal("108")),
            PricePoint("2026-09-19", Decimal("105")),
            PricePoint("2026-09-20", Decimal("102")),
            PricePoint("2026-09-21", Decimal("98")),
        ])
        self.assertEqual(streak.direction, "down")
        self.assertEqual(streak.sessions, 4)
        self.assertEqual(streak.start_date, "2026-09-18")
        self.assertEqual(streak.end_close, Decimal("98"))

    def test_allocation_is_stable_and_does_not_exceed_capacity(self) -> None:
        rows = allocate_capacity(Decimal("100"), [
            AllocationRequest("later", Decimal("80"), 20, "2026-09-24T09:00:00Z"),
            AllocationRequest("first", Decimal("70"), 10, "2026-09-24T10:00:00Z"),
        ])
        self.assertEqual(rows[0]["nomination_id"], "first")
        self.assertEqual(rows[0]["allocated_mu"], "70.000")
        self.assertEqual(rows[1]["allocated_mu"], "30.000")

    def test_inventory_coverage_and_supply_gap(self) -> None:
        coverage = inventory_coverage(
            [{"facility_id": "settlement", "product": "homestead", "available_mu": "250"}],
            [DemandBucket("settlement", "homestead", Decimal("100"), Decimal("20"))],
        )
        self.assertEqual(coverage[0]["coverage_days"], "2.30")
        self.assertTrue(coverage[0]["below_three_days"])
        gap = supply_gap(
            opening_inventory=Decimal("100"),
            confirmed_inbound=Decimal("30"),
            forecast_demand=Decimal("120"),
            protected_reserve=Decimal("40"),
        )
        self.assertEqual(gap["supply_gap"], "30.000")

    def test_mark_to_market_groups_deterministically(self) -> None:
        result = mark_to_market(
            [{"position_id": "p1", "market_index": "PEAK_VALLEY", "quantity_mu": "100", "entry_price_cny": "105"}],
            {"PEAK_VALLEY": Decimal("98")},
        )
        self.assertEqual(result["unrealized_pnl_cny"], "-700.00")


class SupplyServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.connection = sqlite3.connect(":memory:", isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self.clock = FrozenClock(datetime(2026, 9, 24, 8, 0, tzinfo=timezone.utc))
        self.service = SupplyService(self.connection, self.clock)
        for user_id, role in (("plan", "planner"), ("dispatch", "dispatcher"), ("risk", "risk"), ("audit", "auditor")):
            self.service.create_user(user_id, user_id, role)
        self.service.create_facility("plan", {"facility_id": "village-a", "name": "北部示范村", "kind": "storage", "timezone": "Asia/Shanghai", "capacity_mu": "500000"})
        self.service.create_facility("plan", {"facility_id": "settlement-b", "name": "东部安置片区", "kind": "settlement", "timezone": "Asia/Shanghai", "capacity_mu": "800000"})
        self.service.create_route("plan", {"route_id": "pool-a-b", "origin_id": "village-a", "destination_id": "settlement-b", "product": "cultivated-land", "daily_capacity": "100000", "loss_basis_points": 25, "transit_hours": 36})

    def tearDown(self) -> None:
        self.connection.close()

    def quote(self, day: int, close: str) -> dict[str, object]:
        return self.service.record_quote("plan", {"market_index": "PEAK_VALLEY", "trade_date": f"2026-09-{day}", "close_cny": close, "source_revision": f"r-{day}", "observed_at": f"2026-09-{day}T21:00:00Z"})

    def test_quote_revisions_preserve_history(self) -> None:
        first = self.quote(23, "98")
        second = self.service.record_quote("plan", {"market_index": "PEAK_VALLEY", "trade_date": "2026-09-23", "close_cny": "97.8", "source_revision": "r-23-corrected", "observed_at": "2026-09-23T22:00:00Z"})
        self.assertNotEqual(first["quote_id"], second["quote_id"])
        rows = self.connection.execute("SELECT * FROM market_index_quotes ORDER BY quote_id").fetchall()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["supersedes_quote_id"], rows[0]["quote_id"])

    def test_nomination_replay_and_payload_conflict(self) -> None:
        payload = {"nomination_id": "nom-1", "route_id": "pool-a-b", "shipper_id": "household", "service_date": "2026-09-25", "requested_mu": "80000", "priority": 10, "idempotency_key": "key-1"}
        first = self.service.submit_nomination("dispatch", payload)
        self.assertEqual(first, self.service.submit_nomination("dispatch", payload))
        changed = dict(payload, requested_mu="81000")
        with self.assertRaises(Conflict):
            self.service.submit_nomination("dispatch", changed)

    def test_outage_reduces_allocation_and_transfer_consumes_inventory(self) -> None:
        self.service.announce_outage("risk", "pool-a-b", "2026-09-25T00:00:00Z", "2026-09-25T23:59:59Z", "50", "检修")
        for number, requested, priority in ((1, "40000", 10), (2, "30000", 20)):
            self.service.submit_nomination("dispatch", {"nomination_id": f"nom-{number}", "route_id": "pool-a-b", "shipper_id": f"shipper-{number}", "service_date": "2026-09-25", "requested_mu": requested, "priority": priority, "idempotency_key": f"key-{number}"})
        allocation = self.service.allocate("dispatch", "pool-a-b", "2026-09-25")
        self.assertEqual(allocation["available_capacity"], "50000.000")
        self.assertEqual(allocation["allocations"][1]["allocated_mu"], "10000.000")
        self.service.add_inventory_lot("dispatch", {"lot_id": "lot-1", "facility_id": "village-a", "product": "cultivated-land", "grade": "PEAK_VALLEY", "quantity_mu": "60000", "unit_cost_cny": "91", "received_at": "2026-09-24T06:00:00Z"})
        transfer = self.service.dispatch_transfer("dispatch", "transfer-1", "nom-1", "lot-1", 2)
        self.assertEqual(transfer["surveyed_mu"], "40000.000")
        self.assertEqual(self.service.inventory_lot("lot-1")["available_mu"], "20000.000")

    def test_scenario_is_approved_and_replayed_by_input(self) -> None:
        self.quote(23, "98")
        self.service.add_inventory_lot("dispatch", {"lot_id": "lot-1", "facility_id": "village-a", "product": "cultivated-land", "grade": "PEAK_VALLEY", "quantity_mu": "60000", "unit_cost_cny": "91", "received_at": "2026-09-24T06:00:00Z"})
        self.service.create_scenario("plan", {"scenario_id": "restart", "name": "机组检修恢复", "market_index_drop_percent": "9", "route_capacity_changes": {"pool-a-b": "20"}, "demand_changes": {"village-a:cultivated-land": "-5"}})
        with self.assertRaises(Forbidden):
            self.service.approve_scenario("plan", "restart", 1)
        self.service.approve_scenario("risk", "restart", 1)
        first = self.service.run_scenario("plan", "restart", "2026-09-23")
        second = self.service.run_scenario("plan", "restart", "2026-09-23")
        self.assertFalse(first["replayed"])
        self.assertTrue(second["replayed"])
        self.assertEqual(first["run_id"], second["run_id"])

    def test_audit_chain_detects_tampering(self) -> None:
        self.assertTrue(self.service.audit_chain("audit")["valid"])
        self.connection.execute("UPDATE supply_audit_events SET payload_json='{}' WHERE event_id=1")
        self.assertFalse(self.service.audit_chain("audit")["valid"])

    def test_api_exposes_browser_free_boundary(self) -> None:
        app = JsonApplication(self.service)
        self.assertEqual(app.handle("GET", "/health").status, 200)
        response = app.handle("GET", "/quotes/summary/PEAK_VALLEY", {"X-Actor-Id": "plan"})
        self.assertEqual(response.status, 404)
        self.assertEqual(response.body["error"]["code"], "not_found")


class ConcurrentAllocationTests(unittest.TestCase):
    """用文件库、真实线程与可控屏障覆盖并发分配的重放与版本冲突。"""

    DATE = "2026-09-25"

    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tempdir.name) / "allocation.sqlite3")
        connection = connect(self.db_path)
        clock = FrozenClock(datetime(2026, 9, 24, 8, 0, tzinfo=timezone.utc))
        service = SupplyService(connection, clock)
        for user_id, role in (("plan", "planner"), ("dispatch", "dispatcher"), ("risk", "risk"), ("audit", "auditor")):
            service.create_user(user_id, user_id, role)
        service.create_facility("plan", {"facility_id": "village-a", "name": "北部示范村", "kind": "storage", "timezone": "Asia/Shanghai", "capacity_mu": "500000"})
        service.create_facility("plan", {"facility_id": "settlement-b", "name": "东部安置片区", "kind": "settlement", "timezone": "Asia/Shanghai", "capacity_mu": "800000"})
        service.create_route("plan", {"route_id": "pool-a-b", "origin_id": "village-a", "destination_id": "settlement-b", "product": "cultivated-land", "daily_capacity": "100000", "loss_basis_points": 25, "transit_hours": 36})
        for number, requested, priority in ((1, "40000", 10), (2, "30000", 20)):
            service.submit_nomination("dispatch", {"nomination_id": f"nom-{number}", "route_id": "pool-a-b", "shipper_id": f"shipper-{number}", "service_date": self.DATE, "requested_mu": requested, "priority": priority, "idempotency_key": f"key-{number}"})
        connection.close()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _service(self, **gates) -> SupplyService:
        return SupplyService(
            connect(self.db_path),
            FrozenClock(datetime(2026, 9, 24, 8, 0, tzinfo=timezone.utc)),
            **gates,
        )

    def _run_concurrent(self) -> list[dict[str, object]]:
        barrier = threading.Barrier(2, timeout=10)
        parked = threading.Event()
        release = threading.Event()

        class CommitGate:
            def __call__(self, route_id: str, service_date: str) -> None:
                # 先到写锁的运行停在写入后、状态更新前，确保后到请求真实争锁。
                parked.set()
                release.wait(timeout=10)

        outcomes: list[dict[str, object]] = []

        def worker() -> None:
            service = self._service(
                allocation_start_gate=lambda route_id, service_date: barrier.wait(),
                allocation_commit_gate=CommitGate(),
            )
            try:
                outcomes.append(service.allocate("dispatch", "pool-a-b", self.DATE))
            except Exception as exc:  # noqa: BLE001 - 并发下任何异常都要暴露给断言
                outcomes.append({"error": exc})
            finally:
                service.connection.close()

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for thread in threads:
            thread.start()
        self.assertTrue(parked.wait(timeout=5), "赢家未在提交点停住")
        release.set()
        for thread in threads:
            thread.join(timeout=10)
            self.assertFalse(thread.is_alive(), "并发分配线程超时")
        return outcomes

    def test_concurrent_identical_requests_replay_same_business_result(self) -> None:
        outcomes = self._run_concurrent()
        self.assertEqual(len(outcomes), 2)
        for outcome in outcomes:
            self.assertNotIn("error", outcome)
            self.assertNotIsInstance(outcome.get("error"), sqlite3.Error)
        flags = sorted(bool(outcome["replayed"]) for outcome in outcomes)
        self.assertEqual(flags, [False, True])
        first, second = outcomes
        self.assertEqual(first["allocation_id"], second["allocation_id"])
        self.assertEqual(first["allocations"], second["allocations"])
        self.assertEqual(first["version"]["snapshot_sha256"], second["version"]["snapshot_sha256"])
        self.assertEqual(first["version"]["request_sha256"], second["version"]["request_sha256"])

        audit_connection = connect(self.db_path)
        try:
            run_rows = audit_connection.execute("SELECT * FROM allocation_runs").fetchall()
            self.assertEqual(len(run_rows), 1)
            nominations = audit_connection.execute(
                "SELECT state,revision,allocation_id FROM nominations ORDER BY nomination_id"
            ).fetchall()
            self.assertEqual({row["state"] for row in nominations}, {"allocated"})
            self.assertEqual({row["revision"] for row in nominations}, {2})
            self.assertEqual({row["allocation_id"] for row in nominations}, {run_rows[0]["allocation_id"]})
        finally:
            audit_connection.close()

    def test_retry_after_commit_is_idempotent_replay(self) -> None:
        service = self._service()
        first = service.allocate("dispatch", "pool-a-b", self.DATE)
        second = service.allocate("dispatch", "pool-a-b", self.DATE)
        self.assertFalse(first["replayed"])
        self.assertTrue(second["replayed"])
        self.assertEqual(first["allocation_id"], second["allocation_id"])
        self.assertEqual(first["allocations"], second["allocations"])
        service.connection.close()

    def test_changed_application_set_is_domain_conflict_without_half_state(self) -> None:
        service = self._service()
        service.allocate("dispatch", "pool-a-b", self.DATE)
        service.submit_nomination("dispatch", {"nomination_id": "nom-3", "route_id": "pool-a-b", "shipper_id": "shipper-3", "service_date": self.DATE, "requested_mu": "5000", "priority": 30, "idempotency_key": "key-3"})
        with self.assertRaises(DomainConflict):
            service.allocate("dispatch", "pool-a-b", self.DATE)
        run_count = service.connection.execute("SELECT COUNT(*) AS n FROM allocation_runs").fetchone()["n"]
        self.assertEqual(run_count, 1)
        states = dict(
            (row["nomination_id"], row["state"])
            for row in service.connection.execute("SELECT nomination_id,state FROM nominations")
        )
        self.assertEqual(states["nom-1"], "allocated")
        self.assertEqual(states["nom-2"], "allocated")
        self.assertEqual(states["nom-3"], "submitted")
        service.connection.close()

    def test_bumped_limit_version_is_domain_conflict(self) -> None:
        service = self._service()
        committed = service.allocate("dispatch", "pool-a-b", self.DATE)
        self.assertEqual(committed["version"]["limit_version"], 0)
        service.announce_outage("risk", "pool-a-b", f"{self.DATE}T00:00:00Z", f"{self.DATE}T23:59:59Z", "80", "新增限制")
        with self.assertRaises(DomainConflict) as context:
            service.allocate("dispatch", "pool-a-b", self.DATE)
        self.assertNotIsInstance(context.exception.__cause__, sqlite3.Error)
        row = service.connection.execute("SELECT * FROM allocation_runs").fetchone()
        self.assertEqual(row["limit_version"], 0)
        current_version = service.connection.execute("SELECT schedule_version FROM routes WHERE route_id='pool-a-b'").fetchone()["schedule_version"]
        self.assertEqual(current_version, 1)
        service.connection.close()

    def test_failure_inside_transaction_leaves_no_half_application_state(self) -> None:
        class Abort(RuntimeError):
            pass

        def fail_after_run_insert(route_id: str, service_date: str) -> None:
            raise Abort("模拟提交前失败")

        service = self._service(allocation_commit_gate=fail_after_run_insert)
        with self.assertRaises(Abort):
            service.allocate("dispatch", "pool-a-b", self.DATE)
        self.assertEqual(service.connection.execute("SELECT COUNT(*) AS n FROM allocation_runs").fetchone()["n"], 0)
        self.assertEqual(
            {row["state"] for row in service.connection.execute("SELECT state FROM nominations")},
            {"submitted"},
        )
        # 回滚干净后同一请求仍可成功，且不需要清理任何半成品。
        service.allocation_commit_gate = None
        retry = service.allocate("dispatch", "pool-a-b", self.DATE)
        self.assertFalse(retry["replayed"])
        service.connection.close()

    def test_audit_query_restores_snapshot_and_commit_version(self) -> None:
        service = self._service()
        service.announce_outage("risk", "pool-a-b", f"{self.DATE}T00:00:00Z", f"{self.DATE}T23:59:59Z", "50", "检修")
        allocated = service.allocate("dispatch", "pool-a-b", self.DATE)
        audit = service.allocation_audit("audit", "pool-a-b", self.DATE)
        self.assertEqual(audit["allocation_id"], allocated["allocation_id"])
        self.assertEqual(audit["version"]["limit_version"], allocated["version"]["limit_version"])
        self.assertEqual(audit["version"]["request_sha256"], allocated["version"]["request_sha256"])
        self.assertEqual(audit["version"]["snapshot_sha256"], allocated["version"]["snapshot_sha256"])
        self.assertEqual(audit["committed_at"], allocated["version"]["committed_at"])
        self.assertEqual(len(audit["snapshot"]["applications"]), 2)
        self.assertEqual(len(audit["snapshot"]["outages"]), 1)
        self.assertEqual(audit["snapshot"]["available_capacity"], "50000.000")
        self.assertEqual(digest(audit["snapshot"]), audit["version"]["snapshot_sha256"])
        self.assertEqual(
            audit["result"],
            {
                "route_id": allocated["route_id"],
                "service_date": allocated["service_date"],
                "available_capacity": allocated["available_capacity"],
                "allocations": allocated["allocations"],
            },
        )
        service.connection.close()

    def test_api_hides_storage_exception_and_maps_domain_conflict(self) -> None:
        outcomes: list[object] = []
        barrier = threading.Barrier(2, timeout=10)
        parked = threading.Event()
        release = threading.Event()

        class CommitGate:
            def __call__(self, route_id: str, service_date: str) -> None:
                parked.set()
                release.wait(timeout=10)

        def request() -> None:
            connection = connect(self.db_path)
            service = SupplyService(
                connection,
                FrozenClock(datetime(2026, 9, 24, 8, 0, tzinfo=timezone.utc)),
                allocation_start_gate=lambda route_id, service_date: barrier.wait(),
                allocation_commit_gate=CommitGate(),
            )
            app = JsonApplication(service)
            try:
                response = app.handle(
                    "POST",
                    "/routes/pool-a-b/allocate",
                    {"X-Actor-Id": "dispatch"},
                    json.dumps({"service_date": self.DATE}).encode("utf-8"),
                )
                outcomes.append((response.status, response.body))
            finally:
                connection.close()

        threads = [threading.Thread(target=request) for _ in range(2)]
        for thread in threads:
            thread.start()
        self.assertTrue(parked.wait(timeout=5))
        release.set()
        for thread in threads:
            thread.join(timeout=10)

        self.assertEqual(len(outcomes), 2)
        self.assertEqual({status for status, _ in outcomes}, {200})
        allocation_ids = {body["allocation_id"] for _, body in outcomes}
        self.assertEqual(len(allocation_ids), 1)
        self.assertFalse(any("sqlite" in str(body).lower() for _, body in outcomes))

        connection = connect(self.db_path)
        try:
            service = SupplyService(connection)
            app = JsonApplication(service)
            service.announce_outage("risk", "pool-a-b", f"{self.DATE}T00:00:00Z", f"{self.DATE}T23:59:59Z", "70", "再检修")
            conflict = app.handle(
                "POST",
                "/routes/pool-a-b/allocate",
                {"X-Actor-Id": "dispatch"},
                json.dumps({"service_date": self.DATE}).encode("utf-8"),
            )
            self.assertEqual(conflict.status, 409)
            self.assertEqual(conflict.body["error"]["code"], "domain_conflict")
            self.assertNotIn("sqlite", str(conflict.body).lower())
            missing = app.handle(
                "GET",
                f"/routes/pool-x/allocations?service_date={self.DATE}",
                {"X-Actor-Id": "audit"},
            )
            self.assertEqual(missing.status, 404)
        finally:
            connection.close()


if __name__ == "__main__":
    unittest.main()
