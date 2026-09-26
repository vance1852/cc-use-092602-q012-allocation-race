from __future__ import annotations

import json
import sqlite3
import tempfile
import threading
import time
import unittest
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from rural_allocation.api import JsonApplication
from rural_allocation.clock import FrozenClock
from rural_allocation.errors import Conflict, Forbidden, InvalidState, NotFound
from rural_allocation.planning import AllocationRequest, PricePoint, allocate_capacity, latest_streak
from rural_allocation.service import SupplyService
from rural_allocation.storage import connect, initialize
from rural_allocation.risk import DemandBucket, inventory_coverage, mark_to_market, supply_gap


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
    """集中分配日的可控竞争测试：每个线程使用独立连接，共享同一 SQLite 文件。"""

    DATE = "2026-09-25"

    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.db_path = Path(self.tempdir.name) / "allocation.sqlite3"
        self.clock = FrozenClock(datetime(2026, 9, 24, 8, 0, tzinfo=timezone.utc))
        self.connections: list[sqlite3.Connection] = []
        self.addCleanup(self._close_connections)
        service = self._service()
        for user_id, role in (("plan", "planner"), ("dispatch", "dispatcher"), ("risk", "risk"), ("audit", "auditor")):
            service.create_user(user_id, user_id, role)
        service.create_facility("plan", {"facility_id": "village-a", "name": "北部示范村", "kind": "storage", "timezone": "Asia/Shanghai", "capacity_mu": "500000"})
        service.create_facility("plan", {"facility_id": "settlement-b", "name": "东部安置片区", "kind": "settlement", "timezone": "Asia/Shanghai", "capacity_mu": "800000"})
        service.create_route("plan", {"route_id": "pool-a-b", "origin_id": "village-a", "destination_id": "settlement-b", "product": "cultivated-land", "daily_capacity": "100000", "loss_basis_points": 25, "transit_hours": 36})
        for number, requested, priority in ((1, "40000", 10), (2, "30000", 20)):
            service.submit_nomination("dispatch", self._nomination_payload(number, requested, priority))

    def _close_connections(self) -> None:
        for connection in self.connections:
            connection.close()

    def _service(self, probe=None, *, track: bool = True) -> SupplyService:
        connection = connect(self.db_path)
        if track:
            self.connections.append(connection)
        # track=False 时由调用方在创建连接的线程内通过 service.connection.close() 关闭
        return SupplyService(connection, self.clock, commit_probe=probe)

    @staticmethod
    def _nomination_payload(number: int, requested: str, priority: int) -> dict[str, object]:
        return {
            "nomination_id": f"nom-{number}",
            "route_id": "pool-a-b",
            "shipper_id": f"shipper-{number}",
            "service_date": ConcurrentAllocationTests.DATE,
            "requested_mu": requested,
            "priority": priority,
            "idempotency_key": f"key-{number}",
        }

    def _nomination_rows(self) -> dict[str, sqlite3.Row]:
        connection = connect(self.db_path)
        self.connections.append(connection)
        rows = connection.execute(
            "SELECT nomination_id,state,allocated_mu,revision,allocation_id FROM nominations ORDER BY nomination_id"
        ).fetchall()
        return {row["nomination_id"]: row for row in rows}

    def _run_rows(self) -> list[sqlite3.Row]:
        connection = connect(self.db_path)
        self.connections.append(connection)
        return connection.execute("SELECT * FROM allocation_runs ORDER BY allocation_id").fetchall()

    def test_concurrent_same_input_replays_single_committed_run(self) -> None:
        entered = threading.Event()
        release = threading.Event()

        def probe(label: str) -> None:
            if label == "allocate":
                entered.set()
                release.wait(5)

        outcomes: dict[str, object] = {}

        def run(name: str, use_probe: bool) -> None:
            # 连接必须在使用它的线程内创建并关闭
            service = self._service(probe if use_probe else None, track=False)
            try:
                outcomes[name] = service.allocate("dispatch", "pool-a-b", self.DATE)
            except Exception as exc:  # 并发结果统一收集后断言
                outcomes[name] = exc
            finally:
                service.connection.close()

        first = threading.Thread(target=run, args=("first", True))
        first.start()
        self.assertTrue(entered.wait(5), "第一个分配请求未进入事务")
        second = threading.Thread(target=run, args=("second", False))
        second.start()
        time.sleep(0.2)  # 让第二个请求抵达 BEGIN IMMEDIATE 并在写锁上排队
        release.set()
        first.join(5)
        second.join(5)
        self.assertFalse(first.is_alive() or second.is_alive(), "并发线程未在超时内结束")
        first_result = outcomes.get("first")
        second_result = outcomes.get("second")
        self.assertIsInstance(first_result, dict, repr(first_result))
        self.assertIsInstance(second_result, dict, repr(second_result))
        self.assertFalse(first_result["replayed"])
        self.assertTrue(second_result["replayed"])
        self.assertEqual(first_result["allocation_id"], second_result["allocation_id"])
        self.assertEqual(first_result["commit_version"], second_result["commit_version"])
        self.assertEqual(first_result["allocations"], second_result["allocations"])
        runs = self._run_rows()
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["commit_version"], first_result["commit_version"])
        for row in self._nomination_rows().values():
            self.assertEqual(row["revision"], 2)
            self.assertEqual(row["allocation_id"], first_result["allocation_id"])
        service = self._service()
        self.assertTrue(service.audit_chain("audit")["valid"])

    def test_concurrent_allocate_conflicts_when_application_set_grows(self) -> None:
        committed = self._service().allocate("dispatch", "pool-a-b", self.DATE)
        self.assertFalse(committed["replayed"])
        entered = threading.Event()
        release = threading.Event()

        def probe(label: str) -> None:
            if label == "nomination":
                entered.set()
                release.wait(5)

        outcomes: dict[str, object] = {}

        def submit() -> None:
            service = self._service(probe, track=False)
            try:
                outcomes["submit"] = service.submit_nomination("dispatch", self._nomination_payload(3, "10000", 30))
            except Exception as exc:  # 并发结果统一收集后断言
                outcomes["submit"] = exc
            finally:
                service.connection.close()

        def allocate() -> None:
            service = self._service(track=False)
            try:
                outcomes["allocate"] = service.allocate("dispatch", "pool-a-b", self.DATE)
            except Conflict as exc:
                outcomes["allocate"] = exc
            finally:
                service.connection.close()

        submitter = threading.Thread(target=submit)
        submitter.start()
        self.assertTrue(entered.wait(5), "补充申请未进入事务")
        allocator = threading.Thread(target=allocate)
        allocator.start()
        time.sleep(0.2)  # 分配请求在写锁后排队，等补充申请先提交
        release.set()
        submitter.join(5)
        allocator.join(5)
        self.assertFalse(submitter.is_alive() or allocator.is_alive(), "并发线程未在超时内结束")
        self.assertIsInstance(outcomes.get("submit"), dict, repr(outcomes.get("submit")))
        outcome = outcomes.get("allocate")
        self.assertIsInstance(outcome, Conflict, repr(outcome))
        self.assertIn("未纳入", str(outcome))
        # 领域冲突不得留下半套申请状态
        self.assertEqual(len(self._run_rows()), 1)
        rows = self._nomination_rows()
        self.assertEqual(rows["nom-3"]["state"], "submitted")
        self.assertEqual(rows["nom-3"]["allocated_mu"], "0")
        self.assertEqual(rows["nom-3"]["revision"], 1)
        self.assertIsNone(rows["nom-3"]["allocation_id"])
        self.assertEqual(rows["nom-1"]["revision"], 2)
        self.assertEqual(rows["nom-1"]["allocation_id"], committed["allocation_id"])

    def test_retry_returns_committed_result(self) -> None:
        service = self._service()
        committed = service.allocate("dispatch", "pool-a-b", self.DATE)
        replay = service.allocate("dispatch", "pool-a-b", self.DATE)
        self.assertFalse(committed["replayed"])
        self.assertTrue(replay["replayed"])
        self.assertEqual(committed["allocation_id"], replay["allocation_id"])
        self.assertEqual(committed["commit_version"], replay["commit_version"])
        self.assertEqual(committed["allocations"], replay["allocations"])
        self.assertEqual(len(self._run_rows()), 1)
        for row in self._nomination_rows().values():
            self.assertEqual(row["revision"], 2)

    def test_new_application_after_commit_conflicts(self) -> None:
        service = self._service()
        service.allocate("dispatch", "pool-a-b", self.DATE)
        service.submit_nomination("dispatch", self._nomination_payload(3, "10000", 30))
        with self.assertRaises(Conflict):
            service.allocate("dispatch", "pool-a-b", self.DATE)
        rows = self._nomination_rows()
        self.assertEqual(rows["nom-3"]["state"], "submitted")
        self.assertEqual(rows["nom-3"]["revision"], 1)

    def test_capacity_limit_change_after_commit_conflicts(self) -> None:
        service = self._service()
        service.announce_outage("risk", "pool-a-b", "2026-09-25T00:00:00Z", "2026-09-25T12:00:00Z", "50", "检修")
        committed = service.allocate("dispatch", "pool-a-b", self.DATE)
        self.assertEqual(committed["available_capacity"], "50000.000")
        replay = service.allocate("dispatch", "pool-a-b", self.DATE)
        self.assertTrue(replay["replayed"])
        service.announce_outage("risk", "pool-a-b", "2026-09-25T12:00:00Z", "2026-09-25T23:59:59Z", "80", "追加限运")
        with self.assertRaises(Conflict) as raised:
            service.allocate("dispatch", "pool-a-b", self.DATE)
        self.assertIn("限制版本", str(raised.exception))

    def test_allocate_without_pending_nominations_still_invalid(self) -> None:
        service = self._service()
        with self.assertRaises(InvalidState):
            service.allocate("dispatch", "pool-a-b", "2026-09-26")

    def test_audit_restores_snapshot_and_commit_version(self) -> None:
        service = self._service()
        service.announce_outage("risk", "pool-a-b", "2026-09-25T00:00:00Z", "2026-09-25T23:59:59Z", "50", "检修")
        committed = service.allocate("dispatch", "pool-a-b", self.DATE)
        audit = service.allocation_audit("audit", "pool-a-b", self.DATE)
        self.assertEqual(audit["allocation_id"], committed["allocation_id"])
        self.assertEqual(audit["commit_version"], committed["commit_version"])
        self.assertEqual(audit["snapshot"]["available_capacity"], "50000.000")
        self.assertEqual(len(audit["snapshot"]["outages"]), 1)
        self.assertEqual(audit["snapshot"]["route"]["route_id"], "pool-a-b")
        self.assertEqual([n["nomination_id"] for n in audit["snapshot"]["nominations"]], ["nom-1", "nom-2"])
        self.assertEqual(audit["result"]["allocations"], committed["allocations"])
        self.assertEqual([n["nomination_id"] for n in audit["nominations"]], ["nom-1", "nom-2"])
        self.assertTrue(all(n["state"] == "allocated" for n in audit["nominations"]))
        with self.assertRaises(Forbidden):
            service.allocation_audit("dispatch", "pool-a-b", self.DATE)
        with self.assertRaises(NotFound):
            service.allocation_audit("audit", "pool-a-b", "2026-09-26")

    def test_api_allocate_replay_conflict_and_audit(self) -> None:
        service = self._service()
        app = JsonApplication(service)
        headers = {"X-Actor-Id": "dispatch"}
        body = json.dumps({"service_date": self.DATE}).encode("utf-8")
        committed = app.handle("POST", "/routes/pool-a-b/allocate", headers, body)
        self.assertEqual(committed.status, 200)
        self.assertFalse(committed.body["replayed"])
        replayed = app.handle("POST", "/routes/pool-a-b/allocate", headers, body)
        self.assertEqual(replayed.status, 200)
        self.assertTrue(replayed.body["replayed"])
        self.assertEqual(committed.body["allocation_id"], replayed.body["allocation_id"])
        service.submit_nomination("dispatch", self._nomination_payload(3, "10000", 30))
        conflict = app.handle("POST", "/routes/pool-a-b/allocate", headers, body)
        self.assertEqual(conflict.status, 409)
        self.assertEqual(conflict.body["error"]["code"], "conflict")
        self.assertNotIn("IntegrityError", json.dumps(conflict.body, ensure_ascii=False))
        audit = app.handle("GET", f"/routes/pool-a-b/allocations/{self.DATE}", {"X-Actor-Id": "audit"})
        self.assertEqual(audit.status, 200)
        self.assertEqual(audit.body["commit_version"], committed.body["commit_version"])
        self.assertEqual(len(audit.body["snapshot"]["nominations"]), 2)
        forbidden = app.handle("GET", f"/routes/pool-a-b/allocations/{self.DATE}", headers)
        self.assertEqual(forbidden.status, 403)

    def test_api_hides_unexpected_errors(self) -> None:
        connection = connect(self.db_path)
        self.connections.append(connection)
        service = SupplyService(connection, self.clock)
        app = JsonApplication(service)
        connection.execute("PRAGMA foreign_keys=OFF")
        connection.execute("DROP TABLE supply_users")
        with self.assertLogs("rural_allocation.api", level="ERROR"):
            response = app.handle(
                "POST",
                "/routes/pool-a-b/allocate",
                {"X-Actor-Id": "dispatch"},
                json.dumps({"service_date": self.DATE}).encode("utf-8"),
            )
        self.assertEqual(response.status, 500)
        self.assertEqual(response.body["error"]["code"], "internal_error")

    def test_api_factory_serves_concurrent_threads(self) -> None:
        local = threading.local()

        def thread_service() -> SupplyService:
            service = getattr(local, "service", None)
            if service is None:
                service = self._service(track=False)
                local.service = service
            return service

        app = JsonApplication(thread_service)
        body = json.dumps({"service_date": self.DATE}).encode("utf-8")
        outcomes: dict[str, object] = {}

        def post(name: str) -> None:
            try:
                response = app.handle("POST", "/routes/pool-a-b/allocate", {"X-Actor-Id": "dispatch"}, body)
                outcomes[name] = (response.status, response.body)
            finally:
                service = getattr(local, "service", None)
                if service is not None:
                    service.connection.close()

        threads = [threading.Thread(target=post, args=(f"worker-{index}",)) for index in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)
        self.assertEqual(len(outcomes), 2)
        bodies = [outcome[1] for outcome in outcomes.values()]
        self.assertTrue(all(outcome[0] == 200 for outcome in outcomes.values()), repr(outcomes))
        self.assertEqual({body["allocation_id"] for body in bodies}, {bodies[0]["allocation_id"]})
        self.assertEqual(sorted(body["replayed"] for body in bodies), [False, True])
        self.assertEqual(len(self._run_rows()), 1)

    def test_initialize_upgrades_legacy_allocation_schema(self) -> None:
        legacy_path = Path(self.tempdir.name) / "legacy.sqlite3"
        connection = sqlite3.connect(legacy_path, isolation_level=None)
        connection.row_factory = sqlite3.Row
        self.connections.append(connection)
        connection.executescript("""
            CREATE TABLE supply_users (
                user_id TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                role TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL
            );
            CREATE TABLE nominations (
                nomination_id TEXT PRIMARY KEY,
                route_id TEXT NOT NULL,
                service_date TEXT NOT NULL,
                requested_mu TEXT NOT NULL,
                allocated_mu TEXT NOT NULL DEFAULT '0',
                priority INTEGER NOT NULL,
                state TEXT NOT NULL DEFAULT 'submitted',
                revision INTEGER NOT NULL DEFAULT 1,
                idempotency_key TEXT NOT NULL UNIQUE,
                submitted_by TEXT NOT NULL,
                submitted_at TEXT NOT NULL
            );
            CREATE TABLE allocation_runs (
                allocation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                route_id TEXT NOT NULL,
                service_date TEXT NOT NULL,
                input_sha256 TEXT NOT NULL,
                available_capacity TEXT NOT NULL,
                result_json TEXT NOT NULL,
                created_by TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(route_id, service_date, input_sha256)
            );
        """)
        connection.execute("INSERT INTO supply_users VALUES('legacy','旧经办','dispatcher',1,'2026-09-20T00:00:00Z')")
        connection.execute(
            "INSERT INTO allocation_runs(route_id,service_date,input_sha256,available_capacity,result_json,"
            "created_by,created_at) VALUES('pool-a-b','2026-09-25','abc','100000','{}','legacy','2026-09-20T00:00:00Z')"
        )
        initialize(connection)
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(allocation_runs)")}
        self.assertTrue({"limits_sha256", "commit_version", "snapshot_json"} <= columns)
        row = connection.execute("SELECT * FROM allocation_runs").fetchone()
        self.assertEqual(row["input_sha256"], "abc")
        self.assertEqual(row["commit_version"], row["allocation_id"])
        nomination_columns = {item["name"] for item in connection.execute("PRAGMA table_info(nominations)")}
        self.assertIn("allocation_id", nomination_columns)
        unique_columns = []
        for index in connection.execute("PRAGMA index_list(allocation_runs)").fetchall():
            if index["unique"]:
                unique_columns = [item["name"] for item in connection.execute(f"PRAGMA index_info({index['name']})")]
        self.assertEqual(unique_columns, ["route_id", "service_date"])
        # 迁移是幂等的，重复初始化不会破坏数据
        initialize(connection)
        self.assertEqual(connection.execute("SELECT COUNT(*) AS c FROM allocation_runs").fetchone()["c"], 1)


if __name__ == "__main__":
    unittest.main()
