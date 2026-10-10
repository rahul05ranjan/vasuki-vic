"""Authoring source for eval set v2.

v2 revises v1 after four strong models were run on it: requests that several of them read differently
from the gold plan were reworded, labels a capable model could reasonably dispute were dropped or relaxed,
and requests that turned out to be expressible became plan tasks. The project's notes record the evidence.

Run `uv run python evals/v2/build.py` to regenerate tables.json and tasks.jsonl.
Expected results are produced by running each gold plan and are then frozen in
tasks.jsonl, so review the diff after any change here or in the executor.

These tables and requests are held out: nothing in this directory may be used
to generate training data.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from vasuki import Table, run_plan

HERE = Path(__file__).parent


def table(columns: str, *rows: tuple) -> list[dict]:
    names = columns.split()
    assert all(len(row) == len(names) for row in rows), columns
    return [dict(zip(names, row)) for row in rows]


TABLES = {
    "orders": table(
        "order_id customer_id placed_on status items unit_price discount_pct channel",
        (1001, "C01", "2026-01-05", "delivered", 2, 499.0, 10, "web"),
        (1002, "C02", "2026-01-07", "cancelled", 1, 1299.0, None, "app"),
        (1003, "C01", "2026-01-15", "delivered", 5, 120.5, 0, "web"),
        (1004, "C03", "2026-02-02", "pending", 3, 250.0, 5, "store"),
        (1005, "C04", "2026-02-11", "delivered", 1, 3999.0, 15, "app"),
        (1006, "C02", "2026-02-14", "returned", 2, 799.0, None, "web"),
        (1007, "C05", "2026-02-28", "delivered", 4, 60.0, 0, "store"),
        (1008, "C03", "2026-03-03", "delivered", 1, 1500.0, 20, "app"),
        (1009, "C09", "2026-03-09", "pending", 6, 75.25, 4, "web"),
        (1010, "C01", "2026-03-18", "cancelled", 2, 499.0, 10, "app"),
        (1011, "C04", "2026-03-21", "delivered", 3, 310.0, None, "store"),
        (1012, "C05", "2026-03-30", "delivered", 2, 45.0, 0, "web"),
    ),
    "customers": table(
        "customer_id name city tier joined_on",
        ("C01", "Meera Nair", "Kochi", "gold", "2024-06-01"),
        ("C02", "Arjun Rao", "Pune", "silver", "2025-01-12"),
        ("C03", "Lina Costa", "Goa", "basic", "2025-03-30"),
        ("C04", "Tomás Field", "Pune", "gold", "2023-11-05"),
        ("C05", "Wen Li", "Delhi", "basic", "2025-09-17"),
        ("C06", "Sana Qureshi", "Kochi", "silver", "2026-01-02"),
    ),
    "employees": table(
        "emp_id name dept role salary bonus_pct manager_id start_date city",
        ("E1", "Priya Menon", "ENG", "Staff Engineer", 185000, 12, None, "2019-04-01", "Bengaluru"),
        ("E2", "Daniel Okafor", "ENG", "Engineer", 120000, 8, "E1", "2022-07-18", "Bengaluru"),
        ("E3", "Hana Sato", "ENG", "Engineer", 125000, None, "E1", "2021-02-08", "Remote"),
        ("E4", "Luis Ortega", "SALES", "Account Exec", 90000, 20, "E6", "2023-01-09", "Mumbai"),
        ("E5", "Farah Khan", "SALES", "Account Exec", 95000, 20, "E6", "2020-10-26", "Delhi"),
        ("E6", "Greta Lund", "SALES", "Sales Lead", 140000, 25, None, "2018-05-14", "Mumbai"),
        ("E7", "Omar Haddad", "OPS", "Analyst", 70000, 5, "E9", "2024-03-04", "Delhi"),
        ("E8", "Chen Wei", "OPS", "Analyst", 72000, None, "E9", "2023-09-12", "Remote"),
        ("E9", "Ruth Abebe", "OPS", "Ops Manager", 110000, 10, None, "2017-08-21", "Delhi"),
        ("E10", "Ivan Petrov", "LEGAL", "Counsel", 150000, None, None, "2021-11-30", "Mumbai"),
    ),
    "departments": table(
        "dept dept_name floor budget",
        ("ENG", "Engineering", 4, 600000),
        ("SALES", "Sales", 2, 400000),
        ("OPS", "Operations", 1, 250000),
        ("HR", "People", 3, 120000),
    ),
    "readings": table(
        "reading_id device taken_at temp_c humidity_pct battery_mv uptime_s",
        ("r1", "th-01", "2026-05-01T06:00:00", 22.0, 40, 3700, 3600),
        ("r2", "th-01", "2026-05-01T12:00:00", 29.0, 35, 3650, 25200),
        ("r3", "th-01", "2026-05-01T12:00:00", 29.0, 35, 3650, 25200),
        ("r4", "th-02", "2026-05-01T06:00:00", 18.5, 55, 2900, 90),
        ("r5", "th-02", "2026-05-01T12:00:00", None, 52, 2850, 21690),
        ("r6", "th-02", "2026-05-01T18:00:00", 24.5, 50, 2800, 43290),
        ("r7", "th-03", "2026-05-01T06:00:00", -2.0, 80, 4100, 172800),
        ("r8", "th-03", "2026-05-01T12:00:00", 3.5, 75, 4050, 194400),
        ("r9", "th-03", "2026-05-02T06:00:00", -5.5, None, 4000, 259200),
        ("r10", "th-04", "2026-05-02T06:00:00", 37.5, 20, 3300, 600),
        ("r11", "th-04", "2026-05-02T12:00:00", 41.5, 15, 3200, 22200),
    ),
    "shipments": table(
        "shipment_id origin destination weight_lb distance_mi cost_usd carrier shipped_on delivered_on",
        ("S-01", "Austin", "Denver", 120, 920, 340.0, "NorthStar", "2026-04-01", "2026-04-04"),
        ("S-02", "Austin", "Miami", 45.5, 1340, 410.5, "BlueLine", "2026-04-02", "2026-04-08"),
        ("S-03", "Boston", "Denver", 300, 1970, 980.0, "NorthStar", "2026-04-03", "2026-04-09"),
        ("S-04", "Boston", "Austin", 10, 1960, 150.0, "Swift", "2026-04-05", None),
        ("S-05", "Denver", "Miami", 75, 2070, 520.25, "BlueLine", "2026-04-06", "2026-04-10"),
        ("S-06", "Miami", "Boston", 220, 1500, 760.0, "NorthStar", "2026-04-08", "2026-04-11"),
        ("S-07", "Austin", "Boston", 5.5, 1960, 95.0, "Swift", "2026-04-10", "2026-04-12"),
        ("S-08", "Denver", "Austin", 150, 920, 365.0, "BlueLine", "2026-04-11", None),
        ("S-09", "Miami", "Austin", 60, 1340, 300.0, "Swift", "2026-04-12", "2026-04-19"),
        ("S-10", "Boston", "Miami", 90, 1500, 445.0, "NorthStar", "2026-04-15", "2026-04-18"),
    ),
    "carriers": table(
        "carrier rating insured",
        ("NorthStar", 4.5, True),
        ("BlueLine", 3.8, False),
        ("Swift", 4.1, True),
    ),
    "students": table(
        "student_id name section math science english attendance_pct",
        (101, "Aditi", "A", 88, 92, 79, 96),
        (102, "Bilal", "A", 45, 58, None, 81),
        (103, "Carmen", "A", 73, 64, 70, 88),
        (104, "Dev", "B", 91, 85, 94, 99),
        (105, "Esi", "B", None, 77, 82, 74),
        (106, "Faisal", "B", 66, 49, 55, 90),
        (107, "Gauri", "C", 38, 41, 60, 62),
        (108, "Hiro", "C", 80, 80, 80, 93),
        (109, "Ines", "C", 59, None, 71, 85),
    ),
    "requests": table(
        "ts host endpoint method status latency_ms bytes",
        ("2026-06-01T10:00:01", "web-1", "/api/users", "GET", 200, 120, 2048),
        ("2026-06-01T10:00:02", "web-1", "/api/orders", "POST", 201, 340, 512),
        ("2026-06-01T10:00:04", "web-2", "/api/users", "GET", 200, 95, 2048),
        ("2026-06-01T10:00:05", "web-2", "/api/orders", "GET", 500, 2100, 128),
        ("2026-06-01T10:00:07", "web-1", "/health", "GET", 200, 4, 16),
        ("2026-06-01T10:00:09", "web-3", "/api/users", "GET", 404, 30, 256),
        ("2026-06-01T10:00:12", "web-3", "/api/orders", "POST", 201, 410, 512),
        ("2026-06-01T10:00:15", "web-2", "/api/search", "GET", 200, 880, 1048576),
        ("2026-06-01T10:00:18", "web-1", "/api/search", "GET", 503, 5000, 64),
        ("2026-06-01T10:00:21", "web-3", "/health", "GET", 200, 3, 16),
        ("2026-06-01T10:00:25", "web-2", "/api/users", "DELETE", 403, 25, 128),
        ("2026-06-01T10:00:30", "web-1", "/api/orders", "GET", 200, 150, 5242880),
    ),
    "stock": table(
        "sku product category on_hand reorder_level unit_cost supplier restocked_on",
        ("SKU-001", "Basmati Rice 5kg", "grocery", 40, 25, 310.0, "AgroMart", "2026-07-01"),
        ("SKU-002", "Toor Dal 1kg", "grocery", 12, 30, 145.0, "AgroMart", "2026-06-20"),
        ("SKU-003", "Olive Oil 1L", "grocery", 0, 10, 650.0, "MedFoods", "2026-05-11"),
        ("SKU-004", "LED Bulb 9W", "electrical", 200, 50, 62.5, "BrightCo", "2026-07-08"),
        ("SKU-005", "Extension Cord", "electrical", 18, 20, 240.0, "BrightCo", "2026-06-02"),
        ("SKU-006", "Notebook A5", "stationery", 500, 100, 28.0, "PaperWorks", "2026-07-10"),
        ("SKU-007", "Gel Pen Blue", "stationery", 75, 150, 9.5, "PaperWorks", "2026-06-15"),
        ("SKU-008", "Stapler", "stationery", 9, 5, 180.0, None, "2026-04-22"),
        ("SKU-009", "Green Tea 100g", "grocery", 33, 15, 220.0, "MedFoods", "2026-07-03"),
        ("SKU-010", "Battery AA 4pk", "electrical", 0, 40, 95.0, None, "2026-05-30"),
    ),
    "stock_raw": table(
        "sku qty unit_cost active",
        ("A1", "14", "2.50", "yes"),
        ("A2", "0", "10", "no"),
        ("A3", "7", "3.25", "yes"),
        ("A4", "120", "0.99", "yes"),
        ("A5", "3", "45.00", "no"),
        ("A6", "25", "12.40", "yes"),
    ),
    "subscriptions": table(
        "sub_id email plan monthly_fee seats started_on cancelled_on country",
        ("sub_01", "ana@northwind.io", "team", 49.0, 5, "2025-01-10", None, "PT"),
        ("sub_02", "raj@kite.dev", "solo", 12.0, 1, "2025-02-01", "2025-08-01", "IN"),
        ("sub_03", "mei@lotus.co", "enterprise", 199.0, 50, "2024-11-20", None, "SG"),
        ("sub_04", "tom@kite.dev", "team", 49.0, 7, "2025-05-05", None, "IN"),
        ("sub_05", "zoe@harbor.org", "solo", 12.0, 1, "2025-06-18", "2025-07-18", "US"),
        ("sub_06", "ali@northwind.io", "team", 49.0, 3, "2025-07-01", "2026-01-01", "PT"),
        ("sub_07", "eva@lotus.co", "enterprise", 199.0, 30, "2025-09-09", None, "SG"),
        ("sub_08", "sam@harbor.org", "solo", 15.0, 1, "2026-01-15", None, "US"),
        ("sub_09", "noor@dune.ae", "team", 55.0, 12, "2026-02-02", None, "AE"),
        ("sub_10", "kai@kite.dev", "solo", 15.0, 1, "2026-03-03", None, "IN"),
    ),
    "plans": table(
        "plan seat_limit support",
        ("solo", 1, "email"),
        ("team", 10, "chat"),
        ("enterprise", 100, "phone"),
    ),
    "expenses": table(
        "date category amount_inr paid_by mode note",
        ("2026-08-01", "groceries", 2350, "Kabir", "upi", "BigBasket"),
        ("2026-08-02", "rent", 28000, "Kabir", "bank", "August rent"),
        ("2026-08-03", "travel", 640, "Neha", "upi", "Metro card"),
        ("2026-08-05", "food", 1180, "Neha", "card", "Dinner"),
        ("2026-08-07", "groceries", 890, "Neha", "cash", "Sabzi mandi"),
        ("2026-08-09", "utilities", 3120, "Kabir", "upi", "Electricity bill"),
        ("2026-08-12", "food", 450, "Kabir", "upi", "Chai and snacks"),
        ("2026-08-15", "travel", 5400, "Kabir", "card", "Train tickets"),
        ("2026-08-18", "shopping", 2999, "Neha", "card", "Shoes"),
        ("2026-08-21", "groceries", 1640, "Kabir", "upi", "bigbasket order"),
        ("2026-08-25", "utilities", 799, "Neha", "upi", "Broadband"),
        ("2026-08-28", "food", 2200, "Neha", "card", "Birthday dinner"),
    ),
    "trips": table(
        "trip_id driver city distance_km duration_min fare tip started_at",
        ("T1", "Asha", "Pune", 12.4, 35, 310.0, 20.0, "2026-09-01T08:15:00"),
        ("T2", "Asha", "Pune", 3.1, 12, 95.0, None, "2026-09-01T09:05:00"),
        ("T3", "Bram", "Pune", 22.0, 48, 540.0, 50.0, "2026-09-01T09:30:00"),
        ("T4", "Chidi", "Mumbai", 8.7, 41, 260.0, 0.0, "2026-09-01T18:45:00"),
        ("T5", "Chidi", "Mumbai", 15.2, 55, 430.0, 30.0, "2026-09-02T07:50:00"),
        ("T6", "Bram", "Pune", 5.5, 20, 150.0, None, "2026-09-02T10:10:00"),
        ("T7", "Asha", "Pune", 30.5, 62, 720.0, 100.0, "2026-09-02T22:30:00"),
        ("T8", "Dalia", "Mumbai", 2.0, 15, 80.0, 10.0, "2026-09-03T08:00:00"),
        ("T9", "Dalia", "Mumbai", 18.9, 70, 510.0, None, "2026-09-03T19:20:00"),
        ("T10", "Chidi", "Mumbai", 11.0, 38, 330.0, 25.0, "2026-09-03T21:05:00"),
    ),
    "tickets": table(
        "ticket_id priority assignee hours_spent",
        ("K-1", " High", "mo", 3.5),
        ("K-2", "high", None, 1.0),
        ("K-3", "HIGH", "li", 6.0),
        ("K-4", "Low", "mo", 0.5),
        ("K-5", "low ", None, 2.0),
        ("K-6", "Medium", "li", 4.0),
        ("K-7", "medium", "mo", 1.5),
        ("K-8", "High", None, 8.0),
    ),
}


# --- plan shorthand -------------------------------------------------------

def flt(where):
    return {"op": "filter", "where": where}


def der(column, expr):
    return {"op": "derive", "column": column, "expr": expr}


def ren(**columns):
    return {"op": "rename", "columns": columns}


def sel(*columns):
    return {"op": "select", "columns": list(columns)}


def drop(*columns):
    return {"op": "drop", "columns": list(columns)}


def cast(column, to):
    return {"op": "cast", "column": column, "to": to}


def srt(*keys):
    """Sort keys; a leading '-' means descending."""
    by = [{"column": k[1:], "order": "desc"} if k.startswith("-") else {"column": k} for k in keys]
    return {"op": "sort", "by": by}


def dedup(*keys):
    return {"op": "deduplicate", "keys": list(keys)} if keys else {"op": "deduplicate"}


def agg(by, **metrics):
    """Each metric is name="count" or name=(fn, column)."""
    built = []
    for name, spec in metrics.items():
        fn, column = (spec, None) if isinstance(spec, str) else spec
        built.append({"as": name, "fn": fn, **({"column": column} if column else {})})
    return {"op": "aggregate", "group_by": list(by), "metrics": built}


def join(right, *on, how=None):
    step = {"op": "join", "right": right, "on": list(on)}
    return {**step, "how": how} if how else step


def conv(column, src, dst, to=None):
    step = {"op": "convert_unit", "column": column, "from": src, "to": dst}
    return {**step, "to_column": to} if to else step


def lim(n):
    return {"op": "limit", "n": n}


def at(table_name, step):
    return {**step, "input": table_name}


def named(name, step):
    return {**step, "as": name}


# --- task registration ----------------------------------------------------

TASKS: list[dict] = []
_counts: Counter = Counter()


def _add(category, tables, request, expect, tags, gold=None):
    _counts[category] += 1
    task = {
        "id": f"{category}-{_counts[category]:03d}",
        "category": category,
        "tags": list(tags),
        "request": request,
        "tables": [tables] if isinstance(tables, str) else list(tables),
        "expect": expect,
    }
    if gold:
        task["gold"] = gold
    TASKS.append(task)


def plan(category, tables, request, steps, ordered=False, tags=()):
    gold = {"status": "plan", "steps": steps}
    names = [tables] if isinstance(tables, str) else tables
    try:
        result = run_plan(gold, {name: Table.from_rows(name, TABLES[name]) for name in names})
    except Exception as e:
        raise SystemExit(f"gold plan failed for {request!r}: {e}") from e
    size = "steps:1" if len(steps) == 1 else "steps:2-3" if len(steps) <= 3 else "steps:4+"
    expect = {"status": ["plan"], "ordered": ordered, "columns": result.columns, "rows": result.rows}
    _add(category, tables, request, expect, [size, *tags], gold)


def ask(tables, request, or_refuse=False, tags=()):
    statuses = ["needs_clarification", "unsupported"] if or_refuse else ["needs_clarification"]
    _add("clarify", tables, request, {"status": statuses}, tags)


def refuse(tables, request, or_ask=False, tags=()):
    statuses = ["unsupported", "needs_clarification"] if or_ask else ["unsupported"]
    _add("unsupported", tables, request, {"status": statuses}, tags)


HI = ("hinglish",)

# --- basic: one or two simple steps ---------------------------------------

plan("basic", "orders", "Keep only the cancelled orders.", [flt("status == 'cancelled'")])
plan("basic", "orders", "Which orders came through the app or the store? Leave out web orders.",
     [flt("channel != 'web'")])
plan("basic", "orders", "List the orders with more than 2 items.", [flt("items > 2")])
plan("basic", "orders", "Sort the orders by date placed, newest first.", [srt("-placed_on")], ordered=True)
plan("basic", "orders", "Give me just the order id, status and channel columns.",
     [sel("order_id", "status", "channel")])
plan("basic", "orders", "Drop the discount column.", [drop("discount_pct")])
plan("basic", "orders", "Rename placed_on to order_date and items to quantity.",
     [ren(placed_on="order_date", items="quantity")])
plan("basic", "orders", "Show the 3 most recent orders, newest first.", [srt("-placed_on"), lim(3)], ordered=True)
plan("basic", "orders", "List the orders that have no discount recorded at all, with all their columns.", [flt("is_null(discount_pct)")])
plan("basic", "orders", "Orders placed in February 2026.",
     [flt("placed_on >= '2026-02-01' and placed_on <= '2026-02-28'")])
plan("basic", "customers", "Customers based in Pune or Kochi.", [flt("city in ['Pune', 'Kochi']")])
plan("basic", "customers", "Sort the customers alphabetically by name.", [srt("name")], ordered=True)
plan("basic", "customers", "Which distinct cities do we have customers in? One row per city, city column only.",
     [sel("city"), dedup()])
plan("basic", "employees", "Employees who have no manager.", [flt("is_null(manager_id)")])
plan("basic", "employees", "Employees earning at least 120000, highest paid first.",
     [flt("salary >= 120000"), srt("-salary")], ordered=True)
plan("basic", "employees", "Everyone outside the ENG department.", [flt("dept != 'ENG'")])
plan("basic", "employees", "Remote employees: return only their name and role.",
     [flt("city == 'Remote'"), sel("name", "role")])
plan("basic", "employees", "Rename emp_id to id, and remove the manager_id and bonus_pct columns.",
     [ren(emp_id="id"), drop("manager_id", "bonus_pct")])
plan("basic", "readings", "Readings from device th-03.", [flt("device == 'th-03'")])
plan("basic", "readings",
     "Remove repeated readings: two rows with the same device and timestamp are duplicates, keep the first.",
     [dedup("device", "taken_at")])
plan("basic", "readings", "Readings where the temperature is missing.", [flt("is_null(temp_c)")])
plan("basic", "readings", "Readings taken below freezing.", [flt("temp_c < 0")])
plan("basic", "readings", "The five readings with the lowest battery level, lowest first.",
     [srt("battery_mv"), lim(5)], ordered=True)
plan("basic", "shipments", "Shipments that have not been delivered yet.", [flt("is_null(delivered_on)")])
plan("basic", "shipments", "NorthStar shipments that weigh more than 100 lb.",
     [flt("carrier == 'NorthStar' and weight_lb > 100")])
plan("basic", "shipments", "Shipments that leave from Austin or arrive in Austin.",
     [flt("origin == 'Austin' or destination == 'Austin'")])
plan("basic", "shipments", "Keep shipment_id, origin and destination, ordered by shipment id descending.",
     [sel("shipment_id", "origin", "destination"), srt("-shipment_id")], ordered=True)
plan("basic", "students", "Students in section B.", [flt("section == 'B'")])
plan("basic", "students", "Students with attendance under 85 percent.", [flt("attendance_pct < 85")])
plan("basic", "students", "Students who are missing at least one subject score.",
     [flt("is_null(math) or is_null(science) or is_null(english)")])
plan("basic", "students", "Top 3 students by math score, best first. Return name and math only.",
     [srt("-math"), lim(3), sel("name", "math")], ordered=True)
plan("basic", "requests", "Only the server errors: status 500 or above.", [flt("status >= 500")])
plan("basic", "requests", "Requests that are not health checks.", [flt("endpoint != '/health'")])
plan("basic", "requests", "POST requests to /api/orders.", [flt("method == 'POST' and endpoint == '/api/orders'")])
plan("basic", "requests", "The 4 slowest requests, slowest first, showing ts, endpoint and latency_ms.",
     [srt("-latency_ms"), lim(4), sel("ts", "endpoint", "latency_ms")], ordered=True)
plan("basic", "stock", "Items that are completely out of stock.", [flt("on_hand == 0")])
plan("basic", "stock", "Items with no supplier on record.", [flt("is_null(supplier)")])
plan("basic", "stock", "Grocery items sorted by unit cost, cheapest first.",
     [flt("category == 'grocery'"), srt("unit_cost")], ordered=True)
plan("basic", "stock", "Items at or below their reorder level.", [flt("on_hand <= reorder_level")])
plan("basic", "subscriptions", "Subscriptions that are still active, meaning not cancelled.",
     [flt("is_null(cancelled_on)")])
plan("basic", "subscriptions", "Team or enterprise subscriptions from India or Singapore.",
     [flt("plan in ['team', 'enterprise'] and country in ['IN', 'SG']")])
plan("basic", "expenses", "Expenses paid by card.", [flt("mode == 'card'")])
plan("basic", "trips", "Trips longer than 15 km.", [flt("distance_km > 15")])
plan("basic", "trips", "Trips with no tip recorded.", [flt("is_null(tip)")])
plan("basic", "stock", "Jo items khatam ho chuke hain (on_hand 0) unka product aur supplier batao.",
     [flt("on_hand == 0"), sel("product", "supplier")], tags=HI)
plan("basic", "expenses",
     "1000 rupaye se upar wale kharche dikhao, sabse bada pehle. Sirf date, category aur amount_inr chahiye.",
     [flt("amount_inr > 1000"), srt("-amount_inr"), sel("date", "category", "amount_inr")], ordered=True, tags=HI)

# --- derive: row-level computation ----------------------------------------

plan("derive", "orders", "Add a column gross = items * unit_price.", [der("gross", "items * unit_price")])
plan("derive", "orders",
     "Add net_total: items times unit price, less the discount percentage. A missing discount means no "
     "discount. Round to 2 decimals.",
     [der("net_total", "round(items * unit_price * (1 - coalesce(discount_pct, 0) / 100), 2)")])
plan("derive", "orders", "Add a true/false column is_big that is true when items * unit_price is over 1000.",
     [der("is_big", "items * unit_price > 1000")])
plan("derive", "orders", "Add a column size: 'bulk' when items is 4 or more, otherwise 'regular'.",
     [der("size", "'bulk' if items >= 4 else 'regular'")])
plan("derive", "orders", "Replace missing discounts with 0, keeping the same column.",
     [der("discount_pct", "coalesce(discount_pct, 0)")])
plan("derive", "employees",
     "Add bonus = salary * bonus_pct / 100. Leave it empty when there is no bonus percentage.",
     [der("bonus", "salary * bonus_pct / 100")])
plan("derive", "employees",
     "Compute total_comp = salary plus bonus, where bonus is salary * bonus_pct / 100 and a missing percentage "
     "means no bonus. Return only name and total_comp.",
     [der("total_comp", "salary + salary * coalesce(bonus_pct, 0) / 100"), sel("name", "total_comp")])
plan("derive", "employees", "Add monthly_salary = salary / 12, rounded to the nearest whole number.",
     [der("monthly_salary", "round(salary / 12)")])
plan("derive", "employees",
     "Add a band column: 'senior' for salary of 140000 or more, 'mid' for 100000 up to 139999, 'junior' below "
     "100000.",
     [der("band", "'senior' if salary >= 140000 else ('mid' if salary >= 100000 else 'junior')")])
plan("derive", "employees", "Give everyone in OPS a 10% raise by updating salary in place. Others stay unchanged.",
     [der("salary", "salary * 1.1 if dept == 'OPS' else salary")])
plan("derive", "students",
     "Add total = math + science + english. If any of the three is missing, leave total empty.",
     [der("total", "math + science + english")])
plan("derive", "students", "Add total = math + science + english, counting a missing score as 0.",
     [der("total", "coalesce(math, 0) + coalesce(science, 0) + coalesce(english, 0)")])
plan("derive", "students",
     "Add passed: true if math is at least 50, false otherwise. A missing math score counts as not passed.",
     [der("passed", "coalesce(math, 0) >= 50")])
plan("derive", "students",
     "Add best = the highest of the three subject scores for each student, treating a missing score as 0.",
     [der("best", "max(coalesce(math, 0), coalesce(science, 0), coalesce(english, 0))")])
plan("derive", "readings", "Add low_battery, true when battery_mv is under 3000.",
     [der("low_battery", "battery_mv < 3000")])
plan("derive", "shipments", "Add cost_per_mile = cost_usd / distance_mi, rounded to 3 decimals.",
     [der("cost_per_mile", "round(cost_usd / distance_mi, 3)")])
plan("derive", "shipments",
     "Add transit_days, the number of days from shipped_on to delivered_on. Empty if not delivered.",
     [der("transit_days", "days_between(shipped_on, delivered_on)")])
plan("derive", "shipments", "Add a route column that reads like 'Austin -> Denver', from origin and destination.",
     [der("route", "concat(origin, ' -> ', destination)")])
plan("derive", "requests", "Add is_error, true when the status is 400 or higher.",
     [der("is_error", "status >= 400")])
plan("derive", "stock", "Add stock_value = on_hand * unit_cost.", [der("stock_value", "on_hand * unit_cost")])
plan("derive", "stock",
     "Add shortfall: how many units each item is below its reorder level, or 0 if it is at or above it.",
     [der("shortfall", "max(reorder_level - on_hand, 0)")])
plan("derive", "trips",
     "Add speed_kmh = distance divided by duration in hours, rounded to 1 decimal.",
     [der("speed_kmh", "round(distance_km / (duration_min / 60), 1)")])
plan("derive", "trips", "Add total_paid = fare + tip, counting a missing tip as 0.",
     [der("total_paid", "fare + coalesce(tip, 0)")])
plan("derive", "subscriptions", "Add yearly_fee = monthly_fee * 12.", [der("yearly_fee", "monthly_fee * 12")])
plan("derive", "subscriptions", "Add fee_per_seat = monthly_fee / seats, rounded to 2 decimals.",
     [der("fee_per_seat", "round(monthly_fee / seats, 2)")])
plan("derive", "expenses", "Add a true/false column is_large for expenses of 2500 rupees or more.",
     [der("is_large", "amount_inr >= 2500")])
plan("derive", "orders",
     "Har order ka total nikalo (items * unit_price) aur sirf 1000 se zyada wale orders rakho. order_id aur "
     "total dikhao.",
     [der("total", "items * unit_price"), flt("total > 1000"), sel("order_id", "total")], tags=HI)

# --- aggregate: one grouping step ------------------------------------------

plan("aggregate", "orders", "How many orders are there per status? Columns: status, orders.",
     [agg(["status"], orders="count")])
plan("aggregate", "orders", "Total items ordered per channel, as total_items.",
     [agg(["channel"], total_items=("sum", "items"))])
plan("aggregate", "orders", "Average unit price per channel, as avg_price.",
     [agg(["channel"], avg_price=("mean", "unit_price"))])
plan("aggregate", "orders", "How many different customers have placed orders? One row with a column customers.",
     [agg([], customers=("count_distinct", "customer_id"))])
plan("aggregate", "orders",
     "For each customer: number of orders and the highest unit price they paid. Columns customer_id, orders, "
     "max_price.",
     [agg(["customer_id"], orders="count", max_price=("max", "unit_price"))])
plan("aggregate", "orders", "The earliest and latest order dates overall, as first_order and last_order.",
     [agg([], first_order=("min", "placed_on"), last_order=("max", "placed_on"))])
plan("aggregate", "employees", "Headcount and total salary by department: dept, headcount, payroll.",
     [agg(["dept"], headcount="count", payroll=("sum", "salary"))])
plan("aggregate", "employees", "Median salary per department, as median_salary.",
     [agg(["dept"], median_salary=("median", "salary"))])
plan("aggregate", "employees",
     "Average bonus percentage per department, ignoring people with no bonus percentage. Every department "
     "appears; where nobody has a bonus percentage the average is empty. Column avg_bonus_pct.",
     [agg(["dept"], avg_bonus_pct=("mean", "bonus_pct"))])
plan("aggregate", "employees", "How many employees are in each city? Columns city, employees.",
     [agg(["city"], employees="count")])
plan("aggregate", "employees", "The highest and lowest salary in the company, in one row: max_salary, min_salary.",
     [agg([], max_salary=("max", "salary"), min_salary=("min", "salary"))])
plan("aggregate", "employees",
     "Per department, how many people have a bonus percentage recorded? Every department appears, with 0 "
     "where nobody has one. Column with_bonus.",
     [agg(["dept"], with_bonus=("count", "bonus_pct"))])
plan("aggregate", "students", "Average math score per section, as avg_math. Skip missing scores.",
     [agg(["section"], avg_math=("mean", "math"))])
plan("aggregate", "students",
     "Per section: number of students, and how many of them have an english score. Columns section, students, "
     "with_english.",
     [agg(["section"], students="count", with_english=("count", "english"))])
plan("aggregate", "students", "Highest science score in each section, as top_science.",
     [agg(["section"], top_science=("max", "science"))])
plan("aggregate", "students",
     "Overall averages for math, science and english in a single row: avg_math, avg_science, avg_english.",
     [agg([], avg_math=("mean", "math"), avg_science=("mean", "science"), avg_english=("mean", "english"))])
plan("aggregate", "readings", "Per device: number of readings and average temperature (readings, avg_temp).",
     [agg(["device"], readings="count", avg_temp=("mean", "temp_c"))])
plan("aggregate", "readings", "Lowest and highest battery level per device: min_battery, max_battery.",
     [agg(["device"], min_battery=("min", "battery_mv"), max_battery=("max", "battery_mv"))])
plan("aggregate", "shipments", "Total cost per carrier, as total_cost.",
     [agg(["carrier"], total_cost=("sum", "cost_usd"))])
plan("aggregate", "shipments",
     "Number of shipments and total weight leaving each origin city: origin, shipments, total_weight.",
     [agg(["origin"], shipments="count", total_weight=("sum", "weight_lb"))])
plan("aggregate", "shipments", "Average distance per carrier, as avg_distance.",
     [agg(["carrier"], avg_distance=("mean", "distance_mi"))])
plan("aggregate", "requests", "Number of requests per endpoint, as hits.", [agg(["endpoint"], hits="count")])
plan("aggregate", "requests", "Per host: request count and average latency (requests, avg_latency_ms).",
     [agg(["host"], requests="count", avg_latency_ms=("mean", "latency_ms"))])
plan("aggregate", "requests", "Total bytes sent per HTTP method, as total_bytes.",
     [agg(["method"], total_bytes=("sum", "bytes"))])
plan("aggregate", "requests", "How many different endpoints did each host serve? Column endpoints.",
     [agg(["host"], endpoints=("count_distinct", "endpoint"))])
plan("aggregate", "stock", "Per category: number of SKUs and total units on hand (skus, units).",
     [agg(["category"], skus="count", units=("sum", "on_hand"))])
plan("aggregate", "stock",
     "Average unit cost per supplier, as avg_cost. Items with no supplier form their own group, whose "
     "supplier stays empty.",
     [agg(["supplier"], avg_cost=("mean", "unit_cost"))])
plan("aggregate", "subscriptions", "Total seats per plan. Columns plan, seats.",
     [agg(["plan"], seats=("sum", "seats"))])
plan("aggregate", "subscriptions", "Number of subscriptions per country, as subscriptions.",
     [agg(["country"], subscriptions="count")])
plan("aggregate", "expenses", "Total spent per category, as total.", [agg(["category"], total=("sum", "amount_inr"))])
plan("aggregate", "expenses", "Per person: total spent and number of expenses. Columns paid_by, total, count.",
     [agg(["paid_by"], total=("sum", "amount_inr"), count="count")])
plan("aggregate", "trips", "Per driver: number of trips, total distance and total fare (trips, total_km, total_fare).",
     [agg(["driver"], trips="count", total_km=("sum", "distance_km"), total_fare=("sum", "fare"))])
plan("aggregate", "trips", "Average tip per city, ignoring trips with no tip recorded. Column avg_tip.",
     [agg(["city"], avg_tip=("mean", "tip"))])
plan("aggregate", "trips", "Median trip duration per city, as median_duration.",
     [agg(["city"], median_duration=("median", "duration_min"))])
plan("aggregate", "students", "Section-wise sabse zyada attendance kitni hai? Columns section, max_attendance.",
     [agg(["section"], max_attendance=("max", "attendance_pct"))], tags=HI)

# --- multi: several steps on one table --------------------------------------

plan("multi", "orders",
     "For delivered orders only, compute revenue as items * unit_price and total it per channel. Columns "
     "channel, revenue, highest revenue first.",
     [flt("status == 'delivered'"), der("revenue", "items * unit_price"),
      agg(["channel"], revenue=("sum", "revenue")), srt("-revenue")], ordered=True)
plan("multi", "orders",
     "Net revenue per customer, excluding cancelled orders. Net is items * unit_price reduced by the discount "
     "percentage, with a missing discount treated as 0. Return the top 3 customers as customer_id, "
     "net_revenue, biggest first.",
     [flt("status != 'cancelled'"),
      der("net", "items * unit_price * (1 - coalesce(discount_pct, 0) / 100)"),
      agg(["customer_id"], net_revenue=("sum", "net")), srt("-net_revenue"), lim(3)], ordered=True)
plan("multi", "orders",
     "Order counts by month: take the month number from placed_on and count orders per month. Columns month, "
     "orders, latest month first.",
     [der("month", "month(placed_on)"), agg(["month"], orders="count"), srt("-month")], ordered=True)
plan("multi", "orders",
     "Share of orders by channel: the number of orders per channel, and pct, that channel's share of all "
     "orders as a percentage rounded to 1 decimal. Columns channel, orders, pct.",
     [at("orders", der("k", "1")), named("by_channel", agg(["k", "channel"], orders="count")),
      at("orders", agg([], total="count")), named("overall", der("k", "1")),
      at("by_channel", join("overall", "k")), der("pct", "round(orders * 100 / total, 1)"),
      sel("channel", "orders", "pct")])
plan("multi", "orders", "Customers with more than 2 orders: customer_id and orders.",
     [agg(["customer_id"], orders="count"), flt("orders > 2")])
plan("multi", "orders",
     "Compare web orders with all other orders: add is_web (true or false), then give the average order value "
     "(items * unit_price) for each. Columns is_web, avg_value.",
     [der("value", "items * unit_price"), der("is_web", "channel == 'web'"),
      agg(["is_web"], avg_value=("mean", "value"))])
plan("multi", "employees",
     "Total compensation per department, where compensation is salary plus salary * bonus_pct / 100 and a "
     "missing bonus counts as 0. Keep only departments whose total is above 300000. Columns dept, total_comp, "
     "smallest first.",
     [der("comp", "salary + salary * coalesce(bonus_pct, 0) / 100"),
      agg(["dept"], total_comp=("sum", "comp")), flt("total_comp > 300000"), srt("total_comp")], ordered=True)
plan("multi", "employees",
     "How many direct reports does each manager have? Leave out people who have no manager. Columns "
     "manager_id, reports.",
     [flt("not is_null(manager_id)"), agg(["manager_id"], reports="count")])
plan("multi", "employees",
     "How many employees started in each year? Columns start_year, employees, earliest year first.",
     [der("start_year", "year(start_date)"), agg(["start_year"], employees="count"), srt("start_year")],
     ordered=True)
plan("multi", "employees",
     "The 3 highest paid employees outside ENG, highest first. Return name, dept and salary.",
     [flt("dept != 'ENG'"), srt("-salary"), lim(3), sel("name", "dept", "salary")], ordered=True)
plan("multi", "employees",
     "Salary gap per department: highest salary minus lowest, as gap. Only departments with more than one "
     "person. Columns dept, gap.",
     [agg(["dept"], top=("max", "salary"), bottom=("min", "salary"), n="count"), flt("n > 1"),
      der("gap", "top - bottom"), sel("dept", "gap")])
plan("multi", "students",
     "Average total score per section, where a student's total is math + science + english with missing "
     "scores counted as 0. Columns section, avg_total rounded to 1 decimal, best section first.",
     [der("total", "coalesce(math, 0) + coalesce(science, 0) + coalesce(english, 0)"),
      agg(["section"], avg_total=("mean", "total")), der("avg_total", "round(avg_total, 1)"),
      srt("-avg_total")], ordered=True)
plan("multi", "students",
     "Students who scored at least 50 in all three subjects; a missing score is a fail. Return name, section "
     "and attendance_pct, highest attendance first.",
     [flt("math >= 50 and science >= 50 and english >= 50"), sel("name", "section", "attendance_pct"),
      srt("-attendance_pct")], ordered=True)
plan("multi", "students",
     "Math pass rate per section: among students who have a math score, the fraction scoring 50 or more. "
     "Columns section, pass_rate rounded to 2 decimals.",
     [flt("not is_null(math)"), der("ok", "1 if math >= 50 else 0"),
      agg(["section"], passed=("sum", "ok"), n="count"), der("pass_rate", "round(passed / n, 2)"),
      sel("section", "pass_rate")])
plan("multi", "readings",
     "After removing duplicate readings (same device and timestamp), give the average temperature per device "
     "as avg_temp rounded to 1 decimal, warmest device first.",
     [dedup("device", "taken_at"), agg(["device"], avg_temp=("mean", "temp_c")),
      der("avg_temp", "round(avg_temp, 1)"), srt("-avg_temp")], ordered=True)
plan("multi", "readings",
     "Devices whose battery ever dropped below 3000 mV: device and its lowest battery reading as min_battery.",
     [agg(["device"], min_battery=("min", "battery_mv")), flt("min_battery < 3000")])
plan("multi", "readings",
     "Per day of the month (from taken_at): number of readings and the highest temperature. Columns day, "
     "readings, max_temp.",
     [der("day", "day(taken_at)"), agg(["day"], readings="count", max_temp=("max", "temp_c"))])
plan("multi", "shipments",
     "Average transit time in days per carrier, from shipped_on to delivered_on; undelivered shipments do not "
     "count. Columns carrier, avg_days, slowest carrier first.",
     [der("days", "days_between(shipped_on, delivered_on)"), agg(["carrier"], avg_days=("mean", "days")),
      srt("-avg_days")], ordered=True)
plan("multi", "shipments",
     "For shipments under 100 lb, total cost per route (origin and destination). Return the 3 most expensive "
     "routes as origin, destination, total_cost, most expensive first.",
     [flt("weight_lb < 100"), agg(["origin", "destination"], total_cost=("sum", "cost_usd")),
      srt("-total_cost"), lim(3)], ordered=True)
plan("multi", "shipments",
     "Slow shipments: those that took more than 5 days from shipped_on to delivered_on. Return shipment_id, "
     "carrier and transit_days, longest first, ties by shipment_id ascending.",
     [der("transit_days", "days_between(shipped_on, delivered_on)"), flt("transit_days > 5"),
      sel("shipment_id", "carrier", "transit_days"), srt("-transit_days", "shipment_id")], ordered=True)
plan("multi", "requests",
     "Error rate per host: the fraction of that host's requests with status 500 or above. Columns host, "
     "error_rate rounded to 2 decimals.",
     [der("err", "1 if status >= 500 else 0"), agg(["host"], errors=("sum", "err"), n="count"),
      der("error_rate", "round(errors / n, 2)"), sel("host", "error_rate")])
plan("multi", "requests",
     "Average latency per endpoint, leaving out health checks. Columns endpoint, avg_latency_ms, slowest "
     "endpoint first.",
     [flt("endpoint != '/health'"), agg(["endpoint"], avg_latency_ms=("mean", "latency_ms")),
      srt("-avg_latency_ms")], ordered=True)
plan("multi", "requests", "Which endpoint and method pairs were hit more than once? Columns endpoint, method, hits.",
     [agg(["endpoint", "method"], hits="count"), flt("hits > 1")])
plan("multi", "stock",
     "Inventory value per category, where value is on_hand * unit_cost summed. Columns category, value, "
     "smallest first.",
     [der("value", "on_hand * unit_cost"), agg(["category"], value=("sum", "value")), srt("value")],
     ordered=True)
plan("multi", "stock",
     "Reorder list: for items at or below their reorder level, order_qty = 2 * reorder_level - on_hand. "
     "Return sku, product and order_qty, biggest order first.",
     [flt("on_hand <= reorder_level"), der("order_qty", "2 * reorder_level - on_hand"),
      sel("sku", "product", "order_qty"), srt("-order_qty")], ordered=True)
plan("multi", "stock",
     "Per supplier, skipping items with no supplier: number of items and total stock value (on_hand * "
     "unit_cost). Columns supplier, items, value, alphabetical by supplier.",
     [flt("not is_null(supplier)"), der("value", "on_hand * unit_cost"),
      agg(["supplier"], items="count", value=("sum", "value")), srt("supplier")], ordered=True)
plan("multi", "subscriptions",
     "Monthly recurring revenue from active (not cancelled) subscriptions per plan. monthly_fee is the fee "
     "for the whole subscription, so mrr is its sum. Columns plan, mrr, highest first.",
     [flt("is_null(cancelled_on)"), agg(["plan"], mrr=("sum", "monthly_fee")), srt("-mrr")], ordered=True)
plan("multi", "subscriptions",
     "Churn per plan: how many subscriptions were cancelled, and churn_rate, the cancelled fraction of that "
     "plan's subscriptions (between 0 and 1) rounded to 2 decimals. Columns plan, cancelled, churn_rate.",
     [der("gone", "0 if is_null(cancelled_on) else 1"), agg(["plan"], cancelled=("sum", "gone"), n="count"),
      der("churn_rate", "round(cancelled / n, 2)"), sel("plan", "cancelled", "churn_rate")])
plan("multi", "subscriptions",
     "How long did cancelled subscriptions last? days_active is the days from started_on to cancelled_on. "
     "Return sub_id, plan and days_active, longest first.",
     [flt("not is_null(cancelled_on)"), der("days_active", "days_between(started_on, cancelled_on)"),
      sel("sub_id", "plan", "days_active"), srt("-days_active")], ordered=True)
plan("multi", "expenses",
     "The top 2 spending categories, not counting rent. Columns category, total, highest first.",
     [flt("category != 'rent'"), agg(["category"], total=("sum", "amount_inr")), srt("-total"), lim(2)],
     ordered=True)
plan("multi", "expenses",
     "Each person's share of total spending, as a percentage rounded to 1 decimal. Columns paid_by, share_pct.",
     [at("expenses", der("k", "1")), named("by_person", agg(["k", "paid_by"], total=("sum", "amount_inr"))),
      at("expenses", agg([], grand=("sum", "amount_inr"))), named("overall", der("k", "1")),
      at("by_person", join("overall", "k")), der("share_pct", "round(total * 100 / grand, 1)"),
      sel("paid_by", "share_pct")])
plan("multi", "expenses", "How much did each person pay by UPI? Columns paid_by, upi_total.",
     [flt("mode == 'upi'"), agg(["paid_by"], upi_total=("sum", "amount_inr"))])
plan("multi", "trips",
     "Driver earnings: fares plus tips, with a missing tip counted as 0, summed per driver. Columns driver, "
     "earnings, highest first.",
     [der("earned", "fare + coalesce(tip, 0)"), agg(["driver"], earnings=("sum", "earned")),
      srt("-earnings")], ordered=True)
plan("multi", "trips",
     "Average speed per city in km/h, calculated as total distance divided by total hours. Columns city, "
     "speed_kmh rounded to 1 decimal.",
     [agg(["city"], km=("sum", "distance_km"), minutes=("sum", "duration_min")),
      der("speed_kmh", "round(km / (minutes / 60), 1)"), sel("city", "speed_kmh")])
plan("multi", "trips",
     "For trips with a tip above 0, the tip as a percentage of the fare, averaged per driver. Columns driver, "
     "avg_tip_pct rounded to 1 decimal.",
     [flt("tip > 0"), der("tip_pct", "tip / fare * 100"), agg(["driver"], avg_tip_pct=("mean", "tip_pct")),
      der("avg_tip_pct", "round(avg_tip_pct, 1)")])
plan("multi", "expenses", "Neha ne kitna kharcha kiya, payment mode ke hisaab se batao. Columns mode, total.",
     [flt("paid_by == 'Neha'"), agg(["mode"], total=("sum", "amount_inr"))], tags=HI)
plan("multi", "expenses",
     "Rent ko chhod ke, har bande ka total aur average kharcha batao. Columns paid_by, total, avg. avg ko 2 "
     "decimal tak round karo.",
     [flt("category != 'rent'"), agg(["paid_by"], total=("sum", "amount_inr"), avg=("mean", "amount_inr")),
      der("avg", "round(avg, 2)")], tags=HI)
plan("multi", "expenses", "Food aur groceries ka combined total kitna hai? Ek hi row chahiye, column total.",
     [flt("category in ['food', 'groceries']"), agg([], total=("sum", "amount_inr"))], tags=HI)
plan("multi", "trips",
     "Har driver ki average speed km/h mein nikalo (total distance / total ghante), 1 decimal tak. Columns "
     "driver, speed_kmh, sabse dheema driver sabse upar.",
     [agg(["driver"], km=("sum", "distance_km"), minutes=("sum", "duration_min")),
      der("speed_kmh", "round(km / (minutes / 60), 1)"), sel("driver", "speed_kmh"), srt("speed_kmh")],
     ordered=True, tags=HI)
plan("multi", "trips", "Jin trips mein tip nahi mili (blank ya 0), unka trip_id aur fare dikhao.",
     [flt("coalesce(tip, 0) == 0"), sel("trip_id", "fare")], tags=HI)
plan("multi", "orders",
     "Cancelled orders hata do aur baaki orders ka channel-wise count do. Columns channel, orders.",
     [flt("status != 'cancelled'"), agg(["channel"], orders="count")], tags=HI)
plan("multi", "students",
     "Sirf un bachchon ka total nikalo jinke teeno subjects ke marks available hain. Columns name, total, "
     "sabse zyada total pehle.",
     [flt("not is_null(math) and not is_null(science) and not is_null(english)"),
      der("total", "math + science + english"), srt("-total"), sel("name", "total")],
     ordered=True, tags=HI)
plan("multi", "employees",
     "Mumbai aur Delhi ke employees ki average salary city-wise batao. Columns city, avg_salary.",
     [flt("city in ['Mumbai', 'Delhi']"), agg(["city"], avg_salary=("mean", "salary"))], tags=HI)

# --- join: two tables --------------------------------------------------------

OC = ["orders", "customers"]
ED = ["employees", "departments"]
SC = ["shipments", "carriers"]
SP = ["subscriptions", "plans"]

plan("join", OC,
     "Add the customer's name and city to every order. An order whose customer is not in the customers "
     "table stays in the result with empty name and city. Return order_id, name, city.",
     [at("orders", join("customers", "customer_id", how="left")), sel("order_id", "name", "city")])
plan("join", OC, "Orders placed by gold-tier customers: order_id, name, items.",
     [at("orders", join("customers", "customer_id")), flt("tier == 'gold'"), sel("order_id", "name", "items")])
plan("join", OC,
     "Revenue (items * unit_price) by customer city, for delivered orders only. Orders from unknown customers "
     "are excluded. Columns city, revenue.",
     [at("orders", join("customers", "customer_id")), flt("status == 'delivered'"),
      der("revenue", "items * unit_price"), agg(["city"], revenue=("sum", "revenue"))])
plan("join", OC, "Customers who have never placed an order: customer_id and name.",
     [named("counts", at("orders", agg(["customer_id"], orders="count"))),
      at("customers", join("counts", "customer_id", how="left")), flt("is_null(orders)"),
      sel("customer_id", "name")])
plan("join", OC, "Orders whose customer_id has no matching customer record: order_id and customer_id.",
     [at("orders", join("customers", "customer_id", how="left")), flt("is_null(name)"),
      sel("order_id", "customer_id")])
plan("join", OC,
     "Per customer tier: number of orders and total items, counting known customers only. Columns tier, "
     "orders, items.",
     [at("orders", join("customers", "customer_id")), agg(["tier"], orders="count", items=("sum", "items"))])
plan("join", OC,
     "Every customer with their number of orders, 0 for customers with none. Columns name, orders, fewest "
     "orders first, ties alphabetical by name.",
     [named("counts", at("orders", agg(["customer_id"], orders="count"))),
      at("customers", join("counts", "customer_id", how="left")), der("orders", "coalesce(orders, 0)"),
      sel("name", "orders"), srt("orders", "name")], ordered=True)
plan("join", OC,
     "For each order from a known customer, the number of days between the customer joining and the order "
     "being placed. Columns order_id, name, days_since_join.",
     [at("orders", join("customers", "customer_id")),
      der("days_since_join", "days_between(joined_on, placed_on)"), sel("order_id", "name", "days_since_join")])
plan("join", ED,
     "Add the department name and floor to each employee. An employee whose department is not in the "
     "departments table stays in the result with empty dept_name and floor. Return name, dept_name, floor.",
     [at("employees", join("departments", "dept", how="left")), sel("name", "dept_name", "floor")])
plan("join", ED,
     "Payroll against budget for every department in the departments table: payroll is total salary (0 if the "
     "department has no employees) and remaining is budget minus payroll. Columns dept, payroll, budget, "
     "remaining.",
     [named("totals", at("employees", agg(["dept"], payroll=("sum", "salary")))),
      at("departments", join("totals", "dept", how="left")), der("payroll", "coalesce(payroll, 0)"),
      der("remaining", "budget - payroll"), sel("dept", "payroll", "budget", "remaining")])
plan("join", ED,
     "Departments whose total salaries exceed their budget: dept_name and overspend, the amount over budget.",
     [named("totals", at("employees", agg(["dept"], payroll=("sum", "salary")))),
      at("departments", join("totals", "dept")), flt("payroll > budget"),
      der("overspend", "payroll - budget"), sel("dept_name", "overspend")])
plan("join", "employees",
     "Each employee alongside their manager's name. People without a manager are left out. Columns name, "
     "manager_name.",
     [at("employees", sel("emp_id", "name")), named("managers", ren(emp_id="manager_id", name="manager_name")),
      at("employees", join("managers", "manager_id")), sel("name", "manager_name")])
plan("join", ED, "Employees on floor 2 or floor 4 who earn more than 100000: name, floor, salary.",
     [at("employees", join("departments", "dept")), flt("floor in [2, 4] and salary > 100000"),
      sel("name", "floor", "salary")])
plan("join", ED, "Which departments have no employees? Return dept and dept_name.",
     [named("heads", at("employees", agg(["dept"], n="count"))),
      at("departments", join("heads", "dept", how="left")), flt("is_null(n)"), sel("dept", "dept_name")])
plan("join", SC, "Shipments handled by insured carriers: shipment_id, carrier, cost_usd.",
     [at("shipments", join("carriers", "carrier")), flt("insured"), sel("shipment_id", "carrier", "cost_usd")])
plan("join", SC,
     "The average carrier rating across all shipments, each shipment counting once. One row, avg_rating "
     "rounded to 1 decimal.",
     [at("shipments", join("carriers", "carrier")), agg([], avg_rating=("mean", "rating")),
      der("avg_rating", "round(avg_rating, 1)")])
plan("join", SC,
     "Total shipping cost per carrier with the carrier's rating alongside. Columns carrier, rating, "
     "total_cost, best rated first.",
     [at("shipments", agg(["carrier"], total_cost=("sum", "cost_usd"))), join("carriers", "carrier"),
      sel("carrier", "rating", "total_cost"), srt("-rating")], ordered=True)
plan("join", SP, "Subscriptions using more seats than their plan allows: sub_id, plan, seats, seat_limit.",
     [at("subscriptions", join("plans", "plan")), flt("seats > seat_limit"),
      sel("sub_id", "plan", "seats", "seat_limit")])
plan("join", SP,
     "Seat utilisation per plan for active subscriptions: total seats divided by (seat_limit times the number "
     "of active subscriptions on the plan). Columns plan, utilisation rounded to 2 decimals.",
     [at("subscriptions", flt("is_null(cancelled_on)")), agg(["plan"], used=("sum", "seats"), subs="count"),
      join("plans", "plan"), der("utilisation", "round(used / (seat_limit * subs), 2)"),
      sel("plan", "utilisation")])
plan("join", SP, "Active subscriptions whose plan includes phone or chat support: email, plan, support.",
     [at("subscriptions", flt("is_null(cancelled_on)")), join("plans", "plan"),
      flt("support in ['phone', 'chat']"), sel("email", "plan", "support")])

plan("join", "employees",
     "Rank employees by salary within their department, where 1 is the highest paid. Return name, dept, salary "
     "and salary_rank.",
     [named("others", at("employees", ren(salary="other_salary"))), at("employees", join("others", "dept")),
      der("higher", "1 if other_salary > salary else 0"),
      agg(["name", "dept", "salary"], higher=("sum", "higher")), der("salary_rank", "higher + 1"),
      sel("name", "dept", "salary", "salary_rank")])
plan("join", "requests",
     "Add a running total of bytes sent in time order: for each request, running_bytes is the sum of bytes for "
     "it and every earlier request. Columns ts, bytes, running_bytes, newest first.",
     [named("base", at("requests", der("k", "1"))), named("others", ren(ts="other_ts", bytes="other_bytes")),
      at("base", join("others", "k")), flt("other_ts <= ts"),
      agg(["ts", "bytes"], running_bytes=("sum", "other_bytes")), srt("-ts")], ordered=True)

# --- units ---------------------------------------------------------------------

plan("units", "readings", "Add temp_f with the temperature in Fahrenheit.",
     [conv("temp_c", "C", "F", to="temp_f")])
plan("units", "readings", "Add battery_v, the battery level in volts.", [der("battery_v", "battery_mv / 1000")])
plan("units", "readings", "Convert uptime from seconds to hours: replace uptime_s with a column uptime_h.",
     [conv("uptime_s", "s", "h", to="uptime_h"), drop("uptime_s")])
plan("units", "readings", "Add humidity, the humidity as a fraction between 0 and 1.",
     [conv("humidity_pct", "percent", "fraction", to="humidity")])
plan("units", "readings", "Add temp_k with the temperature in kelvin.", [conv("temp_c", "C", "K", to="temp_k")])
plan("units", "shipments", "Add weight_kg, the weight converted from pounds and rounded to 1 decimal.",
     [conv("weight_lb", "lb", "kg", to="weight_kg"), der("weight_kg", "round(weight_kg, 1)")])
plan("units", "shipments", "Add distance_km, the distance converted from miles and rounded to a whole number.",
     [conv("distance_mi", "mi", "km", to="distance_km"), der("distance_km", "round(distance_km)")])
plan("units", "shipments",
     "Total shipped weight per carrier in kilograms. Columns carrier, total_kg rounded to 1 decimal.",
     [agg(["carrier"], lb=("sum", "weight_lb")), conv("lb", "lb", "kg", to="total_kg"),
      der("total_kg", "round(total_kg, 1)"), sel("carrier", "total_kg")])
plan("units", "shipments",
     "Shipments heavier than 50 kg: shipment_id and weight_kg, the weight in kilograms rounded to 1 decimal.",
     [conv("weight_lb", "lb", "kg", to="weight_kg"), flt("weight_kg > 50"),
      der("weight_kg", "round(weight_kg, 1)"), sel("shipment_id", "weight_kg")])
plan("units", "requests", "Add latency_s, the latency in seconds.", [conv("latency_ms", "ms", "s", to="latency_s")])
plan("units", "requests",
     "Total data sent per host in megabytes (1 MB = 1,000,000 bytes). Columns host, mb rounded to 2 decimals.",
     [agg(["host"], total=("sum", "bytes")), conv("total", "B", "MB", to="mb"), der("mb", "round(mb, 2)"),
      sel("host", "mb")])
plan("units", "requests",
     "Requests that sent at least 1 MiB (1,048,576 bytes): ts, endpoint, and mib, the size in mebibytes.",
     [conv("bytes", "B", "MiB", to="mib"), flt("mib >= 1"), sel("ts", "endpoint", "mib")])
plan("units", "trips", "Add distance_mi, the distance in miles rounded to 2 decimals.",
     [conv("distance_km", "km", "mi", to="distance_mi"), der("distance_mi", "round(distance_mi, 2)")])
plan("units", "trips", "Add duration_h, the trip duration in hours.", [conv("duration_min", "min", "h", to="duration_h")])
plan("units", "trips",
     "Trips that averaged under 15 km/h. Return trip_id and speed_kmh rounded to 1 decimal; decide which trips "
     "qualify before rounding.",
     [conv("duration_min", "min", "h", to="hours"), der("speed_kmh", "distance_km / hours"),
      flt("speed_kmh < 15"), der("speed_kmh", "round(speed_kmh, 1)"), sel("trip_id", "speed_kmh")])
plan("units", "students",
     "Express attendance as a fraction between 0 and 1 in a column attendance, replacing attendance_pct.",
     [conv("attendance_pct", "percent", "fraction", to="attendance"), drop("attendance_pct")])
plan("units", "employees", "Add bonus_bps, the bonus percentage expressed in basis points.",
     [conv("bonus_pct", "percent", "bps", to="bonus_bps")])

# --- text and dates ---------------------------------------------------------------

plan("text_date", "customers", "Add name_upper with the name in capital letters.",
     [der("name_upper", "upper(name)")])
plan("text_date", "customers", "Customers who joined during 2025: name and joined_on.",
     [flt("year(joined_on) == 2025"), sel("name", "joined_on")])
plan("text_date", "customers", "Add a label column that reads like 'Meera Nair (Kochi)'.",
     [der("label", "concat(name, ' (', city, ')')")])
plan("text_date", "employees", "People whose role includes the word Engineer: name and role.",
     [flt("contains(role, 'Engineer')"), sel("name", "role")])
plan("text_date", "employees", "Add tenure_days, the number of days from start_date to 2026-10-01.",
     [der("tenure_days", "days_between(start_date, '2026-10-01')")])
plan("text_date", "employees",
     "Employees who started before 2020, longest serving first. Return name and start_date.",
     [flt("start_date < '2020-01-01'"), srt("start_date"), sel("name", "start_date")], ordered=True)
plan("text_date", "requests", "Requests to any endpoint under /api/.", [flt("startswith(endpoint, '/api/')")])
plan("text_date", "stock", "Products whose name ends in 'kg': sku and product.",
     [flt("endswith(product, 'kg')"), sel("sku", "product")])
plan("text_date", "stock", "Items restocked in July 2026: sku and restocked_on.",
     [flt("year(restocked_on) == 2026 and month(restocked_on) == 7"), sel("sku", "restocked_on")])
plan("text_date", "subscriptions", "Subscriptions whose email is at kite.dev.",
     [flt("endswith(email, '@kite.dev')")])
plan("text_date", "subscriptions", "Add start_year and start_month (as numbers) taken from started_on.",
     [der("start_year", "year(started_on)"), der("start_month", "month(started_on)")])
plan("text_date", "expenses", "Expenses whose note mentions BigBasket, in any letter case.",
     [flt("contains(lower(note), 'bigbasket')")])
plan("text_date", "shipments", "Shipments shipped from 1 April to 7 April 2026 inclusive.",
     [flt("shipped_on >= '2026-04-01' and shipped_on <= '2026-04-07'")])

plan("text_date", "subscriptions",
     "Count the subscriptions per email domain, the part of the email after the @. Columns domain, subscriptions.",
     [der("domain", "split_part(email, '@', 2)"), agg(["domain"], subscriptions="count")])
plan("text_date", "trips", "How many trips started in each hour of the day? Columns hour (0 to 23), trips.",
     [der("hour", "hour(started_at)"), agg(["hour"], trips="count")])

# --- tricky: the obvious plan gives the wrong answer ---------------------------------

plan("tricky", "stock_raw",
     "Total quantity of the active items (active is 'yes'). One row, column total_qty.",
     [flt("active == 'yes'"), der("qty", "int(qty)"), agg([], total_qty=("sum", "qty"))])
plan("tricky", "stock_raw", "Add value = qty * unit_cost, rounded to 2 decimals.",
     [der("value", "round(int(qty) * float(unit_cost), 2)")])
plan("tricky", "stock_raw", "Sort by quantity, largest first, and return sku and qty with qty as a number.",
     [cast("qty", "int"), srt("-qty"), sel("sku", "qty")], ordered=True)
plan("tricky", "stock_raw", "Turn the active column into real true/false values.", [cast("active", "bool")])
plan("tricky", "readings",
     "How many readings does each device have once repeats (same device and timestamp) are counted only "
     "once? Columns device, readings.",
     [dedup("device", "taken_at"), agg(["device"], readings="count")])
plan("tricky", "readings",
     "Average humidity across all readings. Some readings are repeated (same device and timestamp) and must "
     "count only once. One row, avg_humidity rounded to 1 decimal.",
     [dedup("device", "taken_at"), agg([], avg_humidity=("mean", "humidity_pct")),
      der("avg_humidity", "round(avg_humidity, 1)")])
plan("tricky", "orders",
     "Average discount percentage across all orders, where an order with no discount recorded counts as 0. "
     "One row, avg_discount rounded to 2 decimals.",
     [der("d", "coalesce(discount_pct, 0)"), agg([], avg_discount=("mean", "d")),
      der("avg_discount", "round(avg_discount, 2)")])
plan("tricky", "orders",
     "Average discount percentage among only the orders that have a discount value recorded; a recorded 0 "
     "counts, a blank does not. One row, avg_discount rounded to 2 decimals.",
     [agg([], avg_discount=("mean", "discount_pct")), der("avg_discount", "round(avg_discount, 2)")])
plan("tricky", "tickets",
     "Count tickets per priority, treating different capitalisation and stray spaces as the same priority. "
     "Show the priority in lowercase. Columns priority, tickets.",
     [der("priority", "lower(strip(priority))"), agg(["priority"], tickets="count")])
plan("tricky", "tickets", "High priority tickets, however the priority is written, that nobody is assigned to: ticket_id only.",
     [flt("lower(strip(priority)) == 'high' and is_null(assignee)"), sel("ticket_id")])
plan("tricky", "tickets",
     "Total hours per assignee, with unassigned tickets grouped under the name 'unassigned'. Columns "
     "assignee, hours.",
     [der("assignee", "coalesce(assignee, 'unassigned')"), agg(["assignee"], hours=("sum", "hours_spent"))])
plan("tricky", "tickets",
     "Average hours per priority after normalising the priority to lowercase without stray spaces. Columns "
     "priority, avg_hours, highest first.",
     [der("priority", "lower(strip(priority))"), agg(["priority"], avg_hours=("mean", "hours_spent")),
      srt("-avg_hours")], ordered=True)
plan("tricky", "students",
     "Average math score per section where a missing score counts as 0. Columns section, avg_math rounded to "
     "1 decimal.",
     [der("m", "coalesce(math, 0)"), agg(["section"], avg_math=("mean", "m")),
      der("avg_math", "round(avg_math, 1)")])
plan("tricky", OC,
     "Number of orders per customer tier. Orders from customers who are not in the customers table go under "
     "the tier 'unknown'. Columns tier, orders.",
     [at("orders", join("customers", "customer_id", how="left")), der("tier", "coalesce(tier, 'unknown')"),
      agg(["tier"], orders="count")])
plan("tricky", "requests",
     "Success rate per endpoint, where success is a status from 200 to 299. Columns endpoint, success_pct as a "
     "percentage rounded to 1 decimal.",
     [der("ok", "1 if status >= 200 and status < 300 else 0"), agg(["endpoint"], good=("sum", "ok"), n="count"),
      der("success_pct", "round(good * 100 / n, 1)"), sel("endpoint", "success_pct")])
plan("tricky", "stock",
     "Total stock value (on_hand * unit_cost) excluding items that have no supplier. One row, column value.",
     [flt("not is_null(supplier)"), der("value", "on_hand * unit_cost"), agg([], value=("sum", "value"))])

# --- clarify: the request cannot be planned without more information --------------------

ask("orders", "Add the tax to each order.")
ask("orders", "Convert unit_price to US dollars.", or_refuse=True)
ask("orders", "Flag the large orders.")
ask("employees", "Give the top performers a raise.", or_refuse=True)
ask("employees", "Fill in the standard bonus for everyone who has no bonus percentage.")
ask("students", "List the students who failed.")
ask("shipments", "Sort the shipments by date.")
ask("tickets", "Convert hours_spent into story points.", or_refuse=True)
ask("stock", "Raise the prices by the usual markup.")
ask("subscriptions", "Apply the loyalty discount to long-standing customers.")
ask(OC, "Break the orders down by region.", or_refuse=True)
ask("trips", "Remove the outlier trips.")
ask("requests", "Bucket the latency into fast, medium and slow.")
ask("expenses", "Bade kharche alag kar do.", tags=HI)

# --- unsupported: outside what the operations can express --------------------------------

refuse("orders", "Plot monthly revenue as a bar chart.")
refuse("orders", "Forecast how many orders we will get next month.", or_ask=True)
refuse("customers", "Find and merge customers whose names look like misspellings of each other.")
refuse("expenses", "Translate the notes into Hindi.")
refuse("stock", "Email each supplier about their items that are below the reorder level.")
refuse("orders", "Look up today's USD to INR exchange rate and convert unit_price to dollars.", or_ask=True)
refuse("subscriptions", "Predict which of the active subscriptions will cancel next quarter.")
refuse("subscriptions", "Check which of these email addresses really exist.")
refuse("trips", "Train a model that predicts the fare from distance and duration.")
refuse("orders", "Is data ka channel ke hisaab se ek pie chart bana do.", tags=HI)


def main() -> None:
    lines = ["{"]
    for i, (name, rows) in enumerate(TABLES.items()):
        lines.append(f"  {json.dumps(name)}: [")
        lines.extend(
            "    " + json.dumps(row, ensure_ascii=False) + ("," if j < len(rows) - 1 else "")
            for j, row in enumerate(rows)
        )
        lines.append("  ]" + ("," if i < len(TABLES) - 1 else ""))
    lines.append("}")
    (HERE / "tables.json").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (HERE / "tasks.jsonl").write_text(
        "".join(json.dumps(task, ensure_ascii=False) + "\n" for task in TASKS), encoding="utf-8"
    )
    print(f"{len(TASKS)} tasks over {len(TABLES)} tables")
    for category, count in _counts.items():
        print(f"  {category:12} {count}")


if __name__ == "__main__":
    main()
