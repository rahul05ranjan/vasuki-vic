"""Authoring source for test set 1.

Run `uv run python evals/test1/build.py` to regenerate tables.json and tasks.jsonl.

This is a TEST set, not a development set. Its domains appear in neither evals/v1, evals/v2 nor the
training topics. Its wording may be corrected only on the evidence of strong reference models, before any
VIC model is scored on it. After that it is frozen: never change training data, prompts or the format
because of how a VIC model did here. Report the score and nothing else.
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
    "vending_machines": table(
        "machine_id location floor installed_on status capacity_slots",
        ("M1", "Lobby", 0, "2024-03-01", "active", 40),
        ("M2", "Cafeteria", 1, "2023-11-15", "active", 60),
        ("M3", "Gym", 2, "2025-01-20", "offline", 36),
        ("M4", "Library", 3, "2022-06-10", "active", 48),
        ("M5", "Parking", -1, "2025-06-05", "retired", 30),
        ("M6", "Rooftop", 5, "2026-02-01", "active", 24),
    ),
    "vends": table(
        "vend_id machine_id sold_at product category price paid_with refunded",
        (1, "M1", "2026-03-02T08:15:00", "Cola", "drink", 1.80, "card", False),
        (2, "M1", "2026-03-02T09:40:00", "Chips", "snack", 1.20, "coin", False),
        (3, "M2", "2026-03-02T12:05:00", "Water", "drink", 1.00, "app", False),
        (4, "M2", "2026-03-02T12:30:00", "Sandwich", "snack", 4.50, "card", True),
        (5, "M2", "2026-03-03T13:10:00", "Cola", "drink", 1.80, None, False),
        (6, "M3", "2026-03-03T18:45:00", "Protein Bar", "snack", 2.75, "app", False),
        (7, "M4", "2026-03-04T10:20:00", "Gum", "candy", 0.90, "coin", False),
        (8, "M4", "2026-03-04T15:55:00", "Water", "drink", 1.00, "card", False),
        (9, "M1", "2026-03-05T08:05:00", "Cola", "drink", 1.80, "app", False),
        (10, "M9", "2026-03-05T11:30:00", "Chips", "snack", 1.20, "coin", False),
        (11, "M2", "2026-03-06T12:45:00", "Juice", "drink", 2.40, "card", False),
        (12, "M4", "2026-03-06T16:10:00", "Chocolate", "candy", 1.60, "app", True),
        (13, "M1", "2026-03-07T19:25:00", "Protein Bar", "snack", 2.75, "card", False),
        (14, "M2", "2026-03-07T20:00:00", "Juice", "drink", 2.40, "coin", False),
    ),
    "elevators": table(
        "elevator_id building capacity_kg speed_mps last_inspected make",
        ("EL-1", "North Tower", 1000, 2.5, "2026-01-12", "Otis"),
        ("EL-2", "North Tower", 1600, 1.6, "2025-11-03", "Kone"),
        ("EL-3", "South Tower", 800, 1.0, None, "Otis"),
        ("EL-4", "South Tower", 1250, 4.0, "2026-02-20", "Schindler"),
        ("EL-5", "Annex", 630, 0.63, "2025-08-30", "Kone"),
    ),
    "callouts": table(
        "call_id elevator_id opened_on closed_on fault downtime_min cost technician",
        ("SC1", "EL-1", "2026-03-01", "2026-03-01", "door", 45, 120.0, "Imani"),
        ("SC2", "EL-2", "2026-03-03", "2026-03-05", "motor", 2880, 1850.0, "Pavel"),
        ("SC3", "EL-1", "2026-03-09", "2026-03-09", "sensor", 30, 95.5, "Imani"),
        ("SC4", "EL-3", "2026-03-11", None, "door", None, None, "Chen"),
        ("SC5", "EL-4", "2026-03-12", "2026-03-13", "other", 600, 410.0, "Pavel"),
        ("SC6", "EL-2", "2026-03-15", "2026-03-15", "door", 90, 150.0, "Chen"),
        ("SC7", "EL-5", "2026-03-18", "2026-03-21", "motor", 4100, 2300.0, "Pavel"),
        ("SC8", "EL-1", "2026-03-22", "2026-03-22", "door", 20, 80.0, "Imani"),
        ("SC9", "EL-4", "2026-03-25", None, "sensor", None, None, "Chen"),
        ("SC10", "EL-3", "2026-03-27", "2026-03-28", "motor", 1500, 990.0, "Pavel"),
        ("SC11", "EL-2", "2026-03-30", "2026-03-30", "sensor", 60, 110.0, "Imani"),
    ),
    "lines": table(
        "line_id owner plan region joined_on cap_gb",
        ("L100", "Aiko", "postpaid", "N", "2023-04-10", 20),
        ("L101", "Bruno", "prepaid", "S", "2025-09-01", 5),
        ("L102", "Chandra", "postpaid", "E", "2024-12-22", 50),
        ("L103", "Dmitri", "prepaid", "N", "2026-01-05", 10),
        ("L104", "Efua", "postpaid", "W", "2022-07-30", 20),
        ("L105", "Farid", "prepaid", "S", "2025-03-14", 5),
    ),
    "usage": table(
        "line_id month data_mb voice_min sms roaming",
        ("L100", "2026-01", 14200, 310, 42, False),
        ("L100", "2026-02", 21500, 280, 35, True),
        ("L101", "2026-01", 3900, 95, 120, False),
        ("L101", "2026-02", 5600, 110, 98, False),
        ("L102", "2026-01", 38800, 620, 12, False),
        ("L102", "2026-02", 51200, 590, 9, True),
        ("L103", "2026-01", 800, 40, 15, False),
        ("L103", "2026-02", 9750, None, 22, False),
        ("L104", "2026-01", 12000, 505, 60, False),
        ("L104", "2026-02", 19900, 480, 51, False),
        ("L199", "2026-02", 2530, 75, 5, False),
    ),
    "desk_bookings": table(
        "booking_id member tier desk day hours rate paid discount_pct",
        ("B01", "Noor", "flex", "D4", "2026-04-01", 3, 5.0, True, None),
        ("B02", "Otto", "day", "D1", "2026-04-01", 8, 6.5, True, 10),
        ("B03", "Noor", "flex", "D4", "2026-04-02", 4, 5.0, False, None),
        ("B04", "Pia", "fixed", "D9", "2026-04-02", 9, 4.0, True, 20),
        ("B05", "Quentin", "day", "D2", "2026-04-03", 3, 6.5, True, 0),
        ("B06", "Otto", "day", "D1", "2026-04-06", 7, 6.5, False, 10),
        ("B07", "Pia", "fixed", "D9", "2026-04-06", 10, 4.0, True, 20),
        ("B08", "Rhea", "flex", "D5", "2026-04-07", 5, 5.0, True, None),
        ("B09", "Noor", "flex", "D4", "2026-04-08", 2, 5.0, True, 5),
        ("B10", "Quentin", "day", "D3", "2026-04-08", 8, 6.5, True, 0),
        ("B11", "Pia", "fixed", "D9", "2026-04-09", 9.5, 4.0, True, 20),
        ("B12", "Rhea", "flex", "D6", "2026-04-10", 1.5, 5.0, False, None),
    ),
    "roasts": table(
        "batch_id origin process roast_level green_kg roasted_kg temp_c duration_s roasted_on cupping",
        ("R-201", "Ethiopia", "washed", " Light", 12.0, 10.3, 205, 610, "2026-05-02", 87.5),
        ("R-202", "Colombia", "washed", "medium", 15.0, 12.6, 215, 700, "2026-05-02", 84.0),
        ("R-203", "Brazil", "natural", "Dark", 20.0, 16.2, 228, 820, "2026-05-03", 80.5),
        ("R-204", "Ethiopia", "natural", "light", 12.0, 10.4, 204, 595, "2026-05-05", 88.5),
        ("R-205", "Kenya", "washed", "MEDIUM", 10.0, 8.5, 214, 690, "2026-05-06", None),
        ("R-206", "Colombia", "honey", "Medium ", 15.0, 12.5, 216, 715, "2026-05-08", 85.0),
        ("R-207", "Brazil", "natural", "dark", 20.0, 16.0, 230, 840, "2026-05-09", 78.5),
        ("R-208", "Kenya", "washed", "Light", 10.0, 8.7, 206, 620, "2026-05-11", 86.3),
        ("R-209", "Guatemala", "honey", "medium", 8.0, 6.6, 213, 680, "2026-05-12", None),
        ("R-210", "Ethiopia", "washed", "LIGHT", 12.0, 10.2, 203, 600, "2026-05-14", 89.0),
    ),
    "tank_tests": table(
        "test_id tank tested_at ph temp_f nitrate_ppm tester",
        ("t01", "Reef", "2026-06-01T07:00:00", 8.2, 78.5, 4.0, "Lara"),
        ("t02", "Reef", "2026-06-01T19:00:00", 8.1, 79.0, 5.5, "Sunil"),
        ("t03", "Reef", "2026-06-01T19:00:00", 8.1, 79.0, 5.5, "Sunil"),
        ("t04", "Amazon", "2026-06-01T07:00:00", 6.6, 82.0, 18.0, "Lara"),
        ("t05", "Amazon", "2026-06-02T07:00:00", 6.4, 83.5, None, "Lara"),
        ("t06", "Kelp", "2026-06-01T07:00:00", 7.9, 55.0, 2.0, "Sunil"),
        ("t07", "Kelp", "2026-06-02T07:00:00", 7.8, 57.5, 2.5, "Sunil"),
        ("t08", "Jelly", "2026-06-02T07:00:00", 8.0, 68.0, None, "Mo"),
        ("t09", "Jelly", "2026-06-02T19:00:00", 8.3, 70.0, 1.0, "Mo"),
        ("t10", "Amazon", "2026-06-02T19:00:00", 6.9, 84.0, 22.5, "Mo"),
    ),
    "hives": table(
        "hive_id apiary queen_year breed",
        ("H1", "Orchard", 2025, "Italian"),
        ("H2", "Orchard", 2024, "Carniolan"),
        ("H3", "Meadow", 2026, "Italian"),
        ("H4", "Meadow", 2023, "Buckfast"),
        ("H5", "Rooftop", 2025, "Carniolan"),
    ),
    "hive_checks": table(
        "inspection_id hive_id inspected_on brood_frames honey_kg mites_per_100 mood",
        ("i1", "H1", "2026-07-01", 6, 4.5, 1, "calm"),
        ("i10", "H1", "2026-07-29", 8, 8.5, 2, "calm"),
        ("i2", "H1", "2026-07-15", 7, 6.0, 2, "calm"),
        ("i3", "H2", "2026-07-01", 4, None, 5, "defensive"),
        ("i4", "H2", "2026-07-15", 5, 2.5, 7, "defensive"),
        ("i5", "H3", "2026-07-02", 8, 7.5, 0, "calm"),
        ("i6", "H3", "2026-07-16", 9, 9.0, 1, "calm"),
        ("i7", "H4", "2026-07-02", 3, 1.0, 9, "defensive"),
        ("i8", "H4", "2026-07-16", 2, None, 12, "calm"),
        ("i9", "H7", "2026-07-03", 5, 3.0, 3, "calm"),
    ),
    "entries": table(
        "team track members submitted_at demo_minutes",
        ("Quasar", "ai", 4, "2026-08-09T16:42:00", 5),
        ("Bitloom", "web", 3, "2026-08-09T17:58:00", 4),
        ("Voltaic", "hardware", 5, "2026-08-09T15:10:00", 6),
        ("Nimbus", "ai", 2, "2026-08-09T17:59:00", 3),
        ("Fernweh", "web", 4, "2026-08-09T14:30:00", 5),
        ("Octave", "hardware", 3, "2026-08-09T18:05:00", 4),
        ("Lumen", "ai", 1, "2026-08-09T12:00:00", 2),
    ),
    "scores": table(
        "team judge innovation execution polish",
        ("Quasar", "J1", 9, 7, 8),
        ("Quasar", "J2", 8, 8, 7),
        ("Bitloom", "J1", 6, 9, 9),
        ("Bitloom", "J2", 7, 8, None),
        ("Voltaic", "J1", 8, 6, 5),
        ("Voltaic", "J2", 9, 7, 6),
        ("Nimbus", "J1", 7, 5, 6),
        ("Nimbus", "J2", 6, 6, 7),
        ("Fernweh", "J1", 5, 8, 8),
        ("Fernweh", "J2", 6, 9, 9),
        ("Octave", "J1", 8, 8, 7),
        ("Octave", "J2", None, 7, 8),
        ("Zephyr", "J1", 4, 4, 4),
    ),
    "repairs": table(
        "claim_id product purchased_on claimed_on issue parts_cost labour_hours approved",
        ("W1", "Blender", "2025-02-10", "2026-01-15", "motor", "34.50", "1.5", "yes"),
        ("W2", "Kettle", "2025-11-20", "2026-02-02", "switch", "8", "0.5", "yes"),
        ("W3", "Vacuum", "2024-06-01", "2026-02-20", "battery", "59.90", "1", "no"),
        ("W4", "Blender", "2025-08-08", "2026-03-01", "blade", "12.25", "0.75", "yes"),
        ("W5", "Toaster", "2026-01-05", "2026-03-10", "element", "6.40", "0.5", "yes"),
        ("W6", "Vacuum", "2024-09-17", "2026-03-22", "motor", "88", "2", "yes"),
        ("W7", "Kettle", "2024-01-30", "2026-04-04", "lid", "3.10", "0.25", "no"),
        ("W8", "Blender", "2025-12-12", "2026-04-18", "lid", "4.75", "0.5", "yes"),
    ),
    "animals": table(
        "animal_id name species age_months weight_kg intake_on adopted_on fee neutered",
        ("A1", "Biscuit", "dog", 30, 18.5, "2026-01-04", "2026-01-20", 120.0, True),
        ("A2", "Miso", "cat", 8, 3.2, "2026-01-10", "2026-02-14", 60.0, False),
        ("A3", "Thumper", "rabbit", 14, 2.1, "2026-01-18", None, 30.0, False),
        ("A4", "Rex", "dog", 72, 31.0, "2026-02-01", None, 90.0, True),
        ("A5", "Pepper", "cat", 26, 4.6, "2026-02-09", "2026-02-11", 60.0, True),
        ("A6", "Clover", "rabbit", 5, 1.4, "2026-02-20", "2026-03-30", 30.0, False),
        ("A7", "Juno", "dog", 11, 12.0, "2026-03-02", "2026-03-09", 150.0, False),
        ("A8", "Olive", "cat", 50, 5.1, "2026-03-11", None, 45.0, True),
        ("A9", "Scout", "dog", 20, 22.4, "2026-03-15", "2026-04-01", 120.0, True),
        ("A10", "Mochi", "cat", 4, 1.1, "2026-03-28", None, 75.0, False),
    ),
    "cloud_costs": table(
        "line_id service team env month usage_hours cost_usd credits_usd",
        ("c1", "compute", "search", "prod", "2026-07", 720, 1840.0, 200.0),
        ("c2", "storage", "search", "prod", "2026-07", 720, 310.5, None),
        ("c3", "compute", "search", "staging", "2026-07", 300, 420.0, None),
        ("c4", "db", "payments", "prod", "2026-07", 720, 2650.0, 500.0),
        ("c5", "network", "payments", "prod", "2026-07", 720, 180.25, None),
        ("c6", "compute", "payments", "dev", "2026-07", 150, 95.0, 95.0),
        ("c7", "compute", "search", "prod", "2026-08", 744, 1975.0, 200.0),
        ("c8", "storage", "search", "prod", "2026-08", 744, 322.0, None),
        ("c9", "db", "payments", "prod", "2026-08", 744, 2810.0, 500.0),
        ("c10", "compute", "ml", "dev", "2026-08", 410, 3120.0, 1000.0),
        ("c11", "storage", "ml", "dev", "2026-08", 744, 88.4, None),
        ("c12", "network", "payments", "prod", "2026-08", 744, 201.75, None),
    ),
    "found_items": table(
        "item_id kind found_on found_at claimed_on est_value colour",
        ("F1", "umbrella", "2026-09-01", "Platform 2", "2026-09-03", 15.0, "black"),
        ("F2", "wallet", "2026-09-01", "Ticket Hall", "2026-09-01", 80.0, "brown"),
        ("F3", "phone", "2026-09-02", "Platform 1", None, 450.0, "silver"),
        ("F4", "scarf", "2026-09-04", "Cafe", None, None, "red"),
        ("F5", "laptop", "2026-09-05", "Platform 2", "2026-09-12", 900.0, "grey"),
        ("F6", "umbrella", "2026-09-07", "Cafe", None, 10.0, "blue"),
        ("F7", "keys", "2026-09-08", "Ticket Hall", "2026-09-08", None, "silver"),
        ("F8", "backpack", "2026-09-10", "Platform 1", "2026-09-18", 60.0, "green"),
        ("F9", "phone", "2026-09-11", "Ticket Hall", None, 300.0, "black"),
        ("F10", "glasses", "2026-09-14", "Platform 2", None, 120.0, "brown"),
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


def dedup(*keys, keep=None):
    step = {"op": "deduplicate", **({"keys": list(keys)} if keys else {})}
    return {**step, "keep": keep} if keep else step


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
MV = ["vends", "vending_machines"]
ES = ["callouts", "elevators"]
LU = ["usage", "lines"]
HV = ["hive_checks", "hives"]
SS = ["scores", "entries"]

# --- vending machines ---------------------------------------------------------

plan("basic", "vends", "Sales that were refunded.", [flt("refunded")])
plan("basic", "vends", "Everything bought with a card, newest first.",
     [flt("paid_with == 'card'"), srt("-sold_at")], ordered=True)
plan("basic", "vends", "Drinks that cost more than 1.50.", [flt("category == 'drink' and price > 1.5")])
plan("basic", "vends", "Show the sales that have no payment method recorded.", [flt("is_null(paid_with)")])
plan("basic", "vends", "One row per product that has been sold, with just the product column.",
     [sel("product"), dedup()])
plan("basic", "vends", "App se pay ki gayi sales dikhao, sirf product aur price.",
     [flt("paid_with == 'app'"), sel("product", "price")], tags=HI)
plan("derive", "vends", "Add a column premium that is true when the price is 2 or more.",
     [der("premium", "price >= 2")])
plan("derive", "vends", "Add price_with_tax, the price plus 8% tax, to 2 decimals.",
     [der("price_with_tax", "round(price * 1.08, 2)")])
plan("aggregate", "vends", "How many sales did each machine make, refunded ones included? Columns machine_id, sales.",
     [agg(["machine_id"], sales="count")])
plan("aggregate", "vends",
     "Sum of price per product category over every sale, refunded or not. Columns category, takings.",
     [agg(["category"], takings=("sum", "price"))])
plan("aggregate", "vends", "How many different products has each machine sold? Columns machine_id, products.",
     [agg(["machine_id"], products=("count_distinct", "product"))])
plan("multi", "vends",
     "Takings per machine, not counting refunded sales, highest first. Columns machine_id, takings.",
     [flt("not refunded"), agg(["machine_id"], takings=("sum", "price")), srt("-takings")], ordered=True)
plan("multi", "vends",
     "What share of each machine's sales were drinks? Columns machine_id, drink_share, a fraction to 2 decimals.",
     [der("d", "1 if category == 'drink' else 0"), agg(["machine_id"], drinks=("sum", "d"), n="count"),
      der("drink_share", "round(drinks / n, 2)"), sel("machine_id", "drink_share")])
plan("text_date", "vends",
     "How many sales, refunded ones included, happened in each hour of the day? Columns hour, sales.",
     [der("hour", "hour(sold_at)"), agg(["hour"], sales="count")])
plan("join", MV,
     "Every sale with the location of the machine it came from. A sale from a machine we have no record of "
     "stays in, with its location left null. Return vend_id, product, location.",
     [at("vends", join("vending_machines", "machine_id", how="left")), sel("vend_id", "product", "location")])
plan("join", MV, "Which machines have never sold anything? Return machine_id and location.",
     [named("sold", at("vends", agg(["machine_id"], n="count"))),
      at("vending_machines", join("sold", "machine_id", how="left")), flt("is_null(n)"), sel("machine_id", "location")])
plan("join", MV,
     "Sum of price by location over every sale, refunded or not, for machines that are currently active. "
     "Columns location, takings.",
     [at("vends", join("vending_machines", "machine_id")), flt("status == 'active'"),
      agg(["location"], takings=("sum", "price"))])
plan("join", MV, "Sales that came from a machine we have no record of: vend_id and machine_id.",
     [at("vends", join("vending_machines", "machine_id", how="left")), flt("is_null(location)"),
      sel("vend_id", "machine_id")])
ask("vends", "Raise every price by the new surcharge.")
refuse("vending_machines", "Text me whenever a machine goes offline.")

# --- elevators ----------------------------------------------------------------

plan("basic", "callouts", "Service calls that are still open.", [flt("is_null(closed_on)")])
plan("basic", "callouts", "Door faults that Imani handled.", [flt("fault == 'door' and technician == 'Imani'")])
plan("basic", "callouts",
     "The three most expensive service calls, dearest first. Show call_id, elevator_id and cost.",
     [srt("-cost"), lim(3), sel("call_id", "elevator_id", "cost")], ordered=True)
plan("basic", "elevators", "Elevators that can carry at least 1000 kg, fastest first.",
     [flt("capacity_kg >= 1000"), srt("-speed_mps")], ordered=True)
plan("aggregate", "callouts", "Total repair cost per elevator, as total_cost.",
     [agg(["elevator_id"], total_cost=("sum", "cost"))])
plan("aggregate", "callouts",
     "For each kind of fault: the number of calls and the average downtime in minutes. Columns fault, calls, "
     "avg_downtime.",
     [agg(["fault"], calls="count", avg_downtime=("mean", "downtime_min"))])
plan("aggregate", "callouts", "Har technician ne kitni calls handle ki? Columns technician, calls.",
     [agg(["technician"], calls="count")], tags=HI)
plan("multi", "callouts", "Total downtime per technician, as total_downtime, biggest first.",
     [agg(["technician"], total_downtime=("sum", "downtime_min")), srt("-total_downtime")], ordered=True)
plan("multi", "callouts",
     "How many days did each closed call take from opening to closing? Return call_id and days_open, longest "
     "first, ties by call_id.",
     [flt("not is_null(closed_on)"), der("days_open", "days_between(opened_on, closed_on)"),
      sel("call_id", "days_open"), srt("-days_open", "call_id")], ordered=True)
plan("multi", "callouts", "Elevators with more than two service calls: elevator_id and calls.",
     [agg(["elevator_id"], calls="count"), flt("calls > 2")])
plan("tricky", "callouts",
     "Average cost of a service call, counting a call with no cost yet as 0. One row, avg_cost to 2 decimals.",
     [der("c", "coalesce(cost, 0)"), agg([], avg_cost=("mean", "c")), der("avg_cost", "round(avg_cost, 2)")])
plan("join", ES, "Each service call with the building it was in: call_id, building, fault.",
     [at("callouts", join("elevators", "elevator_id")), sel("call_id", "building", "fault")])
plan("join", ES, "Repair spend per building, as spend, highest first.",
     [at("callouts", join("elevators", "elevator_id")), agg(["building"], spend=("sum", "cost")),
      srt("-spend")], ordered=True)
plan("join", ES, "Open calls on elevators that have never been inspected: call_id and elevator_id.",
     [at("callouts", join("elevators", "elevator_id")),
      flt("is_null(closed_on) and is_null(last_inspected)"), sel("call_id", "elevator_id")])
plan("units", "callouts", "Add downtime_h, the downtime in hours, to 1 decimal.",
     [conv("downtime_min", "min", "h", to="downtime_h"), der("downtime_h", "round(downtime_h, 1)")])
plan("units", "elevators", "Add capacity_lb, the capacity in pounds, rounded to the nearest whole number.",
     [conv("capacity_kg", "kg", "lb", to="capacity_lb"), der("capacity_lb", "round(capacity_lb)")])
plan("units", "elevators", "The speed of each elevator in km/h: elevator_id and speed_kmh.",
     [der("speed_kmh", "speed_mps * 3.6"), sel("elevator_id", "speed_kmh")])
ask("callouts", "Flag the calls that took too long.")
refuse("callouts", "Predict which elevator will break down next.", or_ask=True)

# --- phone lines --------------------------------------------------------------

plan("basic", "lines", "Prepaid lines.", [flt("plan == 'prepaid'")])
plan("basic", "lines", "Lines in the north or west region.", [flt("region in ['N', 'W']")])
plan("basic", "usage", "Usage records where the line was roaming.", [flt("roaming")])
plan("derive", "usage", "Add heavy, true when a record shows more than 20000 MB of data.",
     [der("heavy", "data_mb > 20000")])
plan("aggregate", "usage", "Total SMS sent in each month, as total_sms.", [agg(["month"], total_sms=("sum", "sms"))])
plan("units", "usage", "Add data_gb, the data used in gigabytes (1 GB = 1000 MB).",
     [conv("data_mb", "MB", "GB", to="data_gb")])
plan("units", "usage", "Add voice_h, the voice time in hours, to 2 decimals.",
     [conv("voice_min", "min", "h", to="voice_h"), der("voice_h", "round(voice_h, 2)")])
plan("multi", "usage",
     "Total data used per line across both months, in GB (1 GB = 1000 MB). Columns line_id, total_gb, heaviest "
     "user first.",
     [agg(["line_id"], mb=("sum", "data_mb")), conv("mb", "MB", "GB", to="total_gb"),
      sel("line_id", "total_gb"), srt("-total_gb")], ordered=True)
plan("multi", "usage", "Total data per month in GB (1 GB = 1000 MB), to 1 decimal. Columns month, total_gb.",
     [agg(["month"], mb=("sum", "data_mb")), conv("mb", "MB", "GB", to="total_gb"),
      der("total_gb", "round(total_gb, 1)"), sel("month", "total_gb")])
plan("multi", "usage",
     "Roaming wale records ko chhod kar, har line ke total voice minutes batao. Columns line_id, total_voice.",
     [flt("not roaming"), agg(["line_id"], total_voice=("sum", "voice_min"))], tags=HI)
plan("text_date", "lines", "Lines that joined during 2025: line_id, owner, joined_on.",
     [flt("year(joined_on) == 2025"), sel("line_id", "owner", "joined_on")])
plan("text_date", "lines", "How many days had each line been with us on 1 March 2026? Add days_with_us.",
     [der("days_with_us", "days_between(joined_on, '2026-03-01')")])
plan("join", LU,
     "Who went over their monthly data cap, and in which month? Return owner, month and data_gb, the data used "
     "in GB (1 GB = 1000 MB).",
     [at("usage", join("lines", "line_id")), conv("data_mb", "MB", "GB", to="data_gb"),
      flt("data_gb > cap_gb"), sel("owner", "month", "data_gb")])
plan("join", LU, "Lines with no usage recorded at all: line_id and owner.",
     [named("used", at("usage", agg(["line_id"], n="count"))), at("lines", join("used", "line_id", how="left")),
      flt("is_null(n)"), sel("line_id", "owner")])
plan("join", LU,
     "Average voice minutes per plan type, skipping records with no voice figure and lines we have no record "
     "of. Columns plan, avg_voice.",
     [at("usage", join("lines", "line_id")), agg(["plan"], avg_voice=("mean", "voice_min"))])
plan("join", LU, "Total data per region in GB (1 GB = 1000 MB), for lines we have on record. Columns region, total_gb.",
     [at("usage", join("lines", "line_id")), agg(["region"], mb=("sum", "data_mb")),
      conv("mb", "MB", "GB", to="total_gb"), sel("region", "total_gb")])
ask("usage", "Upgrade the heavy users to the next plan.", or_refuse=True)

# --- coworking desk bookings --------------------------------------------------------

plan("basic", "desk_bookings", "Bookings that have not been paid for.", [flt("not paid")])
plan("basic", "desk_bookings", "Full-day bookings: 8 hours or more.", [flt("hours >= 8")])
plan("basic", "desk_bookings",
     "Just the member, day and hours of each booking, in date order, with the longest booking first within a "
     "day.",
     [sel("member", "day", "hours"), srt("day", "-hours")], ordered=True)
plan("basic", "desk_bookings", "Noor ki saari bookings dikhao, sabse lambi pehle.",
     [flt("member == 'Noor'"), srt("-hours")], ordered=True, tags=HI)
plan("derive", "desk_bookings", "Add amount = hours times rate.", [der("amount", "hours * rate")])
plan("derive", "desk_bookings",
     "Add amount_due: hours times rate, less the discount percentage. No discount recorded means no discount. "
     "2 decimals.",
     [der("amount_due", "round(hours * rate * (1 - coalesce(discount_pct, 0) / 100), 2)")])
plan("aggregate", "desk_bookings", "Hours booked per member, as total_hours.",
     [agg(["member"], total_hours=("sum", "hours"))])
plan("aggregate", "desk_bookings", "Bookings and total hours for each tier. Columns tier, bookings, total_hours.",
     [agg(["tier"], bookings="count", total_hours=("sum", "hours"))])
plan("multi", "desk_bookings",
     "Unpaid money per member: hours times rate, summed over that member's unpaid bookings. Columns member, "
     "unpaid.",
     [flt("not paid"), der("amount", "hours * rate"), agg(["member"], unpaid=("sum", "amount"))])
plan("multi", "desk_bookings", "Days with more than one booking, and how many: day, bookings.",
     [agg(["day"], bookings="count"), flt("bookings > 1")])
plan("multi", "desk_bookings",
     "Top 2 members by money billed, where billed is hours times rate before any discount. Columns member, "
     "billed, biggest first.",
     [der("amount", "hours * rate"), agg(["member"], billed=("sum", "amount")), srt("-billed"), lim(2)],
     ordered=True)
plan("tricky", "desk_bookings",
     "Average discount across all bookings, treating a booking with no discount recorded as 0%. One row, "
     "avg_discount to 1 decimal.",
     [der("d", "coalesce(discount_pct, 0)"), agg([], avg_discount=("mean", "d")),
      der("avg_discount", "round(avg_discount, 1)")])
plan("text_date", "desk_bookings", "Bookings from 6 April to 10 April 2026, both days included.",
     [flt("day >= '2026-04-06' and day <= '2026-04-10'")])
ask("desk_bookings", "Apply the member discount to Rhea's bookings.")
refuse("desk_bookings", "Email an invoice to everyone with an unpaid booking.")

# --- coffee roasting ------------------------------------------------------------------

plan("basic", "roasts", "Natural-process batches.", [flt("process == 'natural'")])
plan("basic", "roasts", "Batches with no cupping score yet.", [flt("is_null(cupping)")])
plan("derive", "roasts",
     "Add loss_pct: the weight lost in roasting as a percentage of the green weight, to 1 decimal.",
     [der("loss_pct", "round((green_kg - roasted_kg) / green_kg * 100, 1)")])
plan("aggregate", "roasts", "Roasted kilograms per origin, as roasted_total.",
     [agg(["origin"], roasted_total=("sum", "roasted_kg"))])
plan("tricky", "roasts",
     "How many batches at each roast level? The level is typed inconsistently: treat spellings that differ "
     "only in capitals or stray spaces as one level, written in lowercase. Columns roast_level, batches.",
     [der("roast_level", "lower(strip(roast_level))"), agg(["roast_level"], batches="count")])
plan("tricky", "roasts",
     "Average cupping score of the light roasts, however 'light' is typed. One row, avg_score to 1 decimal.",
     [flt("lower(strip(roast_level)) == 'light'"), agg([], avg_score=("mean", "cupping")),
      der("avg_score", "round(avg_score, 1)")])
plan("tricky", "roasts", "The medium roasts, whatever the capitalisation or spacing: batch_id and origin.",
     [flt("lower(strip(roast_level)) == 'medium'"), sel("batch_id", "origin")])
plan("units", "roasts", "The roast temperature of each batch in Fahrenheit: batch_id and temp_f.",
     [conv("temp_c", "C", "F", to="temp_f"), sel("batch_id", "temp_f")])
plan("units", "roasts", "Add duration_min, the roast time in minutes, to 1 decimal.",
     [conv("duration_s", "s", "min", to="duration_min"), der("duration_min", "round(duration_min, 1)")])
plan("multi", "roasts",
     "Which origin scores best? Average cupping score per origin over the batches that have a score, to 1 "
     "decimal, best first. An origin with no scored batch still appears, with a null average. Columns "
     "origin, avg_score.",
     [agg(["origin"], avg_score=("mean", "cupping")), der("avg_score", "round(avg_score, 1)"),
      srt("-avg_score")], ordered=True)
plan("multi", "roasts",
     "Yield per process: total roasted weight divided by total green weight. Columns process, yield to 3 "
     "decimals.",
     [agg(["process"], roasted=("sum", "roasted_kg"), green=("sum", "green_kg")),
      der("yield", "round(roasted / green, 3)"), sel("process", "yield")])
plan("multi", "roasts",
     "Colombia ke batches me sabse zyada cupping score wala batch kaun sa hai? batch_id aur cupping.",
     [flt("origin == 'Colombia'"), srt("-cupping"), lim(1), sel("batch_id", "cupping")], ordered=True, tags=HI)
plan("text_date", "roasts",
     "Batches roasted from 8 May to 14 May 2026, both days included: batch_id and roasted_on.",
     [flt("roasted_on >= '2026-05-08' and roasted_on <= '2026-05-14'"), sel("batch_id", "roasted_on")])
ask("roasts", "Mark the batches that were roasted too hot.")
refuse("roasts", "Describe the flavour notes of each batch.", or_ask=True)

# --- aquarium water tests --------------------------------------------------------------

plan("basic", "tank_tests", "Tests done by Lara.", [flt("tester == 'Lara'")])
plan("basic", "tank_tests", "Readings where the pH was below 7.", [flt("ph < 7")])
plan("derive", "tank_tests", "Add in_range, true when the pH is between 7.8 and 8.4 inclusive.",
     [der("in_range", "ph >= 7.8 and ph <= 8.4")])
plan("aggregate", "tank_tests", "The highest and lowest pH seen in each tank: tank, max_ph, min_ph.",
     [agg(["tank"], max_ph=("max", "ph"), min_ph=("min", "ph"))])
plan("tricky", "tank_tests",
     "How many separate tests does each tank have? One test was entered twice, with the same tank and time, "
     "and must count once. Columns tank, tests.",
     [dedup("tank", "tested_at"), agg(["tank"], tests="count")])
plan("units", "tank_tests", "Add temp_c, the water temperature in Celsius, to 1 decimal.",
     [conv("temp_f", "F", "C", to="temp_c"), der("temp_c", "round(temp_c, 1)")])
plan("multi", "tank_tests",
     "Average nitrate per tank, ignoring tests with no nitrate reading and counting the test that was entered "
     "twice (same tank and time) only once. Columns tank, avg_nitrate, highest first.",
     [dedup("tank", "tested_at"), agg(["tank"], avg_nitrate=("mean", "nitrate_ppm")), srt("-avg_nitrate")],
     ordered=True)
plan("multi", "tank_tests",
     "Per tester: the number of tests they logged and the warmest water they recorded. Columns tester, tests, "
     "warmest.",
     [agg(["tester"], tests="count", warmest=("max", "temp_f"))])
plan("text_date", "tank_tests", "Tests taken in the evening, at 18:00 or later: test_id, tank, tested_at.",
     [flt("hour(tested_at) >= 18"), sel("test_id", "tank", "tested_at")])
ask("tank_tests", "Which tanks are unhealthy?")
refuse("tank_tests", "Plot the pH of each tank over time.")
refuse("tank_tests", "Order replacement filters for the tanks with nitrate above 10 ppm.")

# --- beehives ------------------------------------------------------------------------------

plan("basic", "hive_checks", "Inspections where the bees were defensive.", [flt("mood == 'defensive'")])
plan("basic", "hive_checks", "Inspections with a mite count of 5 or more, worst first.",
     [flt("mites_per_100 >= 5"), srt("-mites_per_100")], ordered=True)
plan("basic", "hive_checks", "Remove the mood column and rename inspected_on to date.",
     [drop("mood"), ren(inspected_on="date")])
plan("derive", "hive_checks",
     "Add risk: 'high' when mites_per_100 is 9 or more, 'watch' from 3 to 8, otherwise 'low'.",
     [der("risk", "'high' if mites_per_100 >= 9 else ('watch' if mites_per_100 >= 3 else 'low')")])
plan("aggregate", "hive_checks",
     "Honey harvested per hive, as honey_total. An inspection with no honey figure adds nothing.",
     [agg(["hive_id"], honey_total=("sum", "honey_kg"))])
plan("multi", "hive_checks",
     "The most recent inspection of each hive: hive_id, inspected_on and brood_frames, ordered by hive_id.",
     [srt("inspected_on"), dedup("hive_id", keep="last"), sel("hive_id", "inspected_on", "brood_frames"),
      srt("hive_id")], ordered=True)
plan("multi", "hive_checks",
     "Shaant (calm) mood wale inspections me average honey kitna tha? Ek row, column avg_honey.",
     [flt("mood == 'calm'"), agg([], avg_honey=("mean", "honey_kg"))], tags=HI)
plan("text_date", "hive_checks",
     "Inspections done on or after 15 July 2026: inspection_id and inspected_on.",
     [flt("inspected_on >= '2026-07-15'"), sel("inspection_id", "inspected_on")])
plan("join", HV, "Honey per apiary, as honey_total, counting only the hives we have on record.",
     [at("hive_checks", join("hives", "hive_id")), agg(["apiary"], honey_total=("sum", "honey_kg"))])
plan("join", HV, "Hives that have not been inspected: hive_id and apiary.",
     [named("seen", at("hive_checks", agg(["hive_id"], n="count"))), at("hives", join("seen", "hive_id", how="left")),
      flt("is_null(n)"), sel("hive_id", "apiary")])
plan("join", HV,
     "Inspections of hives whose queen is from 2024 or earlier: inspection_id, hive_id, queen_year.",
     [at("hive_checks", join("hives", "hive_id")), flt("queen_year <= 2024"),
      sel("inspection_id", "hive_id", "queen_year")])
plan("join", HV,
     "Add the apiary to every inspection. An inspection of a hive we have no record of stays in, with its apiary "
     "left null. Return inspection_id, hive_id, apiary.",
     [at("hive_checks", join("hives", "hive_id", how="left")), sel("inspection_id", "hive_id", "apiary")])
plan("join", HV,
     "Average mite count per breed, for hives we have on record. Columns breed, avg_mites, worst first.",
     [at("hive_checks", join("hives", "hive_id")), agg(["breed"], avg_mites=("mean", "mites_per_100")),
      srt("-avg_mites")], ordered=True)
ask("hive_checks", "Treat the infested hives.", or_refuse=True)

# --- hackathon -------------------------------------------------------------------------------

plan("basic", "entries", "The entries in the AI track, with all their columns.", [flt("track == 'ai'")])
plan("basic", "entries", "The entries from teams with only one member, with all their columns.", [flt("members == 1")])
plan("basic", "entries", "Web track ki teams ke naam aur members batao, zyada members wali pehle.",
     [flt("track == 'web'"), sel("team", "members"), srt("-members")], ordered=True, tags=HI)
plan("derive", "scores", "Add total = innovation + execution + polish, counting a missing mark as 0.",
     [der("total", "coalesce(innovation, 0) + coalesce(execution, 0) + coalesce(polish, 0)")])
plan("aggregate", "scores", "Average innovation mark per team, as avg_innovation.",
     [agg(["team"], avg_innovation=("mean", "innovation"))])
plan("aggregate", "entries", "Number of teams and total members per track. Columns track, teams, people.",
     [agg(["track"], teams="count", people=("sum", "members"))])
plan("multi", "scores",
     "How generous is each judge? The average of every mark they gave, across all three criteria, skipping "
     "missing marks. Columns judge, avg_mark to 1 decimal.",
     [der("s", "coalesce(innovation, 0) + coalesce(execution, 0) + coalesce(polish, 0)"),
      der("n", "(0 if is_null(innovation) else 1) + (0 if is_null(execution) else 1) + (0 if is_null(polish) else 1)"),
      agg(["judge"], s=("sum", "s"), n=("sum", "n")), der("avg_mark", "round(s / n, 1)"),
      sel("judge", "avg_mark")])
plan("text_date", "entries", "Submissions that came in after the 18:00 deadline: team and submitted_at.",
     [flt("submitted_at > '2026-08-09T18:00:00'"), sel("team", "submitted_at")])
plan("text_date", "entries", "Add a label that reads like 'Quasar [ai]', from the team and its track.",
     [der("label", "concat(team, ' [', track, ']')")])
plan("join", SS,
     "Final ranking: each team's total marks across both judges and all three criteria, counting a missing "
     "mark as 0. Only teams that submitted, and teams with no marks are left out. Columns team, track, total. "
     "Best first.",
     [at("scores", der("t", "coalesce(innovation, 0) + coalesce(execution, 0) + coalesce(polish, 0)")),
      agg(["team"], total=("sum", "t")), join("entries", "team"), sel("team", "track", "total"),
      srt("-total")], ordered=True)
plan("join", SS, "Which teams submitted but have no marks at all? Return team and track.",
     [named("marked", at("scores", agg(["team"], n="count"))), at("entries", join("marked", "team", how="left")),
      flt("is_null(n)"), sel("team", "track")])
plan("join", SS, "Marks given to a team that never submitted: team and judge.",
     [at("scores", join("entries", "team", how="left")), flt("is_null(track)"), sel("team", "judge")])
refuse("entries", "Pick the winner based on which project sounds most original.", or_ask=True)

# --- warranty repairs: numbers stored as text --------------------------------------------------

plan("tricky", "repairs", "Total parts cost of the approved claims. One row, column parts_total.",
     [flt("approved == 'yes'"), der("p", "float(parts_cost)"), agg([], parts_total=("sum", "p"))])
plan("tricky", "repairs",
     "Add total_cost = parts_cost plus labour_hours times 40, the hourly rate. 2 decimals.",
     [der("total_cost", "round(float(parts_cost) + float(labour_hours) * 40, 2)")])
plan("tricky", "repairs",
     "Claims by parts cost, most expensive first: claim_id and parts_cost, with parts_cost as a number.",
     [cast("parts_cost", "float"), srt("-parts_cost"), sel("claim_id", "parts_cost")], ordered=True)
plan("tricky", "repairs", "Turn the approved column into real true/false values.", [cast("approved", "bool")])
plan("aggregate", "repairs", "Number of claims per product, as claims.", [agg(["product"], claims="count")])
plan("text_date", "repairs", "How many days after purchase was each claim made? Add days_owned.",
     [der("days_owned", "days_between(purchased_on, claimed_on)")])
plan("multi", "repairs",
     "Approved claims that were made within 365 days of purchase: claim_id and product.",
     [flt("approved == 'yes' and days_between(purchased_on, claimed_on) <= 365"), sel("claim_id", "product")])
ask("repairs", "Reject the claims that are out of warranty.")

# --- animal shelter ----------------------------------------------------------------------------

plan("basic", "animals", "Animals still waiting for a home.", [flt("is_null(adopted_on)")])
plan("basic", "animals", "Dogs under two years old.", [flt("species == 'dog' and age_months < 24")])
plan("basic", "animals", "The cats, youngest first: name and age_months.",
     [flt("species == 'cat'"), srt("age_months"), sel("name", "age_months")], ordered=True)
plan("basic", "animals",
     "Jo janwar abhi tak adopt nahi hue, unka naam, species aur fee batao, sabse mehenga pehle.",
     [flt("is_null(adopted_on)"), sel("name", "species", "fee"), srt("-fee")], ordered=True, tags=HI)
plan("derive", "animals", "Add age_years, the age in years, to 1 decimal.",
     [der("age_years", "round(age_months / 12, 1)")])
plan("derive", "animals", "Add ready, true when the animal is neutered and at least 6 months old.",
     [der("ready", "neutered and age_months >= 6")])
plan("derive", "animals",
     "Add size: 'small' under 5 kg, 'medium' from 5 kg up to but not including 20 kg, 'large' at 20 kg and over.",
     [der("size", "'small' if weight_kg < 5 else ('medium' if weight_kg < 20 else 'large')")])
plan("aggregate", "animals",
     "Per species: how many animals, and their average weight. Columns species, animals, avg_weight.",
     [agg(["species"], animals="count", avg_weight=("mean", "weight_kg"))])
plan("units", "animals", "Add weight_lb, the weight in pounds, to 1 decimal.",
     [conv("weight_kg", "kg", "lb", to="weight_lb"), der("weight_lb", "round(weight_lb, 1)")])
plan("text_date", "animals",
     "How long did each adopted animal wait? days_in_care runs from intake to adoption. Return name and "
     "days_in_care, longest wait first.",
     [flt("not is_null(adopted_on)"), der("days_in_care", "days_between(intake_on, adopted_on)"),
      sel("name", "days_in_care"), srt("-days_in_care")], ordered=True)
plan("multi", "animals",
     "Adoption rate per species: the share of its animals already adopted. Columns species, adoption_rate, a "
     "fraction to 2 decimals.",
     [der("a", "0 if is_null(adopted_on) else 1"), agg(["species"], adopted=("sum", "a"), n="count"),
      der("adoption_rate", "round(adopted / n, 2)"), sel("species", "adoption_rate")])
plan("multi", "animals",
     "Fees collected so far: the total fee of adopted animals per species. Columns species, collected, smallest "
     "first.",
     [flt("not is_null(adopted_on)"), agg(["species"], collected=("sum", "fee")), srt("collected")],
     ordered=True)
ask("animals", "Discount the fee for animals that have been here too long.")
refuse("animals", "Print an adoption certificate as a PDF for each adopted animal.")

# --- cloud bills -------------------------------------------------------------------------------

plan("basic", "cloud_costs", "Production line items.", [flt("env == 'prod'")])
plan("basic", "cloud_costs", "Line items over 1000 dollars, biggest first: line_id, service, cost_usd.",
     [flt("cost_usd > 1000"), srt("-cost_usd"), sel("line_id", "service", "cost_usd")], ordered=True)
plan("derive", "cloud_costs",
     "Add net_cost = cost minus credits, where no credits recorded means zero credits.",
     [der("net_cost", "cost_usd - coalesce(credits_usd, 0)")])
plan("aggregate", "cloud_costs", "Cost per team before credits, as spend.", [agg(["team"], spend=("sum", "cost_usd"))])
plan("aggregate", "cloud_costs", "Cost before credits for each service in each month. Columns service, month, spend.",
     [agg(["service", "month"], spend=("sum", "cost_usd"))])
plan("aggregate", "cloud_costs",
     "The smallest and the largest cost of any single line item, before credits. One row with the two amounts, "
     "cheapest and dearest.",
     [agg([], cheapest=("min", "cost_usd"), dearest=("max", "cost_usd"))])
plan("multi", "cloud_costs",
     "Net spend per team after credits, where no credits recorded means none. Columns team, net_spend, highest "
     "first.",
     [der("net", "cost_usd - coalesce(credits_usd, 0)"), agg(["team"], net_spend=("sum", "net")),
      srt("-net_spend")], ordered=True)
plan("multi", "cloud_costs",
     "Which teams had a higher cost before credits in August 2026 than in July 2026? A team with nothing billed "
     "in a month counts as 0 for that month. Return team, july and august, the two monthly totals.",
     [der("july", "cost_usd if month == '2026-07' else 0"), der("august", "cost_usd if month == '2026-08' else 0"),
      agg(["team"], july=("sum", "july"), august=("sum", "august")), flt("august > july")])
plan("multi", "cloud_costs",
     "Hourly cost of the compute line items: cost_usd divided by usage hours, ignoring credits, to 2 decimals, "
     "most expensive first. Columns line_id, team, hourly.",
     [flt("service == 'compute'"), der("hourly", "round(cost_usd / usage_hours, 2)"),
      sel("line_id", "team", "hourly"), srt("-hourly")], ordered=True)
plan("multi", "cloud_costs",
     "Each environment's share of total cost before credits, as a percentage to 1 decimal. Columns env, pct.",
     [at("cloud_costs", der("k", "1")), named("by_env", agg(["k", "env"], spend=("sum", "cost_usd"))),
      at("cloud_costs", agg([], total=("sum", "cost_usd"))), named("overall", der("k", "1")),
      at("by_env", join("overall", "k")), der("pct", "round(spend * 100 / total, 1)"), sel("env", "pct")])
plan("multi", "cloud_costs",
     "Payments team ka prod environment me total cost (credits ghataye bina) kitna hai? Ek row, column total.",
     [flt("team == 'payments' and env == 'prod'"), agg([], total=("sum", "cost_usd"))], tags=HI)
plan("text_date", "cloud_costs", "Line items for August 2026.", [flt("month == '2026-08'")])
ask("cloud_costs", "Charge each team its fair share of the shared costs.", or_refuse=True)
refuse("cloud_costs", "Forecast September's bill.", or_ask=True)

# --- lost and found ------------------------------------------------------------------------------

plan("basic", "found_items", "Items nobody has claimed yet.", [flt("is_null(claimed_on)")])
plan("basic", "found_items", "Phones and laptops.", [flt("kind in ['phone', 'laptop']")])
plan("basic", "found_items",
     "Things found at the cafe or the ticket hall that have no estimated value recorded.",
     [flt("found_at in ['Cafe', 'Ticket Hall'] and is_null(est_value)")])
plan("basic", "found_items", "Silver ya black colour ki cheezein dikhao: item_id, kind, colour.",
     [flt("colour in ['silver', 'black']"), sel("item_id", "kind", "colour")], tags=HI)
plan("derive", "found_items",
     "Add status: 'claimed' if the item has been collected, otherwise 'waiting'.",
     [der("status", "'waiting' if is_null(claimed_on) else 'claimed'")])
plan("aggregate", "found_items", "How many items were found at each place? Columns found_at, items.",
     [agg(["found_at"], items="count")])
plan("text_date", "found_items",
     "How many days did each claimed item sit before it was collected? Return item_id, kind and days_held.",
     [flt("not is_null(claimed_on)"), der("days_held", "days_between(found_on, claimed_on)"),
      sel("item_id", "kind", "days_held")])
plan("text_date", "found_items", "Add a tag that reads like 'umbrella (black)', from the kind and the colour.",
     [der("tag", "concat(kind, ' (', colour, ')')")])
plan("multi", "found_items",
     "Total estimated value of the unclaimed items per kind, as value. An item with no estimate adds nothing.",
     [flt("is_null(claimed_on)"), agg(["kind"], value=("sum", "est_value"))])
plan("multi", "found_items",
     "Unclaimed items found more than 20 days before 30 September 2026: item_id, kind, found_on.",
     [flt("is_null(claimed_on) and days_between(found_on, '2026-09-30') > 20"),
      sel("item_id", "kind", "found_on")])
ask("found_items", "Throw away the old unclaimed stuff.", or_refuse=True)
refuse("found_items", "Find the owner of each unclaimed phone.", or_ask=True)


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
