"""Prompts for inventing training worlds and authoring examples on them, and the recipes that steer both."""

from __future__ import annotations

import json
import random

from vasuki import describe

from .topics import BANNED_TABLES, TOPICS, UNSUPPORTED_ACTIONS

TRAITS = {
    "two_tables": (0.40, "Make two tables that share one key column with the same name. Most keys match; one or "
                         "two rows on each side have no match in the other table."),
    "units": (0.35, "Include one or two measurement columns with the unit in the name, such as weight_kg, "
                    "distance_mi, duration_min, temp_c, size_mb or rate_pct."),
    "text_numbers": (0.15, 'Store one numeric column as text, like "14" or "2.50".'),
    "messy_case": (0.15, 'In one text column, write the same category inconsistently in some rows, such as '
                         '"High", "high " and "HIGH".'),
    "duplicates": (0.15, "Repeat one or two rows exactly, changing only the id column."),
    # Only the target profile asks for these two. They sit last so earlier profiles draw the same worlds.
    "sparse_numbers": (0.0, "Put null in two or three cells of each of two numeric columns that belong together "
                            "in a sum or a product, such as a fee and a tip, or a quantity and a unit price."),
    "state_dates": (0.0, "Include one date column that stays null until something happens, such as cancelled_on, "
                         "delivered_on, closed_on or returned_on, and leave it null in about a third of the rows."),
}

WORLD_SYSTEM = """\
You invent small, realistic datasets for testing data tools.

Reply with one JSON object and nothing else:
{"tables": {"<table_name>": [{...row...}, ...]}}

Rules:
- Each table has 9 to 14 rows and 5 to 9 columns. Every row has every column.
- Table and column names are lowercase snake_case.
- Each table has an id column, one to three text columns with only 2 to 5 distinct values, at least two \
numeric columns (some whole numbers, some decimals), and at least one date ("2026-03-14") or datetime \
("2026-03-14T09:30:00") column written as text.
- Put null in two to four cells, in one or two columns.
- Values are plain JSON scalars: text, numbers, true, false or null. No nested objects or lists.
- Vary the numbers and avoid ties within a numeric column, so sorting and ranking have one answer."""


# A second mix for what the first training run was weakest at: joins, unit
# conversions, data quirks and the rarer operations. Its worlds always have two tables and unit columns.
FOCUS_TRAITS = {"two_tables": 1.0, "units": 1.0, "text_numbers": 0.35, "messy_case": 0.35, "duplicates": 0.35}

# A third mix, aimed at what the k2 model still got wrong on evals/v2: missing values that
# count as 0 in more than one operand, a per-row value totalled per group, the direction of a sort or of a
# filter on a state, everyday wording the data settles, joins, plans of four or more steps, and when to ask.
TARGET_TRAITS = {"two_tables": 0.7, "units": 0.3, "text_numbers": 0.15, "messy_case": 0.15, "duplicates": 0.1,
                 "sparse_numbers": 0.7, "state_dates": 0.7}
_PROFILES = {"broad": ("w", None), "focus": ("f", FOCUS_TRAITS), "target": ("t", TARGET_TRAITS)}


def world_spec(seed: int, index: int, profile: str = "broad") -> dict:
    """The topic and traits of world `index`: the same seed and profile always give the same spec."""
    prefix, chances = _PROFILES[profile]
    rng = random.Random(f"{seed}-{profile}-world-{index}" if chances else f"{seed}-world-{index}")
    chances = chances or {name: chance for name, (chance, _) in TRAITS.items()}
    return {
        "id": f"{prefix}{index:05d}",
        "topic": rng.choice(TOPICS),
        "traits": [name for name in TRAITS if rng.random() < chances.get(name, 0.0)],
        "profile": profile,
    }


def world_prompt(spec: dict) -> str:
    lines = [f"Topic: {spec['topic']}."]
    lines.append(TRAITS["two_tables"][1] if "two_tables" in spec["traits"] else "Make one table.")
    lines += [TRAITS[name][1] for name in spec["traits"] if name != "two_tables"]
    lines.append("Do not name a table any of: " + ", ".join(sorted(BANNED_TABLES)) + ".")
    return "\n".join(lines)


FOCUS = {
    "basic": (
        "filter rows on one condition", "filter rows on two conditions joined by and/or",
        "keep rows whose category is one of several values", "rows where a value is missing",
        "sort by one column", "the top N rows by a column", "return only a few columns", "rename columns",
        "drop columns", "filter, then return a few columns", "one row per distinct value of a column",
    ),
    "derive": (
        "add a column computed from two numeric columns", "add a true/false flag column",
        "add a text label column with if/else", "sort values into three bands with nested if/else",
        "replace missing values in a column with a default", "a computed column rounded to given decimals",
        "update a column in place for only some rows", "the larger or smaller of two columns per row",
    ),
    "aggregate": (
        "count rows per category", "sum a column per category", "mean and count per category",
        "min and max of a column overall, in one row", "count distinct values", "median per group",
        "group by two columns", "count only the rows where a column is not missing, per group",
    ),
    "multi": (
        "filter, then aggregate, then sort", "compute a value per row, total it per group, sort the groups",
        "aggregate, then keep only groups above a threshold", "a ratio of two aggregates per group, rounded",
        "a rate per group: flag rows as 1 or 0, then sum and count", "the top N groups by a total",
        "filter, compute a column, return a few columns, sort", "difference between a group's max and min",
        "share of the overall total per group, as a percentage",
    ),
    "join": (
        "add columns from the other table, keeping rows that have no match",
        "join, then filter on a column from the other table", "join, then aggregate by a column of the other table",
        "rows that have no match in the other table", "total one table per key, join to the other, fill missing with 0",
        "join, compute a column that uses both tables",
    ),
    "units": (
        "convert a measurement column to another unit in a new column", "convert a measurement and round it",
        "convert, then filter on the converted value", "total a measurement per group, then convert the total",
    ),
    "text_date": (
        "filter by the year or month of a date", "count rows per month", "days from a date column to a fixed date",
        "upper or lower case a text column", "rows whose text contains, starts with or ends with something",
        "build a label by joining two columns", "rows between two dates inclusive", "the hour of a datetime",
        "take one piece of a text value split on a separator",
    ),
    "text_numbers": ("use the numeric column that is stored as text in arithmetic, a total or a numeric sort",),
    "messy_case": ("group or filter on the inconsistently written category, treating its spellings as one value",),
    "duplicates": ("count or total after removing the repeated rows, which differ only in the id",),
    "rare_ops": (
        "rename one or more columns", "drop one or more columns", "turn a text column into numbers or true/false",
        "remove duplicate rows by a key, keeping the first or last", "keep only the first N rows after sorting",
        "a plain filter that returns every column, asked in a few casual words",
        "a plain sort that returns every column, asked in a few casual words",
        "one row per distinct combination of two columns",
    ),
    "null_zero": (
        "add a column from two or three numeric columns, where a missing value in any of them counts as 0",
        "total per group of a value built from two columns that both have gaps, a missing value counting as 0",
        "a per-row sum or difference of two columns with gaps, missing counting as 0, then keep rows above a threshold",
        "an average per group where a missing value counts as 0 instead of being skipped",
        "one overall total of two columns added together, each with gaps, missing counting as 0, in one row",
    ),
    "row_value": (
        "total per group of a product of two columns, such as quantity times price",
        "average per group of the difference between two columns",
        "one overall total of a value computed from two columns, in one row",
        "the largest per-row value of a computed quantity in each group",
    ),
    "direction": (
        "sort so the smallest comes first, asked with a word like slowest, cheapest, oldest, earliest, lightest or shortest",
        "the bottom N rows by a column, asked with a word like least, lowest, fewest or worst",
        "rows in a state that a missing date shows, such as still open, not yet delivered, active or unpaid",
        "rows in a state that a present date shows, such as cancelled, delivered, closed or returned",
        "sort by a date so the most recent or the oldest comes first, and return a few columns",
        "groups ordered by a total or a count, smallest first",
    ),
    "everyday": (
        "a condition said in everyday words that the data settles exactly, such as none left for a quantity of 0, "
        "free for a price of 0, unassigned for a missing owner, or overdue for a due date before a stated date",
        "a category referred to by what it means instead of by its code, where the data makes the meaning plain",
        "a threshold given as a number with its unit in words, on a column whose name carries that unit",
    ),
    "long": (
        "filter, compute a column, total it per group, keep only groups above a threshold, sort",
        "join, compute a column that uses both tables, total it per group of the other table, sort, keep the top N",
        "total one table per key, join to the other keeping keys with no rows, fill the missing total with 0, then filter",
        "a rate per group: flag rows as 1 or 0, sum and count per group, divide, round, return the group and the rate",
        "share of the overall total per group, as a percentage",
        "the latest row per key by a date, then a filter on it, returning a few columns",
        "per group, a total for each of two categories side by side, then the groups where one exceeds the other",
    ),
    "clarify": (
        "apply a rate, percentage or fee that is neither in the data nor stated",
        "use a threshold described only with a vague word such as high, large, recent or old",
        "refer to something that could be either of two columns, where the choice changes the answer",
        "depend on a value only the requester knows, such as 'the usual' discount or 'our target'",
    ),
    "unsupported": UNSUPPORTED_ACTIONS,
}

_KINDS = (("basic", 15), ("derive", 14), ("aggregate", 14), ("multi", 22), ("join", 12), ("units", 6),
          ("text_date", 8), ("clarify", 5), ("unsupported", 4))
_FOCUS_KINDS = (("join", 30), ("units", 20), ("rare_ops", 16), ("multi", 12), ("derive", 8), ("aggregate", 5),
                ("text_date", 5), ("clarify", 2), ("unsupported", 2))
_TARGET_KINDS = (("join", 22), ("long", 18), ("null_zero", 14), ("direction", 12), ("row_value", 10), ("everyday", 8),
                 ("clarify", 10), ("text_date", 4), ("unsupported", 2))
_MIXES = {"focus": _FOCUS_KINDS, "target": _TARGET_KINDS}
_STYLES = (("plain", 45), ("terse", 20), ("chatty", 20), ("hinglish", 15))


def recipes(seed: int, spec: dict, count: int = 10) -> list[dict]:
    """A varied batch of example recipes for one world, shaped by the world's traits."""
    rng = random.Random(f"{seed}-recipes-{spec['id']}")
    traits = set(spec["traits"])
    mix = _MIXES.get(spec.get("profile"), _KINDS)
    kinds = rng.choices([k for k, _ in mix], weights=[w for _, w in mix], k=count)
    kinds = ["multi" if k == "join" and "two_tables" not in traits else k for k in kinds]
    kinds = ["derive" if k == "units" and "units" not in traits else k for k in kinds]
    # Each data quirk a world was built with gets at least one example that depends on it.
    for slot, trait in enumerate(t for t in ("text_numbers", "messy_case", "duplicates") if t in traits):
        kinds[slot] = trait
    rng.shuffle(kinds)
    styles = rng.choices([s for s, _ in _STYLES], weights=[w for _, w in _STYLES], k=count)
    batch = []
    for n, (kind, style) in enumerate(zip(kinds, styles), start=1):
        response = kind if kind in ("clarify", "unsupported") else "plan"
        batch.append({"n": n, "kind": response, "topic": kind, "about": rng.choice(FOCUS[kind]), "style": style})
    return batch


AUTHOR_SYSTEM = f"""\
You write training examples for Vasuki, a planner that turns a data request into a plan.

You are given the tables, with a profile and every row, and a numbered list of recipes. For each recipe \
write one example: a request a person might type, and the correct Vasuki response to it.

Reply with one JSON object and nothing else:
{{"examples": [{{"n": 1, "request": "...", "response": {{...}}}}, ...]}}

Writing the request:
- It is answered from the tables alone. A second analyst will solve it in pandas, seeing only the request \
and the table profile, and the example is thrown away unless their table equals yours exactly, column \
names included. So pin the output: say which columns to return unless it is all of them, name every new \
column, say how rows are ordered when order matters, give the number of decimals when rounding, and say \
how missing values count when that changes the answer.
- Loose: "Median distance per region." Tight: "Median distance_mi per region, as median_distance. Columns \
region, median_distance."
- Loose: "Count bookings per priority." Tight: "Count bookings per priority, as bookings. Bookings with no \
priority form their own group. Columns priority, bookings."
- Loose: "Join deliveries with logs and compute the difference." Tight: "For deliveries that have a log, \
add duration_diff = duration_min minus logged_min. Return delivery_id and duration_diff."
- A terse or Hinglish request pins the output too: "region-wise total volume nikalo, column total_volume, \
sabse zyada pehle. Columns region, total_volume."
- When a text column is written inconsistently, either say to treat its spellings as one value and what \
to call that value, or match one exact spelling.
- Follow the recipe's style. "plain": one or two clear sentences. "terse": a short note, as in a chat \
message to a colleague. "chatty": conversational, with a little context. "hinglish": a Hindi-English mix \
in Roman script, the way Indian colleagues write in chat. Column names always stay exactly as they are.
- Write values exactly as they appear in the data.
- Vary the wording from one request to the next.
- "about" says what the example should exercise. If it does not fit these tables, write the closest \
request that does.

Writing the response:
- kind "plan": {{"status": "plan", "steps": [...]}} doing exactly what the request says, no more. When there \
is more than one table the first step must name its `input`; after that, give a step `input` only when \
it does not read the previous step's result, and `as` only when a later step uses the name. Make sure the result has at least one row.
- kind "clarify": the request leaves out the one thing "about" describes, and nothing in the tables \
settles it. Response: {{"status": "needs_clarification", "question": "..."}}, asking for exactly that.
- kind "unsupported": the request asks for the action "about" describes. Response: \
{{"status": "unsupported", "reason": "..."}} with a short reason. Give this response even when part of the \
request could be planned: a partial plan would silently drop the rest.

Reference for the plan format:
{json.dumps(describe(), separators=(",", ":"))}"""


REWRITE_SYSTEM = """\
You shorten data requests. Each item has a request and the Vasuki plan that answers it. Rewrite the \
request the way a busy person would actually type it: say what they want, never how to get it.

Reply with one JSON object and nothing else:
{"rewrites": [{"n": 1, "request": "..."}, ...]}

Rules:
- Name no operation and no step: no "join", "merge", "match", "combine", "filter", "group", "aggregate", \
"convert", "deduplicate", "cast", "then". Do not say which tables to bring together; a column's name is \
enough to tell where it lives.
- Keep everything that changes the answer: conditions, exact values, how missing values count, rounding, \
ordering, limits, and the names of new columns.
- Say which columns come back only when it is not all of them or not the obvious ones, and do it \
briefly, like "X: a, b." or "as total_kg".
- Aim for half the length or less, one sentence where possible. Keep the original's voice: a Hinglish \
request stays Hinglish, a chatty one stays chatty.
- A second analyst must still get exactly the same table from your wording alone. If a request cannot \
be shortened without losing that, return it unchanged.

Examples:
- Long: "Join package_bookings with tour_packages on package_id (keep only bookings that match a \
package), then keep bookings for destination Tokyo. Return booking_id, customer_type, destination."
  Short: "Bookings for packages going to Tokyo: booking_id, customer_type, destination."
- Long: "Convert oven_temp_c from Celsius to Fahrenheit as a new column oven_temp_f, then keep only the \
batches where oven_temp_f is at least 400. Return batch_id, oven_temp_f."
  Short: "Batches that baked at 400 F or hotter: batch_id and oven_temp_f."
- Long: "For each field_zone, count the pumps with status 'active' as active_pumps. Return field_zone, \
active_pumps, sorted by active_pumps descending."
  Short: "Active pumps per field_zone, as active_pumps, most first."
- Long: "Add a column total_km that converts distance_mi to kilometres and round it to 1 decimal. \
Return all columns."
  Short: "Add total_km, the distance in kilometres, to 1 decimal."
- Long: "Har region ke liye volume_liters ka sum nikalo, column ka naam total_volume rakho, aur sabse \
bada total pehle dikhao. Columns region, total_volume."
  Short: "Region-wise total volume_liters, total_volume naam se, sabse zyada pehle."
"""


NATURAL_SYSTEM = """\
You reword data requests so they sound the way people talk about their data. Each item has a request \
and the Vasuki plan that answers it.

Reply with one JSON object and nothing else:
{"rewrites": [{"n": 1, "request": "..."}, ...]}

Rules:
- Refer to existing columns by what they mean, not by their names: "amount" or "how much" for \
amount_usd, "not checked out yet" for a missing checkout_time, "no copies left" for copies_available \
equal to 0. Keep a column's exact name only where the request names a NEW column to create, or where \
two columns would otherwise be confused.
- Say values the way a person would when the data makes the meaning plain: "the north region" for 'N', \
"walk-ins" for 'walk-in'. Keep the exact spelling of names, ids and anything that could be mistaken.
- Keep numbers, dates, thresholds, rounding, ordering, limits and how missing values count exactly as \
they are.
- Where the request lists the columns to return, you may name them in words ("the loan id and the \
amount"), as long as it stays clear which columns are meant and what any new column is called.
- Add no requirement and drop none. Name no operation (join, filter, group, convert).
- Keep it short, and keep the original's voice: a Hinglish request stays Hinglish.
- A second analyst who sees only your wording and the table profile must still produce exactly the same \
table. If an item cannot be reworded safely, return it unchanged.

Examples:
- Before: "Loans with amount_usd over 5000 and branch 'Central': loan_id, amount_usd."
  After: "Central branch loans above 5,000 dollars: the loan id and the amount."
- Before: "Books where copies_available == 0: isbn, title."
  After: "Which books have no copies left? Give the isbn and title."
- Before: "Rows where checkout_time is null, all columns."
  After: "Guests who have not checked out yet."
- Before: "Average wait_min per clinic as avg_wait, excluding clinic 'walk-in'."
  After: "Average wait per clinic, leaving out walk-ins, as avg_wait."
- Before: "region in ['N','S'] wale stores, sirf store_id aur manager."
  After: "North aur South ke stores, sirf store id aur manager."
"""

# Each style: the system prompt that produces it, and the suffix its rewritten example ids carry.
REWRITE_STYLES = {"intent": (REWRITE_SYSTEM, "-r"), "natural": (NATURAL_SYSTEM, "-n")}
