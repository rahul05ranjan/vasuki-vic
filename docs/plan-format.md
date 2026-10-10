# The plan format

A planner, whether a model or a person, describes a computation as JSON. The engine checks the whole plan against the input columns, runs it, and returns a compact report. This page describes what a planner returns and how the engine behaves.

`vasuki schema` prints the full reference: every operation and its fields, the expression functions and the units. Its `response_schema` is a JSON Schema, so it can drive constrained decoding.

## The response

A planner returns one of three things:

```json
{"status": "plan", "steps": [{"op": "filter", "where": "amount > 100"}]}
{"status": "needs_clarification", "question": "Is GST included in amount?"}
{"status": "unsupported", "reason": "needs a currency exchange rate"}
```

A request the operations can only partly do gets `unsupported`, never a partial plan. A partial plan that says nothing about the part it dropped would mislead the caller.

## Steps

Steps run in order. A step reads the previous step's result unless `input` names a table. With more than one input table, the first step must name its `input`. `as` names a step's result so a later step or a join can use it. The last step's result is the output.

This is `examples/invoice_totals.json`: drop repeated invoices, compute GST per row, total per vendor, largest first.

```json
{"status": "plan", "steps": [
  {"op": "deduplicate", "keys": ["vendor", "invoice_number"]},
  {"op": "derive", "column": "gst", "expr": "round(amount * gst_rate, 2)"},
  {"op": "aggregate", "group_by": ["vendor"], "metrics": [
    {"as": "invoices", "fn": "count"},
    {"as": "amount", "fn": "sum", "column": "amount"},
    {"as": "gst", "fn": "sum", "column": "gst"}]},
  {"op": "sort", "by": [{"column": "amount", "order": "desc"}]}
]}
```

## Operations

| Operation | What it does |
|---|---|
| `filter` | Keep the rows where an expression is true |
| `derive` | Set a column to the value of an expression for every row, adding it if it is new |
| `rename` | Rename columns |
| `select` | Keep the named columns, in that order |
| `drop` | Remove the named columns |
| `cast` | Convert a column to another type; a value that cannot convert is an error |
| `sort` | Order rows by one or more columns; nulls go last |
| `deduplicate` | Drop rows that repeat an earlier row, on the named keys or on all columns |
| `aggregate` | Group rows and compute metrics per group |
| `join` | Combine two tables on key columns |
| `convert_unit` | Convert a column between units of length, mass, time, data size, ratio or temperature |
| `limit` | Keep the first rows |

## Expressions

`where` and `expr` take a Python-style expression over the row's columns. Bare names are columns; `col("name with spaces")` reaches a name that is not an identifier. Literals are numbers, `'text'`, `true`, `false` and `null`. Operators are `+ - * / // %`, the comparisons, `in` and `not in` with a list, `and`, `or`, `not`, and `x if cond else y`. Functions are called as `f(x, y)`.

Expressions are not passed to `eval`. They are parsed to a syntax tree and interpreted against a whitelist. There is no indexing, slicing or `**`. A few Python habits are accepted as other spellings: `round(x, digits=2)`, `x.lower()`, `x.startswith(y)`, `x is null`.

## Behaviour worth knowing

- **Checked before run.** Every column, table, unit and expression is resolved against the input columns before any step runs. A typo in step 5 fails immediately and names the closest match.
- **Step names can be reused.** `as` may name a result after an existing table; later steps see the newest one.
- **Strict types.** Arithmetic on text is an error, not a coercion. Use `cast`, or `int()` and `float()` in an expression. The one exception is a join key: when it is a number in one table and text in the other, the text is read as the number it spells, as a database would.
- **Nulls.** They propagate through arithmetic and functions, compare as false, sort last, never match in a join, and are skipped by aggregates.
- **Rounding** is half-up on the decimal as written: `round(2.675, 2)` is `2.68`.
- **CSV typing** is per column: all-integer columns become ints, all-numeric become floats, anything else stays text. Values with leading zeros (`01234`) stay text.
- **No currency units.** Exchange rates are data, so a plan must multiply by an explicit rate.

## Errors

A plan that cannot run fails with a structured error that names the step and, for a misspelt name, the closest match:

```json
{
  "status": "error",
  "code": "unknown_column",
  "message": "unknown column 'amont', did you mean 'amount'? (available: vendor, invoice_number, amount, gst_rate)",
  "step": 0,
  "op": "filter"
}
```

`vasuki check PLAN -i FILE` reports the same errors without running anything.
