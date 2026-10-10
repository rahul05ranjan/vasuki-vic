"""Subprocess entry point: run model-written pandas code and write the result as JSON.

Reads {"code": ..., "tables": {name: {"columns": [...], "rows": [...]}}} on stdin and writes
{"columns": [...], "rows": [[...], ...]} to the path in argv[1].
"""

import datetime
import json
import sys
import traceback

import numpy as np
import pandas as pd


def clean(value):
    """Turn a pandas cell into the plain JSON value the scorer compares."""
    if value is None or value is pd.NaT or value is pd.NA:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and value != value:
        return None
    if isinstance(value, (datetime.datetime, datetime.date)):
        stamp = pd.Timestamp(value)
        return stamp.date().isoformat() if stamp == stamp.normalize() else stamp.isoformat()
    if isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def main() -> int:
    payload = json.load(sys.stdin)
    tables = {name: pd.DataFrame(t["rows"], columns=t["columns"]) for name, t in payload["tables"].items()}
    namespace = {"pd": pd, "np": np}
    try:
        exec(payload["code"], namespace)
        result = namespace["solve"](tables)
        if isinstance(result, pd.Series):
            result = result.to_frame()
        if not isinstance(result, pd.DataFrame):
            raise TypeError(f"solve() returned {type(result).__name__}, not a DataFrame")
        columns = [str(c) for c in result.columns]
        if len(set(columns)) != len(columns):
            raise ValueError("solve() returned duplicate column names")
        rows = [[clean(v) for v in row] for row in result.itertuples(index=False, name=None)]
    except Exception:
        sys.stderr.write(traceback.format_exc(limit=3))
        return 1
    with open(sys.argv[1], "w", encoding="utf-8") as f:
        json.dump({"columns": columns, "rows": rows}, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
