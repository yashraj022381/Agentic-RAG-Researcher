import json
import re
from pathlib import Path
from typing import Optional
 
try:
    import pandas as pd
except ImportError:
    pd = None
 
ALLOWED_AGGS = {"mean", "sum", "count", "median", "min", "max", "std", "nunique"}
 
 
def is_csv(path: Path) -> bool:
    return path.suffix.lower() == ".csv"
 
 
def load_csv_schema(path: Path, sample_rows: int = 3) -> Optional[dict]:
    """
    Return column names, dtypes, shape, and a few sample rows — NOT the
    full dataset — so the LLM can plan an operation without needing (or
    being able) to eyeball the actual data itself.
    """
    if pd is None:
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
 
    return {
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "shape": df.shape,
        "sample_rows": df.head(sample_rows).to_dict(orient="records"),
    }
 
 
def plan_operation(query: str, schema: dict, llm) -> Optional[dict]:
    """
    Ask the LLM to translate the question into a structured, whitelisted
    pandas operation — never to compute the answer itself by reading data.
    """
    prompt = (
        f"You have a CSV dataset with these columns: {schema['columns']}\n"
        f"Data types: {schema['dtypes']}\n"
        f"Shape: {schema['shape'][0]} rows, {schema['shape'][1]} columns\n"
        f"Sample rows: {schema['sample_rows']}\n\n"
        f"QUESTION: {query}\n\n"
        f"Does this question ask for a computable statistic from this "
        f"dataset (e.g. an average, sum, count, or comparison grouped by a "
        f"column)? If yes, respond in EXACTLY this JSON format and nothing "
        f"else:\n"
        f'{{"applicable": true, "group_by": "<column name or null>", '
        f'"target_column": "<column name>", "agg": "<one of {sorted(ALLOWED_AGGS)}>"}}\n\n'
        f"If the question is NOT answerable as a simple aggregation from "
        f"this dataset, respond:\n"
        f'{{"applicable": false, "reason": "<brief reason>"}}\n\n'
        f"Respond with ONLY the JSON object, nothing else — no explanation, "
        f"no markdown formatting."
    )
    try:
        raw = llm.chat(
            system=(
                "You translate data questions into structured pandas "
                "operations. You never compute the answer yourself — you "
                "only identify which column(s) and which operation apply. "
                "Respond only with a JSON object."
            ),
            user=prompt,
            max_tokens=300,
        )
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not match:
            return None
        plan = json.loads(match.group(0))
        return plan
    except Exception:
        return None
 
 
def execute_plan(path: Path, plan: dict) -> Optional[str]:
    """
    Deterministically execute a whitelisted aggregation. This NEVER
    eval()s or exec()s anything the LLM produced — only a fixed, small set
    of pandas .agg() calls can run, gated by the ALLOWED_AGGS whitelist and
    a real column-name check against the actual dataframe.
    """
    if pd is None or not plan.get("applicable"):
        return None
 
    agg = plan.get("agg")
    target = plan.get("target_column")
    group_by = plan.get("group_by")
 
    if agg not in ALLOWED_AGGS or not target:
        return None
 
    try:
        df = pd.read_csv(path)
        if target not in df.columns:
            return None
 
        if group_by and group_by in df.columns:
            result = df.groupby(group_by)[target].agg(agg)
            lines = [
                f"{group_by}={idx}: {agg}({target}) = {val:.2f}"
                for idx, val in result.items()
            ]
            return (
                "VERIFIED COMPUTED RESULT (from the actual dataset via "
                "pandas — exact, not estimated):\n" + "\n".join(lines)
            )
        else:
            result = df[target].agg(agg)
            return (
                f"VERIFIED COMPUTED RESULT (from the actual dataset via "
                f"pandas — exact, not estimated): {agg}({target}) = {result:.2f}"
            )
    except Exception:
        return None
