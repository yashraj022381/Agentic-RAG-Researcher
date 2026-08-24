import json
import re
from pathlib import Path
from typing import Optional
 
try:
    import pandas as pd
except ImportError:
    pd = None
 
ALLOWED_AGGS = {"mean", "sum", "count", "median", "min", "max", "std", "nunique"}
ALLOWED_OPS = ALLOWED_AGGS | {"top_n", "bottom_n"}

_TOP_N_PATTERN = re.compile(r'\b(top|highest|largest|greatest)\s+(\d+)?', re.IGNORECASE)
_BOTTOM_N_PATTERN = re.compile(r'\b(bottom|lowest|smallest|least)\s+(\d+)?', re.IGNORECASE)


 
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
 
def _normalize_words(text: str) -> set:
    """Tokenize treating underscores as spaces, so 'Account_Manager' and
    'Account Manager' are recognized as the same words regardless of which
    naming convention the column or the query happens to use."""
    return set(re.findall(r'[a-z0-9]+', text.lower().replace('_', ' ')))

def detect_top_n_plan(query: str, schema: dict) -> Optional[dict]:
    """Deterministically detect a 'top N / bottom N sorted by column' question
    via regex, rather than relying on the LLM to recognize this shape amid a
    long instruction prompt. Only the sort column and return columns are
    fuzzy-matched against the actual schema — nothing is invented."""
    q_lower = query.lower()
    top_match = _TOP_N_PATTERN.search(q_lower)
    bottom_match = _BOTTOM_N_PATTERN.search(q_lower)
    print(f"      [DEBUG] detect_top_n_plan: query_lower={q_lower!r}, top_match={bool(top_match)}, bottom_match={bool(bottom_match)}")

    if not top_match and not bottom_match:
        return None

    match = top_match or bottom_match
    op = "top_n" if top_match else "bottom_n"
    n = int(match.group(2)) if match.group(2) else 5

    query_words = set(re.findall(r'\w+', q_lower))
    columns = schema["columns"]

    # Find the best-matching sort column: prefer a column whose full
    # underscore-joined name appears as a token in the query.
    sort_by = None
    best_overlap = 0
    for col in columns:
        col_words = set(re.findall(r'\w+', col.lower()))
        overlap = len(col_words & query_words) #+ (2 if col.lower() in q_lower else 0)
        if overlap > best_overlap:
            best_overlap = overlap
            sort_by = col

    print(f"      [DEBUG] detect_top_n_plan: query_words={query_words}, best sort_by='{sort_by}' (overlap={best_overlap})")

    if not sort_by or best_overlap == 0:
        return None  # no confident column match — let the LLM path try instead

    # Any other columns explicitly referenced in the query get included in
    # the output alongside the sort column (e.g. "and their Account Manager").
    return_columns = [sort_by]
    for col in columns:
        if col == sort_by:
            continue
        #col_words = set(re.findall(r'\w+', col.lower()))
        col_words = _normalize_words(col)
        if col_words & query_words: #or col.lower() in q_lower:
            return_columns.append(col)

    plan = {
        "applicable": True,
        "op": op,
        "sort_by": sort_by,
        "n": n,
        "return_columns": return_columns,
    }

    print(f"      [DEBUG] detect_top_n_plan RETURNING: {plan}")
    return plan

def plan_operation(query: str, schema: dict, llm) -> Optional[dict]:
    

    top_n_plan = detect_top_n_plan(query, schema)
    print(f"      [DEBUG] plan_operation: top_n_plan={top_n_plan}")
    if top_n_plan:
        return top_n_plan

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
         f"STRICT RULE: Only mark this 'applicable' if the SPECIFIC column(s) the "
         f"question asks about — by name or clear synonym — actually appear in the "
         f"columns list above. ...\n\n"
         f"Does this question ask for a computable statistic that GENUINELY EXISTS "
         f"in this dataset (e.g. an average, sum, count, or comparison grouped by a "
         f"column)? If yes, respond in EXACTLY this JSON format and nothing "
         f"else:\n"
         f'{{"applicable": true, "group_by": "<column name or null>", '
         f'"target_column": "<column name>", "agg": "<one of {sorted(ALLOWED_AGGS)}>"}}\n\n'
         f"If instead the question asks for the TOP or BOTTOM N rows sorted by a "
         f"column (e.g. 'top 5 customers by churn_risk_score', 'the 3 lowest "
         f"scoring accounts'), and the sort column GENUINELY EXISTS in the columns "
         f"list above, respond in EXACTLY this JSON format instead:\n"
         f'{{"applicable": true, "op": "top_n", "sort_by": "<column name>", '
         f'"n": <int>, "return_columns": ["<column1>", "<column2>", ...]}}\n'
         f"Use \"op\": \"bottom_n\" for lowest/smallest N instead of top_n. Include "
         f"in return_columns whatever other columns the question asks to see "
         f"alongside the ranking (e.g. names, IDs, assigned manager) — only "
         f"columns that actually exist in the columns list above.\n\n"
         f"If the specific column(s) the question needs are NOT in the columns "
         f"list, respond:\n"
         f'{{"applicable": false, "reason": "<name the missing column(s)>"}}\n\n'
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

    op = plan.get("op")
    if op in ("top_n", "bottom_n"):
        sort_by = plan.get("sort_by")
        n = plan.get("n", 5)
        return_cols = plan.get("return_columns") or []
        try:
            df = pd.read_csv(path)
            if sort_by not in df.columns:
                return None
            valid_cols = [c for c in return_cols if c in df.columns]
            ascending = (op == "bottom_n")
            result_df = df.sort_values(sort_by, ascending=ascending).head(n)
            cols_to_show = valid_cols or list(df.columns)
            lines = [
                ", ".join(f"{c}={row[c]}" for c in cols_to_show)
                for _, row in result_df.iterrows()
            ]
            return (
                f"VERIFIED COMPUTED RESULT (top {n} by {sort_by}, from the actual "
                f"dataset via pandas — exact, not estimated):\n" + "\n".join(lines)
            )
        except Exception:
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
