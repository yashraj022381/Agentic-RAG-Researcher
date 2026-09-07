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

_ID_VALUE_PATTERN = re.compile(
    r'\b(?:passenger\s*id|customer\s*id|record\s*id|\bid)\s*(?:number|no\.?|#|is|of|:|=)?\s*(\d+)\b',
    re.IGNORECASE)

_MISSING_PATTERN = re.compile(
    r'\b(missing|null|nan|empty)\s+(values?|entries|counts?|data)\b', re.IGNORECASE
)

_DERIVED_COLUMN_PATTERN = re.compile(
    r'\b(engineer|create|compute|calculate)\s+(?:a\s+|an\s+)?[\w\s]{0,30}?'
    r'(variance|difference|per\s*capita|ratio|margin)\b',
    re.IGNORECASE,
)


 
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
        col_words = _normalize_words(col)
        if col_words & query_words: 
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



def detect_row_lookup_plan(query: str, schema: dict) -> Optional[dict]:
    """Deterministically detect a 'find/get X for row ID N' question —
    filter to one row by an ID-like column, return the requested
    column(s). Mirrors detect_top_n_plan's approach: regex-first,
    schema-checked, no LLM guesswork for this common query shape."""
    q_lower = query.lower()
    id_match = _ID_VALUE_PATTERN.search(q_lower)
    if not id_match:
        return None

    id_value = id_match.group(1)
    columns = schema["columns"]
    q_words = set(re.findall(r'\w+', q_lower))

    best_id_col, best_score = None, 0
    for c in columns:
        c_words = set(re.findall(r'\w+', c.lower()))
        if not any('id' in w for w in c_words):
            continue
        score = len(c_words & q_words)
        if score > best_score:
            best_score = score
            best_id_col = c

        if best_id_col is None:
            id_like = [c for c in columns if 'id' in c.lower()]
            if len(id_like) == 1:
                best_id_col = id_like[0]
            else:
                return None  # ambiguous or no id-like column — don't guess


    return {
        "applicable": True,
        "op": "row_lookup",
        "id_column": best_id_col,
        "id_value": id_value,
        "return_columns": [],  # empty = return all columns
    }

def detect_missing_values_plan(query: str, schema: dict) -> Optional[dict]:
    q_lower = query.lower()
    if not _MISSING_PATTERN.search(q_lower):
        return None
    query_words = set(re.findall(r'\w+', q_lower))
    columns = schema["columns"]
    matched_cols = [c for c in columns if _normalize_words(c) & query_words]
    if not matched_cols:
        matched_cols = columns  # no specific columns named → report all
    return {"applicable": True, "op": "missing_count", "columns": matched_cols}

_METRIC_PATTERNS = [
    (re.compile(r'\bvariance\b', re.IGNORECASE), 'subtract',
     ['actual', 'spend'], ['budget', 'allocated', 'planned']),
    (re.compile(r'\bper\s*capita\b|\bper\s*person\b|\bper\s*employee\b', re.IGNORECASE), 'divide',
     ['actual', 'spend', 'revenue', 'cost', 'amount'], ['headcount', 'employees', 'staff']),
    (re.compile(r'\bmargin\b', re.IGNORECASE), 'subtract', ['revenue'], ['cost', 'expense']),
]


def _pick_column(hint_words, numeric_cols):
    for col in numeric_cols:
        col_words = _normalize_words(col)
        if any(h in col_words for h in hint_words):
            return col
    return None

def detect_derived_arithmetic_plan(query: str, schema: dict) -> Optional[dict]:
    """Detect 'engineer/compute X for every row' where X is simple
    arithmetic (subtract or divide) between two existing numeric columns —
    e.g. 'engineer a Spend_Variance column' (Actual - Budget), 'compute
    spend per capita' (Actual / Headcount). Executed vectorized across
    every row in one pandas call, not looped through calculator N times."""
    q_lower = query.lower().replace('_', ' ')
    
    columns = schema["columns"]
    numeric_cols = [c for c in columns if any(t in str(schema["dtypes"].get(c, "")) for t in ("int", "float"))]
    print(f"      [DEBUG] derived_arithmetic: numeric_cols={numeric_cols}")


    metrics = []
    for pattern, arith_op, num_hints, den_hints in _METRIC_PATTERNS:
        matched = bool(pattern.search(q_lower))
        print(f"      [DEBUG] derived_arithmetic: pattern={pattern.pattern!r} matched={matched}")
        if not matched:
            continue
        numerator = _pick_column(num_hints, numeric_cols)
        denominator = _pick_column(den_hints, numeric_cols)
        print(f"      [DEBUG] derived_arithmetic: numerator={numerator}, denominator={denominator}")
        if numerator and denominator and numerator != denominator:
            metrics.append({"op": arith_op, "numerator": numerator, "denominator": denominator})

    print(f"      [DEBUG] derived_arithmetic: final metrics={metrics}")
    if not metrics:
        return None
    all_cols = list({m["numerator"] for m in metrics} | {m["denominator"] for m in metrics})
    return {"applicable": True, "op": "derived_arithmetic", "metrics": metrics, "return_columns": all_cols}


def plan_operation(query: str, schema: dict, llm) -> Optional[dict]:
    

    row_plan = detect_row_lookup_plan(query, schema)
    if row_plan:
        return row_plan
    top_n_plan = detect_top_n_plan(query, schema)
    print(f"      [DEBUG] plan_operation: top_n_plan={top_n_plan}")
    if top_n_plan:
        return top_n_plan
    missing_plan = detect_missing_values_plan(query, schema)
    if missing_plan:
        return missing_plan
    derived_plan = detect_derived_arithmetic_plan(query, schema)
    if derived_plan:
        return derived_plan

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
         f"If the question asks for a RATE or PERCENTAGE of a binary/categorical outcome, "
         f"filtered by one attribute and compared across another (e.g. 'survival rate of "
         f"female passengers in Pclass 1 vs Pclass 3'), respond:\n"
         f'{{"applicable": true, "op": "conditional_rate", "filter_column": "<column>", '
         f'"filter_value": "<value>", "group_by": "<column>", "rate_column": "<outcome column>", '
         f'"rate_positive_value": "<value meaning positive outcome, e.g. 1>"}}\n\n'
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

def execute_row_wise_derivation(path, columns_needed, formula_type):
    # e.g. formula_type="variance": Actual_Spend - Budget_Allocated per row
    # formula_type="per_capita": Actual_Spend / Headcount per row
    df = pd.read_csv(path)
    ...
    lines = [f"{row['Department']} {row['Quarter']}: Variance={...}, Per_Capita={...:.2f}" for _, row in df.iterrows()]
    return "VERIFIED COMPUTED RESULT (row-wise derivation...)\n" + "\n".join(lines)
 
 
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
        
    #op = plan.get("op")
    if op == "row_lookup":
        id_column = str(plan.get("id_column")or "")
        id_value = plan.get("id_value")
        return_cols = plan.get("return_columns") or []
       
        
        if not id_column:
            return None
            
        try:
            df = pd.read_csv(path)
            if id_column not in df.columns:
                return None
            
            matched_rows = df[df[id_column].astype(str) == str(id_value)]
            if matched_rows.empty:
                return None 

            row = matched_rows.iloc[0]
            cols_to_show = [c for c in return_cols if c in df.columns] or list(df.columns)
            details = ", ".join(f"{c}={row[c]}" for c in cols_to_show)
            return (
                f"VERIFIED COMPUTED RESULT (row lookup, {id_column}={id_value}, "
                f"from the actual dataset via pandas — exact, not estimated): {details}"
            )
        except Exception:
            return None

    if op == "conditional_rate":
        filter_col, filter_val = plan.get("filter_column"), plan.get("filter_value")
        group_col, rate_col = plan.get("group_by"), plan.get("rate_column")
        pos_val = plan.get("rate_positive_value")
        try:
            df = pd.read_csv(path)
            for c in (filter_col, group_col, rate_col):
                if c not in df.columns:
                    return None
            filtered = df[df[filter_col].astype(str).str.lower() == str(filter_val).lower()]
            lines = []
            for group_val, sub in filtered.groupby(group_col):
                total = len(sub)
                positive = (sub[rate_col].astype(str) == str(pos_val)).sum()
                rate = (positive / total * 100) if total else 0
                lines.append(f"{group_col}={group_val}: {positive}/{total} = {rate:.2f}%")
            return (
                f"VERIFIED COMPUTED RESULT ({rate_col} rate for {filter_col}={filter_val}, "
                f"grouped by {group_col}, from the actual dataset via pandas — exact):\n" + "\n".join(lines)
            )
        except Exception:
            return None

    if op == "missing_count":
        columns = plan.get("columns") or []
        try:
            df = pd.read_csv(path)
            valid_cols = [c for c in columns if c in df.columns]
            if not valid_cols:
                return None
            lines = [f"{c}: {int(df[c].isna().sum())} missing" for c in valid_cols]
            return (
                "VERIFIED COMPUTED RESULT (missing-value counts, from the actual "
                "dataset via pandas — exact, not estimated):\n" + "\n".join(lines)
            )
        except Exception:
            return None

    if op == "derived_arithmetic":
        metrics = plan.get("metrics") or []
        if not metrics:
            return None
        try:
            df = pd.read_csv(path)
            for m in metrics:
                if m["numerator"] not in df.columns or m["denominator"] not in df.columns:
                    return None
            used_cols = {m["numerator"] for m in metrics} | {m["denominator"] for m in metrics}
            id_cols = [c for c in df.columns if c not in used_cols]
            raw_cols = sorted(used_cols)  # the actual Budget_Allocated / Actual_Spend / Headcount inputs
 
            lines = []
            for _, row in df.iterrows():
                id_desc = ", ".join(f"{c}={row[c]}" for c in id_cols)
                raw_desc = ", ".join(f"{c}={row[c]}" for c in raw_cols)
                parts = []
                for m in metrics:
                    a, b = row[m["numerator"]], row[m["denominator"]]
                    result = (a - b) if m["op"] == "subtract" else (a / b if b else None)
                    label = (f"{m['numerator']}_minus_{m['denominator']}" if m["op"] == "subtract" else f"{m['numerator']}_per_{m['denominator']}")
                    parts.append(f"{label}={result:.2f}" if result is not None else f"{label}=undefined")
                lines.append(f"{id_desc}, {raw_desc}: " + ", ".join(parts))
            return (
                "VERIFIED COMPUTED RESULT (derived metrics, for every row, from the actual "
                "dataset via pandas — exact, not estimated):\n" + "\n".join(lines)
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
