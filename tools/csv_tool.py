import re
from .base import BaseTool, ToolResult
from pathlib import Path
from utils.paths import DOCS_DIR
from utils.csv_analyzer import load_csv_schema, plan_operation, execute_plan

class CSVAnalyzerTool(BaseTool):
    def __init__(self, llm=None):
        # llm is injected at registration time (see registry.py) since
        # BaseTool.run()'s standard signature doesn't pass one through —
        # this tool is the one exception that genuinely needs it, to
        # classify whether a question is a computable aggregation.
        self.llm = llm
        
    @property
    def name(self) -> str:
        return "csv_analyzer"

    @property
    def description(self) -> str:
        return (
            "Inspect CSV file schemas and run safe, whitelisted aggregations "
            "(mean, sum, count, etc.) on uploaded CSV/dataset files. Use this "
            "instead of document_reader for questions about data files, "
            "columns, or computed statistics."
        )

    def run(self, query, context=None, llm=None) -> ToolResult:
        csv_files = [f for f in Path(DOCS_DIR).rglob("*.csv") if f.is_file()]
        if not csv_files:
            return ToolResult(content="No CSV files available.", source="csv_analyzer: none", confidence=0.1)

        if self.llm is None:
            return ToolResult(
                content="csv_analyzer is unavailable (no LLM reference configured).",
                source="csv_analyzer: unconfigured",
                confidence=0.1,
            )
        
        query_words = set(w for w in re.findall(r'\w+', query.lower()) if len(w) > 3)
        checked = []
        reasons = []
        
        for path in csv_files:
            schema = load_csv_schema(path)
            if not schema:
                continue
            # requires an llm reference — this tool needs special wiring
            # since BaseTool.run() doesn't currently receive llm; worth
            # discussing how to pass it through registry.get(...).run(...)
            checked.append(path.name)

            plan = plan_operation(query, schema, self.llm)
            if not plan or not plan.get("applicable"):
                 reason = (plan or {}).get("reason", "not a computable aggregation from this file")
                 reasons.append(f"'{path.name}' (columns: {schema['columns']}): {reason}")
                 continue

            target = str(plan.get("target_column", "")).lower()
            group_by = str(plan.get("group_by") or "").lower()
            sort_by = str(plan.get("sort_by") or "").lower()
            return_cols = " ".join(plan.get("return_columns") or []).lower()
            
            column_words = set(re.findall(
                r'\w+',
                target + " " + group_by + " " + sort_by + " " + return_cols.replace('_', ' ')
            ))
            query_words_normalized = set(re.findall(r'\w+', query.lower().replace('_', ' ')))
            
            
            if not (query_words & column_words):
                display_col = target or sort_by or "unknown"
                print(f"      ⚠️ csv_analyzer: plan picked '{display_col}'/'{group_by}' which "
                      f"doesn't match any word in the question — treating as not applicable.")
                reasons.append(f"'{path.name}': matched column '{display_col}' didn't relate to the question")
                continue


            computed = execute_plan(path, plan)
            if computed:
                return ToolResult(
                    content=computed,
                    source=f"Document: {path.name}",
                    confidence=0.95,
                    metadata={"file": str(path.resolve()), "tool_used": "csv_analyzer"},
                )

            
                # No CSV in the folder had an applicable, computable answer for
                # this question — report that honestly instead of returning None.
        reason_detail = ("\n" + "\n".join(f"  - {r}" for r in reasons)) if reasons else ""
        
        return ToolResult(
            content=(
                    f"None of the available CSV files ({', '.join(checked) or 'none checked'}) "
                    f"contain a computable answer for this question {reason_detail} — either the relevant "
                    f"column doesn't exist, or the question isn't a simple aggregation."
            ),
            source="csv_analyzer: no_match",
            confidence=0.75,
            metadata={"tool_used": "csv_analyzer"},

        )

        return ToolResult
