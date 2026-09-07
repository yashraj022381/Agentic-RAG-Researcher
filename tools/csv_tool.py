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

        def _split_column_name(name: str) -> str:
            s = name.replace('_', ' ').replace('-', ' ')
            s = re.sub(r'(?<!^)(?=[A-Z])', ' ', s)
            return s

        # If the query explicitly names specific CSV files, only check those —
        # don't let an unrelated file with a coincidentally-matching column
        # name (e.g. both have an "Age" column) get pulled into the answer.
        query_lower = query.lower()
        named_files = [f for f in csv_files if f.name.lower() in query_lower]
        files_to_check = named_files if named_files else csv_files

        query_words = set(w for w in re.findall(r'\w+', query.lower()) if len(w) > 3)
        checked, reasons, computed_by_file = [], [], {}

        for path in files_to_check:
            schema = load_csv_schema(path)
            if not schema:
                continue
            checked.append(path.name)

            plan = plan_operation(query, schema, self.llm)
            if not plan or not plan.get("applicable"):
                reason = (plan or {}).get("reason", "not a computable aggregation from this file")
                reasons.append(f"'{path.name}' (columns: {schema['columns']}): {reason}")
                continue

            target = str(plan.get("target_column", "")).lower()
            group_by = str(plan.get("group_by") or "").lower()
            sort_by = str(plan.get("sort_by") or "").lower()
            id_col = str(plan.get("id_column") or "")
            return_cols = plan.get("return_columns") or plan.get("columns") or []

            combined_names = " ".join([target, group_by, sort_by, id_col] + return_cols)
            column_words = set(re.findall(r'\w+', _split_column_name(combined_names).lower()))

            required_overlap = 2 if plan.get("op") == "derived_arithmetic" else 1
            if len(query_words & column_words) < required_overlap:
                display_col = target or sort_by or id_col or "unknown"
                print(f"      ⚠️ csv_analyzer: plan picked '{display_col}'/'{group_by}' which "
                      f"doesn't match any word in the question — treating as not applicable.")
                reasons.append(f"'{path.name}': matched column '{display_col}' didn't relate to the question")
                continue

            # IMPORTANT: this plan was built from THIS file's own schema — only
            # ever execute it against THIS file, never against another file in
            # the folder (that was the source of the cross-file contamination).
            computed = execute_plan(path, plan)
            if computed:
                computed_by_file[path.name] = computed
            else:
                reasons.append(f"'{path.name}': plan looked applicable but execution failed")

        if computed_by_file:
            if len(computed_by_file) == 1:
                name, content = next(iter(computed_by_file.items()))
                return ToolResult(
                    content=content,
                    source=f"CSV Analysis: {name}",
                    confidence=0.95,
                    metadata={"tool_used": "csv_analyzer"},
                )
            combined = "\n\n".join(f"[{name}]\n{content}" for name, content in computed_by_file.items())
            return ToolResult(
                content=combined,
                source=f"Documents: {', '.join(computed_by_file)}",
                confidence=0.95,
                metadata={"tool_used": "csv_analyzer"},
            )

        reason_detail = ("\n" + "\n".join(f"  - {r}" for r in reasons)) if reasons else ""
        return ToolResult(
            content=(
                f"None of the available CSV files ({', '.join(checked) or 'none checked'}) "
                f"contain a computable answer for this question{reason_detail} — either the relevant "
                f"column doesn't exist, or the question isn't a simple aggregation."
            ),
            source="csv_analyzer: no_match",
            confidence=0.75,
            metadata={"tool_used": "csv_analyzer"},
        )
