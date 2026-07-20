import re
from pathlib import Path
from typing import Optional, Tuple, List

from config.settings import Settings
from utils.llm_client import LLMClient
from utils.identity_check import has_local_documents
from utils.text_cleanup import clean_answer_text
from utils.document_excerpt import extract_relevant_excerpt
from utils.csv_analyzer import is_csv, load_csv_schema, plan_operation, execute_plan
from tools.registry import ToolRegistry
from patterns.selector import PatternSelector
from loop.engine import ResearchLoop
from loop.scratchpad import ResearchResult

_NO_INFO_PHRASES = (
    "does not contain", "not provided in", "cannot be found",
    "no information", "does not provide", "unable to find",
    "not found in", "cannot answer", "no relevant information",
    "not contain the information", "does not appear to contain",
)


def _looks_like_no_answer_fallback(text: str) -> bool:
    if not text:
        return True
    lowered = text.lower()
    return any(phrase in lowered for phrase in _NO_INFO_PHRASES)


class AgenticRAGResearcher:

    def __init__(self, settings: Settings = None):
        self.settings = settings or Settings()
        self.settings.validate()

        self.llm = LLMClient(
            api_key=self.settings.api_key,
            model=self.settings.model,
            max_tokens=self.settings.max_tokens,
        )
        self.registry = ToolRegistry()
        self.selector = PatternSelector(self.settings)
        self.loop = ResearchLoop(self.llm, self.registry, self.settings)

    def _answer_from_csv(self, query: str, path: Path) -> Tuple[ResearchResult, bool]:
        schema = load_csv_schema(path)
        if schema is None:
            result = ResearchResult(
                query=query,
                pattern_used="document_reader",
                final_answer=f"Could not load '{path.name}' as a CSV (pandas may not be installed, or the file is malformed).",
                sources=[f"Document: {path.name}"],
                confidence=0.1,
                hops=1,
                api_calls=1,
            )
            return result, False

        plan = plan_operation(query, schema, self.llm)

        if not plan or not plan.get("applicable"):
            reason = (plan or {}).get("reason", "the question doesn't map to a simple computable statistic from this dataset")
            result = ResearchResult(
                query=query,
                pattern_used="document_reader",
                final_answer=f"'{path.name}' doesn't appear to answer this question ({reason}).",
                sources=[f"Document: {path.name}"],
                confidence=0.2,
                hops=1,
                api_calls=1,
            )
            return result, False

        computed = execute_plan(path, plan)

        if computed is None:
            result = ResearchResult(
                query=query,
                pattern_used="document_reader",
                final_answer=f"Could not compute the requested statistic from '{path.name}'.",
                sources=[f"Document: {path.name}"],
                confidence=0.2,
                hops=1,
                api_calls=1,
            )
            return result, False

        synth_prompt = (
            f"QUESTION: {query}\n\n{computed}\n\n"
            f"Write a short, clear answer (1-3 sentences) to the question using "
            f"ONLY the exact computed numbers above. Do NOT recompute, round "
            f"differently, or estimate — state the given numbers exactly as-is. "
            f"Do NOT add any fact, caveat, or context not present above."
        )
        try:
            raw = self.llm.chat(
                system="You state exact, already-computed statistics clearly and concisely. You never recalculate or add outside information.",
                user=synth_prompt,
                max_tokens=300,
            )
            final_answer = clean_answer_text(raw)
        except Exception:
            final_answer = computed

        result = ResearchResult(
            query=query,
            pattern_used="document_reader",
            final_answer=final_answer,
            sources=[f"Document: {path.name} (computed via pandas)"],
            confidence=0.95,
            hops=1,
            api_calls=1,
        )
        return result, True

    def _answer_from_document(self, query: str, explicit_path: str) -> Tuple[ResearchResult, bool]:
        from tools.document_reader import DocumentReaderTool
        doc_tool = DocumentReaderTool()
        tool_result = doc_tool.run(query, file_path=explicit_path)

        excerpt = extract_relevant_excerpt(
            tool_result.content,
            query,
            window_chars=3000,
            max_total_chars=16000,
        )

        synthesis_prompt = (
            f"DOCUMENT CONTENT (excerpted from a longer document — [...] "
            f"marks skipped sections):\n{excerpt}\n\n"
            f"QUESTION: {query}\n\n"
            f"Respond in EXACTLY this format, nothing else:\n"
            f"FOUND: YES or NO\n"
            f"ANSWER: <your answer>\n\n"
            f"Rules for FOUND:\n"
            f"- Favor YES. Say YES if the excerpt discusses the same topic, "
            f"subject, or entity as the question — even if it only partially "
            f"answers it, or requires connecting a couple of details in the "
            f"excerpt. A partial or approximate answer from this document is "
            f"more useful than an unnecessary web search.\n"
            f"- Only say NO if the excerpt is genuinely about something else "
            f"entirely, with no meaningful connection to the question.\n\n"
            f"Rules for ANSWER:\n"
            f"- Every fact you state MUST be traceable to the excerpt above. "
            f"Do NOT fill in gaps using general knowledge, assumptions, or "
            f"anything not explicitly present in the excerpt — if a detail "
            f"isn't there, say it isn't there rather than guessing.\n"
            f"- Clear, direct prose. If FOUND is NO, briefly say why not.\n"
            f"- Do NOT dump raw document text, copyright notices, or "
            f"unrelated boilerplate into ANSWER.\n"
            f"- Do NOT write code, tags, or narrate a research process.\n"
        )

        found = True
        final_answer = ""

        try:
            raw = self.llm.chat(
                system="You answer questions using ONLY the provided document content — never general knowledge.",
                user=synthesis_prompt,
                max_tokens=1000,
            )

            found_match = re.search(r'FOUND:\s*(YES|NO)', raw, re.IGNORECASE)
            answer_match = re.search(r'ANSWER:\s*(.+)', raw, re.IGNORECASE | re.DOTALL)

            if found_match:
                found = found_match.group(1).upper() == "YES"
            else:
                found = not _looks_like_no_answer_fallback(raw)

            final_answer = answer_match.group(1).strip() if answer_match else raw
            final_answer = clean_answer_text(final_answer)

        except Exception:
            found = False
            final_answer = clean_answer_text(excerpt[:1500])

        result = ResearchResult(
            query=query,
            pattern_used="document_reader",
            final_answer=final_answer,
            sources=[tool_result.source],
            confidence=tool_result.confidence,
            hops=1,
            api_calls=1,
        )

        if not found:
            result.confidence = min(result.confidence, 0.3)

        return result, found

    def _try_all_documents(self, query: str) -> Tuple[Optional[ResearchResult], bool, List[str]]:
        docs_folder = Path("./documents")
        if not docs_folder.exists():
            return None, False, []

        files = sorted([f for f in docs_folder.glob("*") if f.is_file()])
        if not files:
            return None, False, []

        csv_files = [f for f in files if is_csv(f)]
        other_files = [f for f in files if not is_csv(f)]

        first_attempt = None
        checked_names: List[str] = []

        for f in csv_files:
            checked_names.append(f.name)
            try:
                result, found = self._answer_from_csv(query, f)
            except Exception:
                continue
            if found:
                return result, True, checked_names
            if first_attempt is None:
                first_attempt = result

        for f in other_files:
            checked_names.append(f.name)
            try:
                result, found = self._answer_from_document(query, str(f))
            except Exception:
                continue
            if found:
                return result, True, checked_names
            if first_attempt is None:
                first_attempt = result

        return first_attempt, False, checked_names

    def research(self, query: str, on_hop=None) -> ResearchResult:
        query = query.strip()
        if not query:
            raise ValueError("Query cannot be empty")

        is_path_query = re.search(r'[A-Za-z]:[\\\/].+\.(pdf|docx|txt|csv)', query, re.IGNORECASE)
        forced_pattern = getattr(self.settings, "forced_pattern", None)

        doc_result = None
        doc_found = False
        checked_names: List[str] = []

        if is_path_query:
            match = re.search(r'([A-Za-z]:[\\\/][^\s]+\.(?:pdf|docx|txt|csv))', query, re.IGNORECASE)
            explicit_path = match.group(1) if match else None
            if explicit_path:
                try:
                    path_obj = Path(explicit_path)
                    if is_csv(path_obj):
                        doc_result, doc_found = self._answer_from_csv(query, path_obj)
                    else:
                        doc_result, doc_found = self._answer_from_document(query, explicit_path)
                except Exception:
                    doc_result, doc_found = None, False

        elif has_local_documents():
            # Documents are checked first regardless of forced_pattern — a
            # reliable, exact local answer (especially for CSVs, computed
            # deterministically) should win over a forced pattern's web-based
            # guess. If a forced pattern was requested but a document
            # answers instead, that's surfaced transparently to the caller
            # via result.pattern_used == "document_reader", so the display
            # layer can show both what was requested and what was used.
            try:
                doc_result, doc_found, checked_names = self._try_all_documents(query)
            except Exception:
                doc_result, doc_found = None, False

        if doc_result is not None and doc_found:
            return doc_result

        pattern = self.selector.select_pattern(query)
        result = self.loop.run(query=query, pattern=pattern, on_hop=on_hop)

        if doc_result is not None and not doc_found:
            if len(checked_names) > 1:
                note = f"Local documents checked (no match): {', '.join(checked_names)}"
            elif len(checked_names) == 1:
                note = f"Document checked (no match): {checked_names[0]}"
            else:
                note = None
            if note:
                result.sources = [note] + result.sources

        result.final_answer = clean_answer_text(result.final_answer)

        return result

    def explain_pattern(self, query: str) -> str:
        return PatternSelector.explain(query)
