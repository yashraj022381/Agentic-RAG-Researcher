import re
import time
from pathlib import Path
from typing import Optional, Tuple, List, Dict

from config.settings import Settings
from utils.llm_client import LLMClient
from utils.identity_check import has_local_documents
from utils.text_cleanup import clean_answer_text
from utils.document_excerpt import extract_relevant_excerpt
from utils.csv_analyzer import is_csv, load_csv_schema, plan_operation, execute_plan
from utils.cost_tracker import log_research_call
from utils.paths import DOCS_DIR
from utils.query_analysis import is_complex_query
from utils.query_planner import plan_query
from utils.taxonomy_classifier import classify_taxonomy, _needs_multi_row_computation
from tools.registry import ToolRegistry
from tools.document_reader import DocumentReaderTool
from patterns.selector import PatternSelector
from loop.engine import ResearchLoop
from loop.scratchpad import ResearchResult


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = str(PROJECT_ROOT / "documents")

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
        self.registry = ToolRegistry(llm=self.llm)
        self.selector = PatternSelector(self.settings)
        self.loop = ResearchLoop(self.llm, self.registry, self.settings)

    def _answer_from_csv(self, query: str, path: Path, plan: Optional[dict] = None): #-> Tuple[ResearchResult, bool]:
        try:
            schema = load_csv_schema(path)
            if not schema:
                return None, False



            #plan = plan_operation(query, schema, self.llm)
            csv_plan = plan_operation(query, schema, self.llm)   # ← renamed, no more collision
            pattern_used = (plan.get("pattern") if plan else None) or "react"       

            if csv_plan and csv_plan.get("applicable"):
                computed = execute_plan(path, csv_plan)
                if computed:
                    result = ResearchResult(
                        query=query,
                        final_answer=computed,
                        confidence=0.95,
                        sources=[f"Document: {path.name}"],
                        pattern_used="document_reader",
                        hops=1,
                    )
                    return result, True
            # plan looked applicable but execution failed (bad column, etc.)
                return None, False

        # Not a computable aggregation — check if the CSV is even topically
        # relevant before giving up, but never invent numbers here.
            prompt = (
                f"CSV columns: {schema['columns']}\n"
                f"Shape: {schema['shape'][0]} rows, {schema['shape'][1]} columns\n\n"
                f"QUESTION: {query}\n\n"
                f"This CSV cannot answer the question with a simple computed statistic. "
                f"Does it contain ANY relevant columns/data that could partially inform "
                f"an answer, even without doing exact computation? Respond in EXACTLY "
                f"this format:\nFOUND: YES or NO\nANSWER: <brief answer using only what "
                f"the columns/shape tell you — no invented numbers, no assumptions about "
                f"data not shown>"
            )
            response = self.llm.chat(
                system="You analyze whether CSV data is relevant to a question. Never invent numbers or data you haven't been shown.",
                user=prompt,
                max_tokens=400,
            )
            found = "FOUND: YES" in response.upper()
            if not found:
                return None, False

            answer = response.split("ANSWER:")[-1].strip() if "ANSWER:" in response else response
            result = ResearchResult(
                query=query,
                final_answer=answer,
                confidence=0.6,  # lower than the computed path — this is a qualitative answer, not verified
                sources=[f"Document: {path.name}"],
                pattern_used="document_reader",
                hops=1,
            )
            return result, True

        except Exception as e:
            print(f"      → CSV Error: {e}")
            return None, False
        

    def _answer_from_document(self, query: str, explicit_path: str, plan: Optional[dict] = None) -> Tuple[ResearchResult, bool]:
        doc_tool = DocumentReaderTool()
        tool_result = doc_tool.run(query, file_path=explicit_path)

        pattern_used = (plan.get("pattern") if plan else None) or "document_reader"

        full_content = tool_result.metadata.get("full_content", tool_result.content)

        excerpt = extract_relevant_excerpt(
            #tool_result.content,
            full_content,
            query,
            window_chars=3000,
            max_total_chars=16000,
        )


        print(f"      [DEBUG] Excerpt length: {len(excerpt)} chars")
        print(f"      [DEBUG] Excerpt preview: {excerpt[:500]}")

        synthesis_prompt = (
            f"DOCUMENT CONTENT (excerpted from a longer document — [...] "
            f"marks skipped sections):\n{excerpt}\n\n"
            f"QUESTION: {query}\n\n"
            f"Rules for FOUND:\n"
            f"- Say YES only if the document contains specific information that "
            f"actually helps answer THIS question — facts, numbers, names, or "
            f"details connected to what's being asked.\n"
            f"- Sharing a general subject area is NOT enough. A resume that mentions "
            f"'Python' as a skill does NOT answer a question about when Python the "
            f"language was created. A postmortem about this project does NOT answer "
            f"a general knowledge question just because it discusses similar topics.\n"
            f"- If the document is about a completely different topic (e.g. karate "
            f"when the question is about dark matter or transformers), you MUST say NO.\n"
            f"- Treat OCR text as valid document content even if incomplete or from only first pages.\n"
            f"- Prefer answering from the local document even if the answer is incomplete, "
            f"but only when the document genuinely addresses the question.\n"
            f"- If the question has multiple distinct parts, evaluate each part separately."
            f"A part with real support in the excerpt should be answered from it."
            f"A part with NO support in the excerpt must be explicitly flagged, e.g. The excerpt"
            f"does not include the [X] table/section, so I cannot confirm these values"
            f"from the document — do NOT substitute a different but similarly-named"
            f"variant (e.g. answering for a different model size) without saying so."
            f"- When in doubt, say NO — a wrong or irrelevant local-document answer is "
            f"worse than falling back to a normal search.\n\n"
            f"Respond in this exact format:\n"
            f"FOUND: YES or NO\n"
            f"ANSWER: <your answer here>\n\n"
            f"Rules for ANSWER:\n"
            f"- Every fact you state MUST be traceable to the excerpt above. "
            f"Do NOT fill in gaps using general knowledge, assumptions, or "
            f"anything not explicitly present in the excerpt — if a detail "
            f"isn't there, say it isn't there rather than guessing.\n"
            f"- Clear, direct prose. If FOUND is NO, briefly say why not.\n"
            f"- Do NOT dump raw document text, copyright notices, or "
            f"unrelated boilerplate into ANSWER.\n"
            f"- Do NOT write code, tags, or narrate a research process.\n"
            f"If a VERIFIED COMPUTED RESULT appears in the findings, you must state that exact number in your answer."
            
            
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
            pattern_used=pattern_used,
            final_answer=final_answer,
            sources=[tool_result.source],
            confidence=tool_result.confidence,
            hops=1,
            api_calls=1,
        )

        
        if "OCR" in (tool_result.content or ""):
            #found = True
            tool_result.confidence = max(tool_result.confidence, 0.78)

        if not found:
            result.confidence = min(result.confidence, 0.3)
        else:
            result.confidence = max(result.confidence, 0.70)

        return result, found

    def _try_all_documents(self, query: str,  plan: Optional[dict] = None): 
        
        reader = DocumentReaderTool(docs_folder=DOCS_DIR)

        available_files = [
            f for f in Path(DOCS_DIR).rglob("*")
            if f.is_file() and f.suffix.lower() in {".pdf", ".docx", ".txt", ".md", ".csv"} 
        ]
        
        if not available_files:
            return None, False, []

       

        first_attempt = None
        checked_names: List[str] = []
        
        
        query_lower = query.lower()

        is_data_question = any(word in query_lower for word in [
            "csv", "dataset", "passenger", "survival rate", "percentage of",
            "male", "female", "pclass", "average age", "total number",
            "how many rows", "how many columns", "training data", "test data",
        ]) or bool(plan and plan.get("needs_data"))
        #plan.setdefault("needs_document", False)
        #plan.setdefault("needs_web", False)

        if is_data_question:
            print("   → Detected data/analysis question. Prioritizing CSV files only.")
            
        
            csv_files = [f for f in available_files if f.suffix.lower() == ".csv"]
            for file_path in csv_files:
                checked_names.append(file_path.name)
                print(f"   Checking CSV: {file_path.name}")

           

                try:
                    result, found = self._answer_from_csv(query, file_path, plan=plan)
                    if found:
                        print(f"      → RELEVANT CSV found!")
                        return result, True, checked_names
                except Exception as e:
                    print(f"      → CSV error: {e}")

       
            print("   → No relevant CSV found.")
            return None, False, checked_names

        print("   → Checking normal documents...")
        other_files = [f for f in available_files if f.suffix.lower() != ".csv"]
        query_words = [w for w in re.findall(r'\w+', query.lower()) if len(w) > 3]

        scored = []

        for file_path in other_files:
            checked_names.append(file_path.name)
            print(f"   Checking: {file_path.name}")

       

            
            if "karate" in file_path.name.lower() and "karate" not in query_lower and "stance" not in query_lower:
                print(f"      → Skipped (unrelated to question)")
                continue 
    
            try:
                
                content = reader.quick_preview(Path(file_path))
                
                if not content or len(content.strip()) < 100:
                    print(f"      → Skipped (empty or too short)")
                    continue
            
                preview = content[:7000].lower()
               
                overlap = sum(1 for w in query_words if w in preview)

                if overlap < 5:
                    #scored.append((overlap, file_path))
                    print(f"   Skipping (low keyword overlap: {overlap})")
                    continue

                scored.append((overlap, file_path))

                
                #scored.append((overlap, file_path))
                print(f"      → Candidate (overlap={overlap})")

            except Exception as e:
                print(f"      → Error reading {file_path.name}: {e}")
                continue

                
        scored.sort(key=lambda x: -x[0])

        for overlap, file_path in scored[:2]:
            result, found = self._answer_from_document(query, str(file_path), plan=plan)


            if found:
                print(f"      → FOUND relevant content")
                return result, True, checked_names

            
            
        print("   → No relevant document found.")
        return None, False, checked_names
   


    def research(self, query: str, on_hop=None) -> ResearchResult:

        verification_words = [
            "verify", "confirm", "assuming", "premise",
            "strictly verify", "numerically verify", "not estimated", "not approximate",
        ]

        q_lower = query.lower()
        keyword_verification_signal = any(w in q_lower for w in verification_words)

        
        query = query.strip()
        if not query:
            raise ValueError("Query cannot be empty")

        # Reset usage counters ONCE per top-level query, before any LLM call
        # happens — including document/CSV checks, which make real LLM calls
        # of their own. Resetting later (e.g. inside the pattern loop) would
        # silently drop those tokens from the logged total for this query.
        if hasattr(self.llm, "reset_usage_counters"):
            self.llm.reset_usage_counters()

        start_time = time.time()

       
        forced_pattern = getattr(self.settings, "forced_pattern", None)
        #skip_precheck = bool(forced_pattern) or is_complex_query(query)
        doc_names = [f.name for f in Path(DOCS_DIR).rglob("*") if f.is_file() and f.suffix.lower() in {".pdf",".docx",".txt",".md",".csv"}]
        #plan = plan_query(query, self.llm, doc_names) if not forced_pattern else None
        taxonomy_match = classify_taxonomy(query, available_documents=doc_names) if not forced_pattern else None

        if taxonomy_match:
            plan = {
                "needs_document": taxonomy_match["needs_document"],
                "needs_data": taxonomy_match["needs_data"],
                "needs_web": taxonomy_match["needs_web"],
                "needs_both": taxonomy_match["needs_document"] and taxonomy_match["needs_web"],
                "needs_verification": taxonomy_match["pattern"] in ("crag", "selfrag"),
                "pattern": taxonomy_match["pattern"],
                "estimated_hops": 2 if (taxonomy_match["needs_web"] or taxonomy_match["needs_data"] and taxonomy_match["needs_web"]) else 1,
                "needs_computation": taxonomy_match.get("needs_computation", False),
                "needs_multi_row_computation": _needs_multi_row_computation(q_lower)
            }
            print(f"      🎯 Taxonomy match: {taxonomy_match['category']} → pattern={plan['pattern']}")
        else:
            doc_names = [f.name for f in Path(DOCS_DIR).rglob("*") if f.is_file() and f.suffix.lower() in {".pdf",".docx",".txt",".md",".csv"}]
            plan = plan_query(query, self.llm, doc_names) if not forced_pattern else None
        #skip_precheck = bool(forced_pattern) or (plan and plan.get("needs_both"))
        skip_precheck = (
            bool(forced_pattern)
            or (plan and (plan.get("needs_both") or plan.get("pattern") in ("crag", "selfrag")))
            or keyword_verification_signal
        )
        #skip_precheck = bool(forced_pattern) or (plan and (plan.get("needs_both") or plan.get("needs_verification")))
            
        doc_result = None
        doc_found = False

        # ---------- Local Document Check (controlled) ----------
        if not skip_precheck and has_local_documents(docs_folder=DOCS_DIR):
            print(f"🔍 Checking local documents for: {query}")
            try:
                doc_result, doc_found, _ = self._try_all_documents(query, plan=plan)
            except Exception as e:
                print(f"⚠️ Local document check failed: {e}")
                doc_result, doc_found = None, False

        # ---------- Decision Logic ----------
        if doc_result is not None and doc_found:
            # Only use local document when the synthesis step said FOUND = YES
            print("✅ Using local document (FOUND=YES)")
            result = doc_result
        else:
            # Fall back to normal agentic flow (web search / knowledge_base / patterns)
            print("➡️ No relevant local document found → using normal agent")
            
            extract_and_compare_signal = (
                any(w in q_lower for w in ["extract", "according to", "from the document", "from the pdf"])
                and any(w in q_lower for w in ["compare", "versus", " vs "])
            )
            if extract_and_compare_signal and not keyword_verification_signal:
                pattern = self.selector._build("react")
            else:
                pattern = self.selector.select_pattern(query, plan)  # pass the plan through

            result = self.loop.run(query=query, pattern=pattern, on_hop=on_hop, plan=plan)

        result.final_answer = clean_answer_text(result.final_answer)

        #result.final_answer = clean_answer_text(result.final_answer)

        if not result.final_answer or not result.final_answer.strip():
            result.final_answer = (
               "The research process completed, but no usable answer text was produced."
            )
            
        
        elapsed = time.time() - start_time

        try:
            total_tokens = (
                getattr(self.llm, "total_input_tokens", 0)
                + getattr(self.llm, "total_output_tokens", 0)
            )
            estimated_cost = getattr(self.llm, "total_cost_usd", 0.0)
            log_research_call(
                query=query,
                pattern_used=result.pattern_used,
                hops=result.hops,
                api_calls=result.api_calls,
                confidence=result.confidence,
                elapsed_seconds=elapsed,
                total_tokens=total_tokens,
                estimated_cost_usd=estimated_cost,
                sources_count=len(result.sources or []),
            )
        except Exception:
            pass

        return result

    def explain_pattern(self, query: str) -> str:
        return PatternSelector.explain(query)
