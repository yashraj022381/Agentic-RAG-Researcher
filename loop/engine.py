import re
import traceback
from typing import TYPE_CHECKING, Callable, Optional, Dict
from .scratchpad import Scratchpad, Step, ResearchResult
from tools.registry import ToolRegistry
from tools.base import ToolResult
from pathlib import Path
from utils.paths import DOCS_DIR
from utils.query_analysis import is_complex_query
from utils.text_cleanup import clean_answer_text
from patterns.base import AgentDecision

_NO_INFO_PHRASES = (
    "does not contain", "not provided in", "cannot be found",
    "no information", "does not provide", "unable to find",
    "not found in", "cannot answer", "no relevant information",
    "were not found", "was not found", "not contain the information",
    "no document matching",
)

# A reasonable floor signaling real multi-row iteration happened, not just
# a single computed row.
MIN_CALCULATOR_CALLS_FOR_MULTI_ROW = 3


def _content_looks_empty(text: str) -> bool:
    if not text:
        return True
    lowered = text.lower()
    return any(phrase in lowered for phrase in _NO_INFO_PHRASES)

if TYPE_CHECKING:
    from patterns.base import BasePattern


class ResearchLoop:
    def __init__(self, llm, registry, settings):
        self.settings = settings
        self.llm = llm
        self.registry = registry

    def _build_calculator_expression(self, query: str, scratchpad) -> str:
        """calculator needs a literal arithmetic expression, not natural
        language. First tries deterministic extraction scoped to the
        paragraph discussing 'competitor', then falls back to a small LLM
        call. ALWAYS returns a non-None string — never lets calculator
        receive raw natural language, which it can't parse."""
        relevant_steps = [
            s for s in scratchpad.steps
            if s.tool_used in ("document_reader", "web_search") and s.observation
        ]
        combined = "\n".join(s.observation[:4000] for s in relevant_steps)[:8000]
        money_re = re.compile(r'\$\s?([\d,]+\.?\d*)\s*(million|billion)?', re.IGNORECASE)

        def _parse(matches):
            vals = []
            for value_str, scale in matches:
                try:
                    val = float(value_str.replace(",", ""))
                    if scale and scale.lower() == "billion":
                        val *= 1000
                    vals.append(val)
                except ValueError:
                    continue
            return vals

        section_match = re.search(r'competitor', combined, re.IGNORECASE)
        if section_match:
            start = max(0, section_match.start() - 100)
            window = combined[start:start + 1200]
            nums = _parse(money_re.findall(window))
            print(f"      [DEBUG] Calculator: dollar figures near 'competitor': {nums[:4]}")
            if len(nums) >= 2:
                expr = f"{nums[0]} - {nums[1]}"
                print(f"      [DEBUG] Calculator expression (section-scoped): {expr!r}")
                return expr

        print("      [DEBUG] No competitor section with 2+ dollar figures found — falling back to LLM.")
        prompt = (
            f"Based on the findings below, write ONLY a single arithmetic expression "
            f"(using just numbers and + - * / ( ), no units, no words, no explanation) "
            f"that computes what the question is asking for.\n\n"
            f"QUESTION: {query}\n\nFINDINGS:\n{combined[:3000]}\n\nExpression:"
        )
        try:
            raw = self.llm.chat(
                system="You output ONLY a valid arithmetic expression using digits and + - * / ( ). Nothing else.",
                user=prompt,
                max_tokens=100,
                purpose="calculator_expression",
            )
            cleaned = re.sub(r'[^0-9+\-*/().\s]', '', raw).strip()
            if cleaned and any(c.isdigit() for c in cleaned):
                print(f"      [DEBUG] Calculator expression (LLM fallback): {cleaned!r}")
                return cleaned
        except Exception as e:
            print(f"      [DEBUG] Calculator expression LLM fallback also failed: {e}")

        print("      [DEBUG] Calculator expression: no usable expression found — using '0'.")
        return "0"

    def _build_web_search_query(self, query: str, scratchpad) -> str:
        """web_search needs a short, specific query — not the raw original
        question, which often contains meta-instructions ('search the web
        to find...') that pollute the actual search intent. If a prior
        document_reader step already surfaced a specific named entity (e.g.
        a competitor's company name), search for THAT plus the core topic."""
        print("      [DEBUG] _build_web_search_query version: v52-competitor-anchor")
        entity = None
        for s in reversed(scratchpad.steps):
            if s.tool_used == "document_reader" and s.observation:
                m = re.search(
                    r'competitor:?\s*\n?\s*([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){1,3})',
                    s.observation, re.IGNORECASE,
                )
                if m:
                    entity = m.group(1).strip()
                    break
               
        year_match = re.search(r'\b20(2[4-9]|3\d)\b', query)
        year = year_match.group(0) if year_match else ""
        topic_words = [w for w in ["revenue", "earnings", "results"] if w in query.lower()]

        if entity:
            built_query = f"{entity} {year} {' '.join(topic_words) or 'revenue'}".strip()
        else:
             web_intent_match = re.search(
                 r'search (?:the )?web for (.{5,60}?)(?:\.|,|$)', query, re.IGNORECASE
             )
             if web_intent_match:
                 built_query = web_intent_match.group(1).strip()
             else:
                  stripped = re.sub(
                      r'\b(search the web|find|search|to|and|calculate|our|read|identify|extract)\b',
                      '', query, flags=re.IGNORECASE,
                  )
                  built_query = " ".join(stripped.split())[:80]

        print(f"      [DEBUG] _build_web_search_query built: {built_query!r}")
        return built_query

    def _required_tools_satisfied(self, required_tools, tools_tried, scratchpad, plan) -> bool:
        """Whether every tool the plan calls for has been sufficiently
        attempted. Defined once as a real method (not a nested closure
        re-defined mid-loop) so it's safe to call from anywhere in run(),
        including before the loop body has executed once."""
        if not required_tools:
            return True
        needs_multi_row = bool(plan and plan.get("needs_multi_row_computation"))
        calculator_calls = sum(1 for s in scratchpad.steps if s.tool_used == "calculator")
        has_vectorized_multi_row_result = any(
            "VERIFIED COMPUTED RESULT" in (s.observation or "") and "for every row" in (s.observation or "")
            for s in scratchpad.steps
        )
        
        already_computed_elsewhere = any(
            "VERIFIED COMPUTED RESULT" in (s.observation or "") for s in scratchpad.steps
        )
        for t in required_tools:
            if t == "calculator":
                if needs_multi_row and not has_vectorized_multi_row_result:
                    if calculator_calls < MIN_CALCULATOR_CALLS_FOR_MULTI_ROW:
                        return False
                elif calculator_calls == 0 and not already_computed_elsewhere:
                    return False
            elif t not in tools_tried:
                return False
        return True

    def _minimal_synthesis_attempt(self, query: str, scratchpad) -> Optional[str]:
        """When the pattern's full synthesis prompt fails (often correlated
        with a large, complex prompt on a reasoning model), try once more
        with a drastically simpler prompt and lower temperature — much less
        likely to trigger an empty completion, even if the result is less
        polished than the full synthesis would have been."""
        calc_steps = [s for s in scratchpad.steps if s.tool_used == "calculator" and s.observation]
        calc_note = f"\nComputed result: {calc_steps[-1].observation}\n" if calc_steps else ""
        context = scratchpad.all_observations(max_chars=3000, per_step_chars=1000)

        prompt = (
            f"Answer this question in 2-4 short sentences, using ONLY the facts below. "
            f"Do not invent anything not shown here.\n\n"
            f"QUESTION: {query}\n\nFACTS:\n{context}\n{calc_note}\n"
            f"Write the answer as plain text now, nothing else:"
        )
        try:
            raw = self.llm.chat(
                system="Answer directly and briefly using only the given facts. Plain text only.",
                user=prompt,
                max_tokens=500,
                purpose="minimal_synthesis_retry",
            )
            text = (raw or "").strip()
            if text and not text.startswith("{") and len(text) > 20:
                return text
        except Exception as e:
            print(f"      [DEBUG] Minimal synthesis retry also failed: {e}")
        return None
    

    def run(self, query: str, pattern: "BasePattern", on_hop=None, plan=None) -> ResearchResult:#on_hop: Optional[Callable] = None, plan: Optional[dict] = None) -> ResearchResult:

        print("      [DEBUG] engine.py version: v23-finish-gate-fix")
        scratchpad = Scratchpad(query)
        self.llm.call_count = 0

        sources = []
        api_calls = 0
        tool_result = None
        pinned_document_path = None

        tools_tried = set()
        query_is_complex = is_complex_query(query)

        max_hops = getattr(self.settings, "max_hops", 6)
        min_hops = getattr(self.settings, "min_hops", 1)

        if plan and isinstance(plan.get("estimated_hops"), int):
            effective_min_hops = max(1, plan["estimated_hops"])
        else:
            effective_min_hops = 2 if query_is_complex else min_hops

        gather_tools_required = []
        if plan and plan.get("needs_document"):
            gather_tools_required.append("document_reader")
        if (plan and plan.get("needs_data")) or any(w in query.lower() for w in ["csv", "dataset", ".csv"]):
            gather_tools_required.append("csv_analyzer")
        if plan and plan.get("needs_web"):
            gather_tools_required.append("web_search")

        computation_required = bool(plan and plan.get("needs_computation"))
        required_tools = gather_tools_required + (["calculator"] if computation_required else [])

        for hop in range(1, max_hops + 1):
            try:
                agent_response = pattern.think(
                    query=query,
                    scratchpad=scratchpad,
                    llm=self.llm,
                    registry=self.registry,
                    plan=plan,
                )
            except Exception as e:
                import traceback
                print(f"      [DEBUG] ❌ pattern.think() crashed on hop {hop}: {e}")
                traceback.print_exc()
                if hop >= min_hops:
                    break
                agent_response = AgentDecision(
                    thought=f"LLM call failed on hop {hop}: {e}",
                    tool_name="",
                    tool_input="",
                    is_final=True,
                )
                break

            # ---- SINGLE, consolidated FINISH-handling block. ----
            if getattr(agent_response, 'is_final', False) and hop >= effective_min_hops: #and hop >= effective_min_hops:

                if not scratchpad.steps:
                    # An agent can't have "enough information" from zero tool calls,
                    # regardless of what the model's raw response said. Never honor
                    # FINISH before at least one real research step has happened.
                    print("      ⚠️ Model said FINISH before any tool was called — "
                          "ignoring and forcing a real research step instead.")
                    agent_response.is_final = False
                    if gather_tools_required:
                        fallback_tool = gather_tools_required[0]
                    elif any(w in query.lower() for w in ["csv", "dataset", ".csv"]):
                        fallback_tool = "csv_analyzer"
                    else:
                        fallback_tool = "document_reader" if any(
                            f.is_file() and f.suffix.lower() in {".pdf", ".docx", ".txt", ".md", ".csv"}
                            for f in Path(DOCS_DIR).rglob("*")
                        ) else "web_search"
                        
                    agent_response.tool_name = fallback_tool
                    agent_response.tool_input = query

                elif not self._required_tools_satisfied(required_tools, tools_tried, scratchpad, plan):
                    not_yet_tried = [t for t in required_tools if t not in tools_tried]
                    next_tool = not_yet_tried[0] if not_yet_tried else "web_search"
                    print(f"      ⚠️ Model wants to FINISH before the minimum of {effective_min_hops} "
                          f"hop(s) — forcing '{next_tool}' instead.")
                    agent_response.is_final = False
                    agent_response.tool_name = next_tool
                    agent_response.tool_input = query
                    
                elif query_is_complex and len(tools_tried) < 2 and hop < max_hops:
                    print(f"      ⚠️ Model requested FINISH after only {tools_tried or 'no'} tool(s) on a multi-part query — forcing a web_search hop instead.")
                    agent_response.is_final = False
                    agent_response.tool_name = "web_search"
                    agent_response.tool_input = query
                else:
                    break

            original_tool_name = getattr(agent_response, 'tool_name', "web_search")
            tool_name = original_tool_name
            override_tool_input = None

            last_failed_tools = {
                s.tool_used for s in scratchpad.steps
                if "no_match" in (s.observation or "").lower() or "none of the available" in (s.observation or "").lower()
            }

            # If the previous hop was flagged by CRAG as needing a web
            # correction, force web_search now — this is what actually
            # performs the correction, as its own real hop.
            #crag_correction_forced = False
            #if any(getattr(s, 'needs_correction', False) for s in scratchpad.steps) and "web_search" not in tools_tried:
            #    print(f"      ⚠️ Prior retrieval flagged for CRAG correction — forcing web_search this hop.")
            #    tool_name = "web_search"
            #    crag_correction_forced = True

            crag_correction_forced = False
            if any(getattr(s, 'needs_correction', False) for s in scratchpad.steps) and "web_search" not in tools_tried:
                print("      ⚠️ Prior retrieval flagged for CRAG correction — forcing web_search this hop.")
                tool_name = "web_search"
                crag_correction_forced = True

            calculation_forced = False
            if any(getattr(s, 'needs_calculation', False) for s in scratchpad.steps) and "calculator" not in tools_tried:
                print("      ⚠️ Prior retrieval flagged that a calculation is needed — forcing calculator this hop.")
                tool_name = "calculator"
                override_tool_input = self._build_calculator_expression(query, scratchpad)
                calculation_forced = True



            if not crag_correction_forced and not calculation_forced:
                if gather_tools_required:
                    ...
                elif computation_required and "calculator" not in tools_tried and tool_name != "calculator":
                    already_computed_elsewhere = any("VERIFIED COMPUTED RESULT" in (s.observation or "") for s in scratchpad.steps)
                    has_vectorized_multi_row_result = any(
                        "VERIFIED COMPUTED RESULT" in (s.observation or "") and "for every row" in (s.observation or "")
                        for s in scratchpad.steps
                    )
                    needs_multi_row = bool(plan and plan.get("needs_multi_row_computation"))
                    if already_computed_elsewhere and (not needs_multi_row or has_vectorized_multi_row_result):
                        pass
                    else:
                        print("      ⚠️ Gather tools complete — plan requires a calculation; switching to 'calculator'.")
                        tool_name = "calculator"
                        override_tool_input = self._build_calculator_expression(query, scratchpad)
                
            if tool_name in last_failed_tools and tool_name in tools_tried:
                if plan and plan.get("needs_web"):
                    print(f"      ⚠️ '{tool_name}' already returned no-match — trying web_search instead.")
                    tool_name = "web_search"
                    if tool_name == "web_search" and (not tool_input or tool_input.strip() == query.strip()):
                        tool_input = self._build_web_search_query(query, scratchpad)
                else:
                    print(f"      ✅ '{tool_name}' returned a confident no-match, and no web search is required — treating this as final.")

                
            if hop == 1 and tool_name not in ("document_reader", "csv_analyzer"):
                needs_local_content = bool(plan and (plan.get("needs_document") or plan.get("needs_data")))
                if needs_local_content:
                    try:
                        has_docs = any(
                            f.is_file() and f.suffix.lower() in {".pdf", ".docx", ".txt", ".md", ".csv"}
                            for f in Path(DOCS_DIR).rglob("*")
                        )
                    except Exception:
                        has_docs = False
                    if has_docs:
                        preferred = "csv_analyzer" if (plan and plan.get("needs_data")) else "document_reader"
                        print(f"      ⚠️ Model chose '{tool_name}' on hop 1 despite local documents existing — overriding to '{preferred}'.")
                        tool_name = preferred

            if gather_tools_required:
                not_yet_tried = [t for t in gather_tools_required if t not in tools_tried]

                if not_yet_tried:
                    if tool_name not in not_yet_tried:
                        next_tool = not_yet_tried[0]
                        print(f"      ⚠️ Model chose '{tool_name}' but plan still needs {not_yet_tried} — switching to '{next_tool}'.")
                        print(f"      ⚠️ Model wants to FINISH, but plan still requires {not_yet_tried} — forcing '{next_tool}' instead.")
                        tool_name = next_tool
                        override_tool_input = None
                    elif tool_name in last_failed_tools and tool_name in tools_tried:
                        print(f"      ⚠️ '{tool_name}' already returned no-match — trying web_search instead.")
                        tool_name = "web_search"
                        if tool_name == "web_search" and (not tool_input or tool_input.strip() == query.strip()):
                            tool_input = self._build_web_search_query(query, scratchpad)
                else:
                    if tool_name in last_failed_tools and tool_name in tools_tried:
                        print(f"      ⚠️ '{tool_name}' already returned no-match — trying web_search instead.")
                        tool_name = "web_search"
                        if tool_name == "web_search" and (not tool_input or tool_input.strip() == query.strip()):
                            tool_input = self._build_web_search_query(query, scratchpad)
                    elif computation_required and "calculator" not in tools_tried and tool_name != "calculator":
                        already_computed_elsewhere = any(
                            "VERIFIED COMPUTED RESULT" in (s.observation or "") for s in scratchpad.steps
                        )
                        needs_multi_row = bool(plan and plan.get("needs_multi_row_computation"))
                        if already_computed_elsewhere and not needs_multi_row:
                            pass
                        else:
                            print(f"      ⚠️ Gather tools complete — plan requires a calculation; switching to 'calculator'.")
                            tool_name = "calculator"
                            override_tool_input = self._build_calculator_expression(query, scratchpad)
            elif computation_required and "calculator" not in tools_tried and tool_name != "calculator":
                print(f"      ⚠️ Plan requires a calculation; switching to 'calculator'.")
                tool_name = "calculator"
                override_tool_input = self._build_calculator_expression(query, scratchpad)
            elif query_is_complex and tool_name in tools_tried and len(tools_tried) < 2:
                print(f"      ⚠️ Model chose '{tool_name}' again after already trying it — forcing web_search instead, since repeating won't surface new information.")
                tool_name = "web_search"
                if tool_name == "web_search" and (not tool_input or tool_input.strip() == query.strip()):
                    tool_input = self._build_web_search_query(query, scratchpad)

            resolved_tool = self.registry.get(tool_name)
            if resolved_tool is None:
                resolved_tool = self.registry.get("web_search")
               
            tool = resolved_tool
            tools_tried.add(tool_name)

            if override_tool_input is not None:
                tool_input = override_tool_input
            elif tool_name == "calculator":
                tool_input = self._build_calculator_expression(query, scratchpad)
            elif tool_name == "csv_analyzer":
                tool_input = query
            elif tool_name == "document_reader":
                tool_input = query
            elif tool_name != original_tool_name:
                if tool_name == "web_search":
                    tool_input = self._build_web_search_query(query, scratchpad)
                else:
                    tool_input = query
            else:
                tool_input = getattr(agent_response, 'tool_input', query)
                if not tool_input or not str(tool_input).strip():
                    tool_input = query

            print(f"      [DEBUG] About to call tool: tool_name='{tool_name}', resolved_tool={type(tool).__name__ if tool else None}, tool.name={getattr(tool, 'name', 'N/A')}")

            if (
                tool_name == "document_reader"
                and pinned_document_path
                and any(s.tool_used == "document_reader" and s.confidence >= 0.5 for s in scratchpad.steps)
            ):
                print("      ⚠️ document_reader already retrieved the pinned document with reasonable "
                      "confidence, and re-reads always use the same input — skipping the redundant "
                      "call and finishing with what's already gathered.")
                break

            
            # ---- SINGLE tool.run() call. If a document was already pinned
            # earlier in this run, reuse it rather than letting later hops
            # independently re-resolve _find_best_match against whatever
            # crafted input this hop's model call happened to produce —
            # which can flip to an unrelated file. ----
            try:
                run_kwargs = {"query": tool_input, "context": scratchpad.context_for_prompt()}
                if tool_name == "document_reader" and pinned_document_path:
                    run_kwargs["file_path"] = pinned_document_path
                new_tool_result = tool.run(**run_kwargs)

                if pinned_document_path and str(pinned_document_path).lower().endswith(".csv"):
                    gather_tools_required = [t for t in gather_tools_required if t != "document_reader"]

                if isinstance(new_tool_result, str):
                    new_tool_result = ToolResult(
                        content=new_tool_result,
                        source=f"{tool_name}: unwrapped",
                        confidence=0.5,
                    )
                while isinstance(getattr(new_tool_result, "content", None), ToolResult):
                    print(f"      ⚠️ '{tool_name}' returned a nested ToolResult as .content — unwrapping.")
                    new_tool_result = new_tool_result.content
                if not isinstance(new_tool_result.content, str):
                    new_tool_result.content = str(new_tool_result.content or "")


                if tool_name == "document_reader":
                    if _content_looks_empty(new_tool_result.content):
                        print(f"      ⚠️ document_reader succeeded but content indicates no answer found — downgrading confidence.")
                        new_tool_result.confidence = min(new_tool_result.confidence, 0.4)
                    elif pinned_document_path is None:
                        resolved_file = (new_tool_result.metadata or {}).get("file")
                        if resolved_file:
                            pinned_document_path = resolved_file
                            print(f"      [DEBUG] Pinned document for this research run: {resolved_file}")

            except Exception as e:
                import traceback
                print(f"      [DEBUG] ❌ tool.run() crashed on hop {hop}: {e}")
                new_tool_result = ToolResult(
                    content=f"Tool error: {str(e)}",
                    source=f"{tool_name}: error",
                    confidence=0.4,
                )

            prior_same_tool = [s for s in scratchpad.steps if s.tool_used == tool_name]
            is_duplicate = bool(prior_same_tool and prior_same_tool[-1].observation == new_tool_result.content)
            if is_duplicate:
                print(f"      ⚠️ '{tool_name}' returned identical content to its last attempt — no new information gained.")
                new_tool_result.content = (
                    f"[Repeated '{tool_name}' call — identical to a previous attempt, no new information found.]"
                )
                new_tool_result.confidence = min(new_tool_result.confidence, 0.3)

            tool_result = new_tool_result
            api_calls += 1

            try:
                tool_result = pattern.post_process(
                    tool_result=tool_result,
                    query=query,
                    scratchpad=scratchpad,
                    llm=self.llm,
                    registry=self.registry,
                )
            except Exception as e:
                print(f"      [DEBUG] ❌ post_process() crashed on hop {hop}: {e}")
                traceback.print_exc()

            try:
                step = Step(
                    hop_number=hop,
                    thought=getattr(agent_response, 'thought', ""),
                    tool_used=getattr(tool_result, 'metadata', {}).get("tool_used", tool_name),
                    tool_input=tool_input,
                    observation=tool_result.content,
                    confidence=getattr(tool_result, 'confidence', 0.5),
                    pattern=pattern.name,
                    grade=getattr(tool_result, 'metadata', {}).get("grade", 0.0),
                    corrected=getattr(tool_result, 'metadata', {}).get("corrected", False),
                    needs_correction=getattr(tool_result, 'metadata', {}).get("needs_correction", False),
                    needs_calculation=getattr(tool_result, 'metadata', {}).get("needs_calculation", False),  # ← new

                )
                scratchpad.add_step(step)
                print(f"      [DEBUG] Step recorded for hop {hop}. Total steps now: {len(scratchpad.steps)}")
                print(f"      [DEBUG] Step built with tool_used='{step.tool_used}'")

            except Exception as e:
                import traceback
                print(f"      [DEBUG] ❌❌❌ CRASH building/recording Step on hop {hop}: {e}")
                traceback.print_exc()
                raise

            if on_hop is not None:
                try:
                    on_hop(step)
                except Exception as e:
                    print(f"      [DEBUG] ⚠️ on_hop rendering failed for hop {step.hop_number}: {e}")

            _FAILURE_SOURCE_MARKERS = ("search-failed", "not-found", "empty-query", "folder empty", "rejected", "error")
            if tool_result.source and not any(m in tool_result.source for m in _FAILURE_SOURCE_MARKERS):
                sources.append(tool_result.source)

            # ---- Efficient-stop conditions. ----
            satisfied = self._required_tools_satisfied(required_tools, tools_tried, scratchpad, plan)

            if is_duplicate and hop >= effective_min_hops and satisfied:
                print(f"      ⚠️ Stopping early — '{tool_name}' produced no new information and minimum hops already met.")
                break

            if hop >= effective_min_hops and satisfied and getattr(tool_result, 'confidence', 0) >= 0.85:
                break

        print(f"      [DEBUG] Loop starting: max_hops={max_hops}, effective_min_hops={effective_min_hops}")
        print(f"      [DEBUG] Loop ended. scratchpad.hop_count={scratchpad.hop_count}")

        try:
            final_answer = pattern.synthesize(
                query=query,
                scratchpad=scratchpad,
                llm=self.llm,
            )
        except RuntimeError as e:
            print("      [DEBUG] engine.py exception-handling version: v53-clean-fallback")

            err_str = str(e).lower()
            if ("rate_limit" in err_str or "429" in err_str or "empty response" in err_str
                    or "too large" in err_str or "413" in err_str
                    or "tool_use_failed" in err_str or "tool choice is none" in err_str or "called a tool" in err_str):
                retry_answer = self._minimal_synthesis_attempt(query, scratchpad)

                if retry_answer:
                    final_answer = retry_answer
                else:
                    calc_steps = [s for s in scratchpad.steps if s.tool_used == "calculator" and s.observation]
                    calc_note = f"\n\nComputed result: {calc_steps[-1].observation}" if calc_steps else ""
                    raw_findings = scratchpad.all_observations()[:1200]
                    try:
                        raw_findings = clean_answer_text(raw_findings)
                    except Exception:
                        pass
                    #final_answer = (
                    #    f"Research completed through {scratchpad.hop_count} hop(s), but the "
                    #    f"final synthesis step failed ({str(e)[:150]}).{calc_note}\n\n"
                        #f"Here's what was found:\n{raw_findings}"
                        #f"Based on what was found: {scratchpad.all_observations()[:150]}.{calc_note}\n\n"
                    #    f"Raw findings gathered (not synthesized — treat as unverified):\n{raw_findings}"
                    #)
                    final_answer = (
                        f"I gathered the relevant information across {scratchpad.hop_count} step(s), "
                        f"but wasn't able to write a fully synthesized summary this time.{calc_note}\n\n"
                        f"Here's what was found:\n{raw_findings}"
                    )
            else:
                #if ("rate_limit" in err_str or "429" in err_str or "empty response" in err_str or "too large" in err_str or "413" in err_str or "tool_use_failed" in err_str or "tool choice is none" in err_str or "called a tool" in err_str):
                final_answer = f"Research completed. {str(e)}"
                
        except Exception as e:
            err_str = str(e).lower()
            if ("rate_limit" in err_str or "429" in err_str or "empty response" in err_str
                    or "too large" in err_str or "413" in err_str
                    or "tool_use_failed" in err_str or "tool choice is none" in err_str or "called a tool" in err_str):
                    
                retry_answer = self._minimal_synthesis_attempt(query, scratchpad)
                        
                if retry_answer:
                    final_answer = retry_answer
                else:
                    #"too large" in err_str or "413" in err_str or "reduce your message size" in err_str:
                    calc_steps = [s for s in scratchpad.steps if s.tool_used == "calculator" and s.observation]
                    calc_note = f"\n\nComputed result: {calc_steps[-1].observation}" if calc_steps else ""
                    raw_findings = scratchpad.all_observations()[:1200]
                    try:
                        raw_findings = clean_answer_text(raw_findings)
                    except Exception:
                        pass
                    #final_answer = (
                    #    f"Research completed through {scratchpad.hop_count} hop(s), but the final "
                    #    f"synthesis step failed because the combined findings exceeded the model's "
                        #f"Based on what was found: {scratchpad.all_observations()[:150]}.{calc_note}\n\n"
                    #    f"token limit.{calc_note}\n\n"
                    #    f"Raw findings gathered (not synthesized — treat as unverified):\n{raw_findings}"
                    #)
                    final_answer = (
                        f"I gathered the relevant information across {scratchpad.hop_count} step(s), "
                        f"but wasn't able to write a fully synthesized summary this time.{calc_note}\n\n"
                        f"Here's what was found:\n{raw_findings}"
                    )
            else:
                #if ("rate_limit" in err_str or "429" in err_str or "empty response" in err_str or "too large" in err_str or "413" in err_str or "tool_use_failed" in err_str or "tool choice is none" in err_str or "called a tool" in err_str):
                final_answer = f"Research completed. {str(e)}"

        if not final_answer or not final_answer.strip():
            final_answer = (
                "The research process gathered information, but the model "
                "did not produce a usable final answer this time. This can "
                "happen with an empty or malformed API response — try "
                "rephrasing the question or running it again."
            )

        sources = list(dict.fromkeys([s for s in sources if s]))
        

        return ResearchResult(
            query=query,
            final_answer=final_answer,
            pattern_used=pattern.name,
            tool_used=", ".join(dict.fromkeys(s.tool_used for s in scratchpad.steps)) if scratchpad.steps else None,
            confidence=scratchpad.last_confidence(),
            hops=scratchpad.hop_count,
            sources=sources,
            api_calls=api_calls,
            steps=scratchpad.steps,
        )
