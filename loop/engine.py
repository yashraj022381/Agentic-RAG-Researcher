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

MIN_CALCULATOR_CALLS_FOR_MULTI_ROW = 3  # a reasonable floor signaling real iteration happened, not just one row

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
        import re
        combined = scratchpad.all_observations(max_chars=8000)
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

    def run(self, query: str, pattern: "BasePattern", on_hop: Optional[Callable] = None, plan: Optional[dict] = None) -> ResearchResult:

        scratchpad = Scratchpad(query)
        self.llm.call_count = 0

        sources = []
        api_calls = 0
        tool_result = None

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

            # ---- SINGLE, consolidated FINISH-handling block. This is the
            # only place in the loop that decides whether an early FINISH
            # is honored, overridden to force a still-required tool, or
            # overridden to force broader coverage on a complex query. ----
            if getattr(agent_response, 'is_final', False) and hop >= effective_min_hops:
                _required_tools_satisfied(not required_tools or all(t in tools_tried for t in required_tools))
                if not _required_tools_satisfied():
                    not_yet_tried = [t for t in required_tools if t not in tools_tried]
                    next_tool = not_yet_tried[0]
                    print(f"      ⚠️ Model wants to FINISH, but plan still requires {not_yet_tried} — forcing '{next_tool}' instead.")
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
            if tool_name in last_failed_tools and tool_name in tools_tried:
                if plan and plan.get("needs_web"):
                    print(f"      ⚠️ '{tool_name}' already returned no-match — trying web_search instead.")
                    tool_name = "web_search"
                else:
                    print(f"      ✅ '{tool_name}' returned a confident no-match, and no web search is required — treating this as final.")

            if hop == 1 and tool_name not in ("document_reader", "csv_analyzer"):
                try:
                    has_docs = any(
                        f.is_file() and f.suffix.lower() in {".pdf", ".docx", ".txt", ".md", ".csv"}
                        for f in Path(DOCS_DIR).rglob("*")
                    )
                except Exception:
                    has_docs = False
                if has_docs:
                    print(f"      ⚠️ Model chose '{tool_name}' on hop 1 despite local documents existing — overriding to document_reader.")
                    tool_name = "document_reader"

            if gather_tools_required:
                not_yet_tried = [t for t in gather_tools_required if t not in tools_tried]

                if not_yet_tried:
                    if tool_name not in not_yet_tried:
                        next_tool = not_yet_tried[0]
                        print(f"      ⚠️ Model chose '{tool_name}' but plan still needs {not_yet_tried} — switching to '{next_tool}'.")
                        tool_name = next_tool
                    elif tool_name in last_failed_tools and tool_name in tools_tried:
                        print(f"      ⚠️ '{tool_name}' already returned no-match — trying web_search instead.")
                        tool_name = "web_search"
                else:
                    if tool_name in last_failed_tools and tool_name in tools_tried:
                        print(f"      ⚠️ '{tool_name}' already returned no-match — trying web_search instead.")
                        tool_name = "web_search"
                    elif computation_required and "calculator" not in tools_tried and tool_name != "calculator":
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

            resolved_tool = self.registry.get(tool_name)
            if resolved_tool is None:
                resolved_tool = self.registry.get("web_search")
                tool_name = "web_search"
            tool = resolved_tool
            tools_tried.add(tool_name)

            if override_tool_input is not None:
                tool_input = override_tool_input
            elif tool_name == "csv_analyzer":
                tool_input = query
            elif tool_name != original_tool_name:
                tool_input = query
            else:
                tool_input = getattr(agent_response, 'tool_input', query)
                if not tool_input or not str(tool_input).strip():
                    tool_input = query

            print(f"      [DEBUG] About to call tool: tool_name='{tool_name}', resolved_tool={type(tool).__name__ if tool else None}, tool.name={getattr(tool, 'name', 'N/A')}")

            try:
                new_tool_result = tool.run(
                    query=tool_input,
                    context=scratchpad.context_for_prompt(),
                )
                if tool_name == "document_reader" and _content_looks_empty(new_tool_result.content):
                    print(f"      ⚠️ document_reader succeeded but content indicates no answer found — downgrading confidence.")
                    new_tool_result.confidence = min(new_tool_result.confidence, 0.4)
            except Exception as e:
                import traceback
                print(f"      [DEBUG] ❌ tool.run() crashed on hop {hop}: {e}")
                traceback.print_exc()
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
                import traceback
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

            # ---- Efficient-stop conditions, restored. Both require every
            # required tool to have been attempted first. ----
            #required_tools_satisfied = not required_tools or all(t in tools_tried for t in required_tools)

            calculator_calls = sum(1 for s in scratchpad.steps if s.tool_used == "calculator")
            needs_multi_row = bool(plan and plan.get("needs_multi_row_computation"))

            def _required_tools_satisfied():
                if not required_tools:
                    return True
                for t in required_tools:
                    if t == "calculator" and needs_multi_row:
                        if calculator_calls < MIN_CALCULATOR_CALLS_FOR_MULTI_ROW:
                            return False
                    elif t not in tools_tried:
                        return False
                return True
            if is_duplicate and hop >= effective_min_hops and _required_tools_satisfied():
                print(f"      ⚠️ Stopping early — '{tool_name}' produced no new information and minimum hops already met.")
                break

            if hop >= effective_min_hops and _required_tools_satisfied() and getattr(tool_result, 'confidence', 0) >= 0.85:
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
            err_str = str(e).lower()
            if "rate_limit" in err_str or "429" in err_str or "empty response" in err_str or "too large" in err_str or "413" in err_str:
                calc_steps = [s for s in scratchpad.steps if s.tool_used == "calculator" and s.observation]
                calc_note = f"\n\nComputed result: {calc_steps[-1].observation}" if calc_steps else ""
                final_answer = (
                    f"Research completed through {scratchpad.hop_count} hop(s), but the "
                    f"final synthesis step failed ({str(e)[:150]}).{calc_note}\n\n"
                    f"Based on what was found: {scratchpad.all_observations()[:1200]}"
                )
            else:
                final_answer = f"Research completed. {str(e)}"
        except Exception as e:
            err_str = str(e).lower()
            if "too large" in err_str or "413" in err_str or "reduce your message size" in err_str:
                calc_steps = [s for s in scratchpad.steps if s.tool_used == "calculator" and s.observation]
                calc_note = f"\n\nComputed result: {calc_steps[-1].observation}" if calc_steps else ""
                final_answer = (
                    f"Research completed through {scratchpad.hop_count} hop(s), but the final "
                    f"synthesis step failed because the combined findings exceeded the model's "
                    f"token limit.{calc_note}\n\nBased on what was found: {scratchpad.all_observations()[:1200]}"
                )
            else:
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
            tool_used=scratchpad.steps[-1].tool_used if scratchpad.steps else None,
            confidence=scratchpad.last_confidence(),
            hops=scratchpad.hop_count,
            sources=sources,
            api_calls=api_calls,
            steps=scratchpad.steps,
        )
