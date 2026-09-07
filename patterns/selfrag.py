import re
import json
from typing import TYPE_CHECKING
from .base import BasePattern, AgentDecision #_check_verified_computed_result, _apply_prior_correction_wrap, _is_trivially_empty 
from utils.parser import ResponseParser
from utils.known_facts import get_known_fact_note

if TYPE_CHECKING:
    from loop.scratchpad import Scratchpad
    from tools.base import ToolResult
    from tools.registry import ToolRegistry
    from utils.llm_client import LLMClient
    from tools.base import ToolResult


RELEVANCE_GRADER_PROMPT = """
You are a strict retrieval quality grader.

Question: {query}
Retrieved chunk: {content}

Score how RELEVANT and USEFUL this chunk is for answering the question.
Score from 0.0 (totally irrelevant) to 1.0 (perfect answer).
Respond with ONLY a decimal number. Nothing else.
"""

SELF_REFLECT_PROMPT = """
You are evaluating your own progress.

Original question: {query}
Information gathered so far: {observations}

On a scale of 0.0 to 1.0
- 0.0 = I have almost no useful information
- 0.5 = I have partial information
- 1.0 = I can fully answer the question

Respond with ONLY a decimal number.
"""

class SelfRAGPattern(BasePattern):

    def __init__(self, relevance_threshold: float = 0.30,
                 max_retries: int = 2):
        self.relevance_threshold = relevance_threshold
        self.max_retries = max_retries

    @property
    def name(self) -> str:
        return "selfrag"

    @property
    def system_prompt(self) -> str:
        return (
            "You are a Self-RAG research agent. "
            "After every retrieval you grade its relevance. "
            "If relevance is low, you re-query with improved search terms. "
            "You also self-reflect on whether your accumulated findings "
            "are sufficient to answer the question."
        )

    def think(self, query, scratchpad, llm, registry, plan=None) -> AgentDecision:
        extra = (
            "SELF-RAG RULES:\n"
            "1. After getting a result you will grade it (handled automatically).\n"
            "2. If this is a retry, choose DIFFERENT search terms.\n"
            "3. Think about whether retrieved chunks actually answer the question.\n"
            "4. Call FINISH only when self-reflection score is high.\n"
            f"NEVER propose a hypothetical mapping between unrelated data and what the "
            f"question asked (e.g. 'if we assume X is related to Y') — if the specific "
            f"data requested doesn't exist, say so plainly and stop there.\n"
        )
        prompt = self._build_think_prompt(query, scratchpad, registry, extra)
        raw = llm.chat(system=self.system_prompt, user=prompt, max_tokens=2048, purpose="think")
        return self._parse_decision(raw, scratchpad)

    def post_process(self, tool_result, query, scratchpad, llm, registry, tool_name=None):
        
        if self._check_verified_computed_result(tool_result, "selfrag_status"):
            return tool_result

        if self._apply_prior_correction_wrap(tool_result, scratchpad, "selfrag_status", "Web correction"):
            return tool_result

        low_grade_steps = sum(1 for s in scratchpad.steps if s.grade < self.relevance_threshold)

        grade = 0.0 if self._is_trivially_empty(tool_result.content) else llm.grade(
            RELEVANCE_GRADER_PROMPT.format(query=query, content=tool_result.content.strip()[:800])
        )
        tool_result.metadata["grade"] = grade

        if self._maybe_flag_web_fallback(
            tool_result, scratchpad, grade, "selfrag_status",
            threshold=self.relevance_threshold, max_retries=self.max_retries,
        ):
            return tool_result

        # react.py, crag.py, selfrag.py — add this line in post_process, e.g.:
        self._maybe_flag_calculation_needed(tool_result, scratchpad, query, "selfrag_status")  # or "crag_status"/"selfrag_status"
        """
        if self._is_trivially_empty(tool_result.content):
            if low_grade_steps < self.max_retries and tool_result.metadata.get("tool_used") != "web_search":
                tool_result.metadata["grade"] = 0.0
                tool_result.metadata["needs_correction"] = True
                tool_result.metadata["corrected"] = True
                tool_result.metadata["selfrag_status"] = "needs_web_fallback"
                tool_result.confidence = 0.2
            return tool_result
    
        #return tool_result
        if "VERIFIED COMPUTED RESULT" in (tool_result.content or ""):
            tool_result.metadata["selfrag_status"] = "verified_computed"
            tool_result.metadata["grade"] = 1.0
            tool_result.confidence = max(tool_result.confidence, 0.95)
            return tool_result
        
        content = tool_result.content.strip()
        grade_prompt = RELEVANCE_GRADER_PROMPT.format(query=query, content=content[:800])
        grade = llm.grade(grade_prompt)
        tool_result.metadata["grade"] = grade

        if (grade < self.relevance_threshold
        and low_grade_steps < self.max_retries
        and tool_result.metadata.get("tool_used") != "web_search"):
            tool_result.metadata["needs_correction"] = True
            tool_result.metadata["corrected"] = True
            tool_result.metadata["selfrag_status"] = "needs_web_fallback"
            tool_result.confidence = grade
            return tool_result


        if (grade < self.relevance_threshold
            and low_grade_steps < self.max_retries
            and tool_result.metadata.get("tool_used") != "web_search"):
            tool_result.metadata["needs_correction"] = True
            tool_result.metadata["corrected"] = True
            tool_result.metadata["selfrag_status"] = "needs_web_fallback"
            tool_result.confidence = grade
            return tool_result

        grade_prompt = RELEVANCE_GRADER_PROMPT.format(
            query=query,
            content=content[:800],
        )
        grade = llm.grade(grade_prompt)
        tool_result.metadata["grade"] = grade
        """
        
        tool_result.confidence = max(tool_result.confidence, grade, 0.5)
        tool_result.metadata["selfrag_status"] = "graded"
        return tool_result


    def synthesize(self, query, scratchpad, llm) -> str:

        all_obs = scratchpad.all_observations(max_chars=6000, per_step_chars=2000)

        if not all_obs.strip() or len(all_obs) < 80:
            return (
                "I was unable to find sufficient information to answer this question."
            )
        reflect_score = llm.grade(
            SELF_REFLECT_PROMPT.format(
                query=query,
                observations=scratchpad.all_observations(max_chars=1500, per_step_chars=750),
            )
        )

        if scratchpad.steps:
            scratchpad.steps[-1].confidence = min(scratchpad.steps[-1].confidence, reflect_score)

        context = all_obs
        known_fact_note = get_known_fact_note(query)
        
        calc_steps = [s for s in scratchpad.steps if s.tool_used == "calculator" and s.observation]
        calc_rule = ""
        if calc_steps:
            calc_results = "\n".join(f"- {s.tool_input} → {s.observation}" for s in calc_steps)
            calc_rule = (
                f"\nCALCULATOR RESULTS — AUTHORITATIVE, DO NOT RECOMPUTE:\n{calc_results}\n"
                f"A calculator tool already computed the exact value(s) needed for this "
                f"answer. You MUST use these exact result(s) verbatim in your answer — "
                f"do NOT perform your own alternate arithmetic, do NOT substitute "
                f"different numbers than what the calculator used, and do NOT derive a "
                f"different percentage or variance using different inputs. If the "
                f"calculator's expression doesn't seem to match what you'd expect, still "
                f"report its exact result and note the expression it used — never quietly "
                f"replace it with your own calculation.\n"
                f"Include this exact line as the FIRST line of your answer, before anything else: "
                f"'Self-reflection score: {reflect_score:.0%}'\n\n"
                f"{calc_rule}\n"
            )
            
        prompt = (
            f"GROUNDING RULE — CRITICAL, READ FIRST:\n"
            f"Every specific number, name, fact, PAGE NUMBER, or CHAPTER REFERENCE you "
            f"state MUST come from the FINDINGS below, verbatim or clearly implied — "
            f"never invented. Do NOT fill gaps using your own general knowledge, "
            f"training data, or assumptions, even if you're confident you know the "
            f"answer from elsewhere. This applies especially to citations like "
            f"'(page 57)' or 'Chapter 2' — if the FINDINGS don't literally show that "
            f"page number or chapter content, do NOT state it, even if it sounds "
            f"plausible for the source. If the FINDINGS don't contain a specific "
            f"value the question asks for, say so explicitly rather than supplying "
            f"yourself.\n\n"
            f"Using these research findings, write a complete answer to:\n"
            f"'{query}'\n\n"
            f"FINDINGS:\n{context}\n"
            f"{known_fact_note}\n"
            f"CRITICAL RULES:\n"
            f"1. If the question compares TWO things (e.g. 'who started first'),\n"
            f"   you MUST have explicit dates/facts for BOTH things before\n"
            f"   making any comparative claim.\n"
            f"2. If you only have information about ONE side of the comparison,\n"
            f"   clearly state 'I have information about X but not Y, so I\n"
            f"   cannot determine which came first' — do NOT guess.\n"
            f"3. Never state a comparative conclusion (e.g. 'X was first')\n"
            f"   unless you have verified data for ALL items being compared.\n\n"
            f"CRITICAL DISAMBIGUATION RULE:\n"
            f"If the FINDINGS describe an ORGANIZATION, COMPANY, or FILM/BRAND that "
            f"shares a name with a person the question asks about, do NOT treat the "
            f"organization's attributes as if they belong to a person. State clearly "
            f"that the name matches an organization, not an individual, and that no "
            f"information about the specific person was found.\n"
            f"Similarly, if FINDINGS describe MULTIPLE DIFFERENT people who share the "
            f"same name, do NOT merge their attributes into one answer. State that the "
            f"name is ambiguous and specify what is known about each distinct entity "
            f"separately, or say the exact person could not be identified.\n\n"
            f"FORMAT RULES (read carefully — these matter as much as the content):\n"
            f"- Write clear, natural prose in 2-4 short paragraphs. Roughly 150-300 words "
            f"total unless the question genuinely requires more detail.\n"
            f"- If the question asks for verbatim quotes, exact citations, or a specific "
            f"list, put THOSE first in your answer, before general explanation — that way "
            f"if space runs short, the explicitly-requested content is never the part cut off.\n"
            f"- IMPORTANT: being concise means cutting narration, repetition, and preamble "
            f"— it does NOT mean cutting specific facts. ALWAYS include exact dates, "
            f"numbers, names, and figures from the FINDINGS when they're part of the answer.\n"
            f"- Do NOT break your answer into a numbered list of sub-questions, and do NOT "
            f"narrate your reasoning process. Just answer.\n"
            f"- Do NOT cite a source after every single sentence. Mention a source naturally "
            f"at most once or twice total.\n"
            f"- Be critical — only include information that truly answers the question. "
            f"If self-reflection < 0.6, briefly note what is still uncertain (one sentence, "
            f"not a separate section).\n"
            f"- No preamble. Start directly with the answer.\n\n"
            f"BEFORE finalizing: re-read your answer. For every number, page reference, "
            f"or specific fact you wrote, confirm it literally appears in FINDINGS above. "
            f"If it doesn't, remove it and say that detail wasn't found instead.\n\n"
            f"Wrap the final answer in <final_answer>...</final_answer>."
        )
        try:
            raw = llm.chat(system=self.system_prompt, user=prompt, max_tokens=1536)
        except RuntimeError as e:
            err_str = str(e).lower()
            if "tool_use_failed" in err_str or "tool choice is none" in err_str or "called a tool" in err_str:
                print("      ⚠️ Model attempted a native tool call during synthesis — "
                      "retrying with a firmer no-tool-calls instruction.")
                firmer_prompt = prompt + (
                    "\n\nIMPORTANT: Do NOT call any tool, function, or browser action of "
                    "any kind — none are available to you right now. You already have "
                    "everything you need in the FINDINGS above; use only that. Write your "
                    "answer as plain prose text, wrapped in <final_answer>...</final_answer>, "
                    "and nothing else."
                )
                raw = llm.chat(system=self.system_prompt, user=firmer_prompt, max_tokens=1536, purpose="synthesize")
            else:
                raise


        parsed = ResponseParser.parse(raw)
        final_text = parsed.final_answer or raw

        def _looks_like_tool_call(text: str) -> bool:
            t = text.strip()
            return t.startswith("{") and ('"action"' in t or '"tool"' in t or '"parameters"' in t)


        if _looks_like_tool_call(final_text):
            print("      ⚠️ Synthesis returned a raw tool-call-shaped JSON instead of prose — retrying with a firmer instruction.")
            retry_prompt = prompt + (
                "\n\nIMPORTANT: Do NOT call any tool or emit JSON of any kind. You "
                "already have everything you need in FINDINGS above — use it. Write "
                "your answer as plain prose text only, wrapped in "
                "<final_answer>...</final_answer>, and nothing else."
            )
            try:
                raw_retry = llm.chat(system=self.system_prompt, user=retry_prompt, max_tokens=1536)
                parsed_retry = ResponseParser.parse(raw_retry)
                retry_text = (parsed_retry.final_answer or raw_retry).strip()
                final_text = retry_text if not _looks_like_tool_call(retry_text) else (
                    "The research gathered relevant information, but the model's "
                    "response could not be converted into a plain-text answer. "
                    "Please try rephrasing the question or running it again."
                )
            except Exception:
                final_text = (
                    "The research gathered relevant information, but the model's "
                    "response could not be converted into a plain-text answer. "
                    "Please try rephrasing the question or running it again."
                )

        final_text = re.sub(
            r'Self-reflection score:\s*\d+%',
            f'Self-reflection score: {reflect_score:.0%}',
            final_text,
            flags=re.IGNORECASE,
        )
        return final_text

    @staticmethod
    def _parse_decision(raw: str, scratchpad: None) -> AgentDecision:
        parsed = ResponseParser.parse(raw)
        thought = parsed.thought or "Evaluating retrieval quality..."
        
        action_data = {}
        #action_str = parsed.action.strip() if parsed.action else ""

        if parsed.action and parsed.action.startswith("{"):
            # Trim anything after the last balanced '}' — repairs stray trailing
            # characters some models append (e.g. an extra closing quote) without
            # touching well-formed JSON.
            action_str = parsed.action.strip()
            end = action_str.rfind("}")
            
            if end != -1:
                action_str = action_str[:end + 1]
        
        #if parsed.action and parsed.action.strip().startswith("{"):
            try:
                action_data = json.loads(action_str)
            except json.JSONDecodeError:
                print(f"      ⚠️ Failed to parse action JSON. Raw action text: {parsed.action!r}")
                
              

        if not action_data:
            return BasePattern._fallback_decision_on_parse_failure(thought, scratchpad)
                    
            # Parsing genuinely failed — don't guess a tool. Retry document_reader
            # with the raw query rather than defaulting to web_search, since a
            # parse failure carries no information about what the model intended.
            print(f"      ⚠️ No action parsed from model output — defaulting to document_reader retry.")
            return AgentDecision(
                thought=thought + " (action parsing failed — retrying with document_reader)",
                tool_name="document_reader",
                tool_input="",
                is_final=False,
            )

        tool = action_data.get("tool") or "web_search"
        inp = action_data.get("input", "")
        
        is_final = tool.upper() == "FINISH"

        return AgentDecision(
            thought=thought,
            tool_name=tool if not is_final else "synthesizer",
            tool_input=inp,
            is_final=is_final,
        )
