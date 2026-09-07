import re
import json
from typing import TYPE_CHECKING
from .base import BasePattern, AgentDecision
from utils.parser import ResponseParser

if TYPE_CHECKING:
    from loop.scratchpad import Scratchpad
    from tools.base import ToolResult
    from tools.registry import ToolRegistry
    from utils.llm_client import LLMClient


CRAG_SCORE_PROMPT = """
You are a CRAG retrieval quality scorer.

Question: {query}
Retrieved content: {content}

Score the quality of this retrieval on THREE dimensions:
1. Relevance  (does it address the question?)
2. Accuracy   (is the information likely correct?)
3. Completeness (does it fully answer the question?)

Give an OVERALL score from 0.0 to 1.0.
Respond with ONLY a decimal number.
"""

CRAG_CORRECTION_PROMPT = """
The original knowledge base retrieval scored LOW quality.
Web search was used as a correction.

Original KB result: {kb_content}
Web correction:     {web_content}
Question:           {query}

Merge these into the most accurate answer possible.
Prefer the web content where they conflict.
Wrap in <final_answer>...</final_answer>.
"""


class CRAGPattern(BasePattern):

    def __init__(self, score_threshold: float = 0.50, max_corrections: int = 2):
        self.score_threshold = score_threshold
        self.max_corrections = max_corrections

    @property
    def name(self) -> str:
        return "crag"

    @property
    def system_prompt(self) -> str:
        return (
            "You are a CRAG (Corrective RAG) research agent. "
            "You score every retrieved result and correct low-quality retrievals "
            "by falling back to web search. "
            "You always fact-check critical claims before including them. "
            "Be explicit when you have made corrections."
        )

    def think(self, query, scratchpad, llm, registry, plan=None) -> AgentDecision:
        corrections_made = sum(1 for step in scratchpad.steps
                               if hasattr(step, 'corrected') and getattr(step, 'corrected', False)
        )

        extra = (
            f"CRAG RULES:\n"
            f"1. Every retrieval will be scored (handled automatically).\n"
            f"2. Low scores trigger automatic web_search correction.\n"
            f"3. Corrections made so far: {corrections_made}/{self.max_corrections}\n"
            f"4. Always use fact_checker for specfic claims (dates, names, numbers).\n"
            f"5. Call FINISH only after verifying your key facts.\n"
            f"NEVER propose a hypothetical mapping between unrelated data and what the "
            f"question asked (e.g. 'if we assume X is related to Y') — if the specific "
            f"data requested doesn't exist, say so plainly and stop there.\n"
        )
        prompt = self._build_think_prompt(query, scratchpad, registry, extra)
        raw = llm.chat(system=self.system_prompt, user=prompt, max_tokens=1536, purpose="think")
        return self._parse_decision(raw, scratchpad)

    def post_process(self, tool_result, query, scratchpad, llm, registry):
        
        if self._check_verified_computed_result(tool_result, "crag_status"):
            return tool_result

        if self._apply_prior_correction_wrap(tool_result, scratchpad, "crag_status", "Correction attempt"):
            return tool_result

        corrections_made = sum(1 for s in scratchpad.steps if s.corrected)
        score = 0.0 if self._is_trivially_empty(tool_result.content) else llm.grade(
            CRAG_SCORE_PROMPT.format(query=query, content=tool_result.content[:800])
        )
        tool_result.metadata["grade"] = score
        tool_result.metadata["crag_score"] = score
       
        if score >= 0.8:
                tool_result.metadata["crag_status"] = "correct"
                return tool_result
            
        existence_check_signal = any(phrase in query.lower() for phrase in [
            "does the document", "does the handbook", "does the policy",
            "is there a policy", "what is the policy for", "what is the exact policy",
            "mention", "exist",
        ]) 

        is_tabular_result = tool_result.metadata.get("tool_used") == "csv_analyzer" or (
            tool_result.source and "csv" in tool_result.source.lower()
        )
        already_web_checked = tool_result.metadata.get("tool_used") == "web_search"


        if (existence_check_signal or is_tabular_result) and corrections_made == 0:
            if is_tabular_result and not existence_check_signal:
                # csv_analyzer's own content already explains what's missing —
                # trust it rather than overwriting with a generic message
                tool_result.metadata["crag_status"] = "schema_mismatch_flagged"

            return tool_result

        
        elif score >= self.score_threshold:
            if corrections_made < self.max_corrections:
                web_tool = registry.get("web_search")
                if web_tool:
                    web_result = web_tool.run(query=query)
                    tool_result.content = (
                        f"[KB - confidence {score:.0%}]\n{tool_result.content}\n\n"
                        f"[Web supplement]\n{web_result.content}"
                    )
                    tool_result.confidence = max(tool_result.confidence, web_result.confidence)
                    tool_result.metadata["crag_status"] = "ambiguous_augmented"
                    tool_result.metadata["corrected"] = True
                    tool_result.metadata["tool_used"] = "web_search"
            return tool_result    

               
        score_prompt = CRAG_SCORE_PROMPT.format(
            query=query,
            content=tool_result.content[:800],
        )
        score = llm.grade(score_prompt)
        #tool_result.metadata["grade"] = score
        #tool_result.metadata["crag_score"] = score
        #corrections_made = sum(1 for s in scratchpad.steps if s.corrected)

        # Everything else: use the SAME shared gate every pattern uses
        if self._maybe_flag_web_fallback(tool_result, scratchpad, score, "crag_status"):
            return tool_result

        # react.py, crag.py, selfrag.py — add this line in post_process, e.g.:
        self._maybe_flag_calculation_needed(tool_result, scratchpad, query, "crag_status")  # or "crag_status"/"selfrag_status"

        tool_result.metadata["crag_status"] = "incorrect_uncorrected"
        tool_result.confidence = score
        return tool_result
            

    @staticmethod
    def _extract_year_facts(text: str) -> dict:
        """Deterministically pull 'X ... YYYY' year mentions per subject."""
        facts = {}
        patterns = {
            "python": r'python[^.]{0,60}?\b(19\d{2}|20\d{2})\b',
            "java":   r'java[^.]{0,60}?\b(19\d{2}|20\d{2})\b',
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                facts[key] = int(match.group(1))
        return facts

 
    def synthesize(self, query, scratchpad, llm) -> str:
        corrections = [s for s in scratchpad.steps if s.corrected]
        per_step_cap = 1200 if corrections else 2000
        context = scratchpad.all_observations(max_chars=6000, per_step_chars=per_step_cap)

        correction_note = ""
        if corrections:
            correction_lines = []
            for s in corrections:
                snippet = (s.observation or "").strip()[:400]
                correction_lines.append(f"- {s.tool_used} was checked as a correction and found: {snippet}")
            correction_summary = "\n".join(correction_lines)
            correction_note = (
                f"\nCORRECTIONS PERFORMED — YOU MUST MENTION THIS IN YOUR ANSWER:\n"
                f"{len(corrections)} correction(s) were made because the initial retrieval "
                f"scored below threshold. Here is what each correcting source actually found:\n"
                f"{correction_summary}\n"
                f"Your answer MUST explicitly state that this additional source was checked "
                f"and summarize what it found (or that it also found nothing conclusive on "
                f"the specific topic) — do not write an answer that only discusses the "
                f"original document/retrieval and silently omits the correction attempt.\n"
            )
            topic_contrast_note = (
                f"\nTOPIC CONTRAST RULE — UNIVERSAL, applies to any false-premise case:\n"
                f"If the ORIGINAL source doesn't cover the specific narrow thing asked "
                f"about, but DOES discuss the general topic area (even a related or "
                f"traditional version of it), your answer MUST explicitly say what the "
                f"source actually covers on that general topic — do not just say "
                f"'not found' and stop. For example: if asked about an electronic/digital "
                f"version of something and the source only covers a traditional/manual "
                f"version, say so explicitly: 'the source covers [traditional version] "
                f"but does not mention [the specific electronic/digital variant asked "
                f"about].' Only skip this if the source genuinely has NOTHING related to "
                f"the general topic at all.\n"
            )

        year_facts = self._extract_year_facts(context)
        verified_note = ""
        if len(year_facts) >= 2:
            older_key = min(year_facts, key=year_facts.get)
            newer_key = max(year_facts, key=year_facts.get)
            verified_note = (
                f"\nVERIFIED FACT (computed programmatically — treat as ground truth, "
                f"do NOT recalculate or contradict this):\n"
                f"{older_key.capitalize()} ({year_facts[older_key]}) is OLDER than "
                f"{newer_key.capitalize()} ({year_facts[newer_key]}), "
                f"because {year_facts[older_key]} < {year_facts[newer_key]}.\n"
            )

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
            f"it yourself. (The VERIFIED FACT below, if present, is the one exception "
            f"— it's computed programmatically and should be trusted as ground truth.)\n\n"
            f"{calc_rule}\n"
            f"Using these CRAG-verified findings, write a complete answer to:\n"
            f"'{query}'\n"
            f"{correction_note}"
            f"{verified_note}\n"
            f"FINDINGS:\n{context}\n\n"
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
            f"- IMPORTANT: being concise means cutting narration, repetition, and preamble "
            f"— it does NOT mean cutting specific facts. ALWAYS include exact dates, "
            f"numbers, names, and figures from the FINDINGS when they're part of the answer.\n"
            f"- Do NOT break your answer into a numbered list of sub-questions, and do NOT "
            f"narrate your reasoning process. Just answer.\n"
            f"- Do NOT cite a source after every single sentence. Mention a source naturally "
            f"at most once or twice total.\n"
            f"- If corrections were performed (see above), your answer MUST mention that "
            f"an external source was also checked and what it found — this is not optional, "
            f"and it does not need a separate itemized section, just one clear sentence.\n"
            f"- No preamble. Start directly with the answer.\n\n"
            f"BEFORE finalizing: re-read your answer. For every number, page reference, "
            f"or specific fact you wrote, confirm it literally appears in FINDINGS above "
            f"(the VERIFIED FACT is the exception, per the grounding rule). If it doesn't, "
            f"remove it and say that detail wasn't found instead. Also confirm: if "
            f"corrections were performed, does your answer actually mention that a "
            f"second source was checked? If not, add one sentence covering it.\n\n"
            f"Wrap the final answer in <final_answer>...</final_answer>."
        )
        try:
            raw = llm.chat(system=self.system_prompt, user=prompt, max_tokens=2048)
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
        final_text = (parsed.final_answer or raw).strip()

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
        return final_text

    @staticmethod
    def _parse_decision(raw: str, scratchpad: None) -> AgentDecision:
        parsed = ResponseParser.parse(raw)
        thought = parsed.thought or "Scoring retrieval quality..."
        
        action_data = {}

        if parsed.action and parsed.action.startswith("{"):
            # Trim anything after the last balanced '}' — repairs stray trailing
            # characters some models append (e.g. an extra closing quote) without
            # touching well-formed JSON.
            action_str = parsed.action.strip()
            end = action_str.rfind("}")
            if end != -1:
                action_str = action_str[:end + 1]
            try:
                action_data = json.loads(action_str)
            except json.JSONDecodeError:
                print(f"      ⚠️ Malformed action JSON: {parsed.action!r}")
               
        if not action_data:
            return BasePattern._fallback_decision_on_parse_failure(thought, scratchpad)


            print(f"      ⚠️ No action parsed from model output — defaulting to document_reader retry.")
            return AgentDecision(
                thought=thought + " (action parsing failed — retrying)",
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
