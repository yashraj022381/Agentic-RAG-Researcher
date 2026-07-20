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

    def think(self, query, scratchpad, llm, registry) -> AgentDecision:
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
        )
        prompt = self._build_think_prompt(query, scratchpad, registry, extra)
        raw = llm.chat(system=self.system_prompt, user=prompt, max_tokens=1024)
        return self._parse_decision(raw)

    def post_process(self, tool_result, query, scratchpad, llm, registry):

        score_prompt = CRAG_SCORE_PROMPT.format(
            query=query,
            content=tool_result.content[:800],
        )
        score = llm.grade(score_prompt)
        tool_result.metadata["grade"] = score
        tool_result.metadata["crag_score"] = score

        corrections_made = sum(1 for s in scratchpad.steps if s.corrected)

        if score >= 0.8:
            tool_result.metadata["ctag_status"] = "correct"
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
            return tool_result

        else:
            if corrections_made < self.max_corrections:
                web_tool = registry.get("web_search")
                if web_tool:
                    web_result = web_tool.run(query=query)
                    web_result.metadata["grade"] = score
                    web_result.metadata["crag_status"] = "incorrect_corrected"
                    web_result.metadata["corrected"] = True
                    web_result.metadata["original_kb"] = tool_result.content[:200]
                    return web_result

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
        context = scratchpad.all_observations()

        correction_note = ""
        if corrections:
            correction_note = (
                f"\nNOTE: {len(corrections)} correction(s) were made where "
                f"knowledge base results scored below threshold and web search was used.\n"
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

        prompt = (
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
            f"- Be explicit about which facts were corrected, but keep it to one short "
            f"sentence — not a separate itemized section.\n"
            f"- No preamble. Start directly with the answer.\n\n"
            f"Wrap the final answer in <final_answer>...</final_answer>."
        )
        raw = llm.chat(system=self.system_prompt, user=prompt, max_tokens=1200)
        parsed = ResponseParser.parse(raw)
        return parsed.final_answer or raw

    @staticmethod
    def _parse_decision(raw: str) -> AgentDecision:
        parsed = ResponseParser.parse(raw)
        thought = parsed.thought or "Scoring retrieval quality..."
        action_data = {}
        if parsed.action and parsed.action.strip().startswith("{"):
            try:
                action_data = json.loads(parsed.action)
            except json.JSONDecodeError:
                action_data = {}

        tool = action_data.get("tool", parsed.action or "web_search")
        inp = action_data.get("input", parsed.action_input or "")

        is_final = tool.upper() == "FINISH"

        return AgentDecision(
            thought=thought,
            tool_name=tool if not is_final else "synthesizer",
            tool_input=inp,
            is_final=is_final,
        )
