import json
from typing import TYPE_CHECKING
from .base import BasePattern, AgentDecision
from utils.parser import ResponseParser
from utils.known_facts import get_known_fact_note

if TYPE_CHECKING:
    from loop.scratchpad import Scratchpad
    from tools.base import ToolResult
    from tools.registry import ToolRegistry
    from utils.llm_client import LLMClient


class ReActPattern(BasePattern):

    @property
    def name(self) -> str:
        return "react"

    @property
    def system_prompt(self) -> str:
        return (
            "You are a ReAct-Style research agent. "
            "You ALWAYS write a <thought> before every <action>. "
            "You break complex questions into sub-questions and answer each hop. "
            "Never skip the thought step - it is mandatory."
        )

    def think(self, query, scratchpad, llm, registry) -> AgentDecision:
        extra = (
            "REACT RULES:\n"
            "1. Write a detailed <thought> explaining your reasoning.\n"
            "2. Pick the BEST tool for this hop.\n"
            "3. Each hop should answer ONE sub-question.\n"
            "4. Only call FINISH when confidence is high.\n"
        )
        prompt = self._build_think_prompt(query, scratchpad, registry, extra)
        raw = llm.chat(system=self.system_prompt, user=prompt, max_tokens=1024)
        return self._parse_decision(raw)

    def post_process(self, tool_result, query, scratchpad, llm, registry):
        return tool_result

    def synthesize(self, query, scratchpad, llm) -> str:
        context = scratchpad.all_observations()
        known_fact_note = get_known_fact_note(query)
        prompt = (
            f"Using these research findings, write a clear, complete answer to:\n"
            f"'{query}'\n\n"
            f"FINDINGS:\n{context}\n"
            f"{known_fact_note}\n"
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
            f"total unless the question genuinely requires more detail (e.g. it asks for "
            f"a list, a formula, or step-by-step instructions).\n"
            f"- IMPORTANT: being concise means cutting narration, repetition, and preamble "
            f"— it does NOT mean cutting specific facts. ALWAYS include exact dates, "
            f"numbers, names, and figures from the FINDINGS when they're part of the "
            f"answer, even in a short response.\n"
            f"- Do NOT break your answer into a numbered list of sub-questions, and do NOT "
            f"narrate your reasoning process ('First I will...', 'Let's break this down...', "
            f"'Now that I know X, I need to find Y...'). Just answer.\n"
            f"- Do NOT cite a source after every single sentence. Mention a source naturally "
            f"at most once or twice total, e.g. '(per the USGS)' — not after every fact.\n"
            f"- No preamble. Start directly with the answer.\n\n"
            f"Wrap the final answer in <final_answer>...</final_answer>."
        )
        raw = llm.chat(system=self.system_prompt, user=prompt, max_tokens=1200)
        parsed = ResponseParser.parse(raw)
        return parsed.final_answer or raw

    @staticmethod
    def _parse_decision(raw: str) -> AgentDecision:
        parsed = ResponseParser.parse(raw)
        thought = parsed.thought or "Thinking about next step..."
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
