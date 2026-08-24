import json
from typing import TYPE_CHECKING
from .base import BasePattern, AgentDecision
from utils.parser import ResponseParser
from utils.known_facts import get_known_fact_note
from pathlib import Path

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

    def think(self, query, scratchpad, llm, registry, plan=None) -> AgentDecision:
        extra = (
            "REACT RULES:\n"
            "1. Write a detailed <thought> explaining your reasoning.\n"
            "2. Pick the BEST tool for this hop.\n"
            "3. Each hop should answer ONE sub-question.\n"
            "4. Only call FINISH when confidence is high.\n"
            "5. Before calling FINISH, check: does the LAST observation actually "
            "contain the SPECIFIC facts/numbers the question asks for — not just "
            "a related topic? If it's only an abstract/intro and the question asks "
            "for specific tables, values, or details, try again with a more "
            "specific tool_input (e.g. 'Table 3 hyperparameters') or a different hop "
            "before finishing.\n"
            f"NEVER propose a hypothetical mapping between unrelated data and what the "
            f"question asked (e.g. 'if we assume X is related to Y') — if the specific "
            f"data requested doesn't exist, say so plainly and stop there.\n"
        )
        prompt = self._build_think_prompt(query, scratchpad, registry, extra)
        raw = llm.chat(system=self.system_prompt, user=prompt, max_tokens=1536, purpose="think")
        return self._parse_decision(raw, scratchpad)

    def post_process(self, tool_result, query, scratchpad, llm, registry, tool_name=None):
        return tool_result

    def synthesize(self, query, scratchpad, llm) -> str:
        context = scratchpad.all_observations()
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
            )

        prompt = (
           f"GROUNDING RULE — CRITICAL, READ FIRST:\n"
           f"Every specific number, name, or fact you state MUST come from the FINDINGS "
           f"below. Do NOT fill gaps using your own general knowledge or training data, "
           f"even if confident. If FINDINGS don't contain a value the question asks for, "
           f"say so explicitly rather than guessing or estimating — do not use words like "
           f"'likely' or 'typically' to soften a guess; either state it as fact from the "
           f"FINDINGS, or say it wasn't found.\n\n"
           f"{calc_rule}\n"
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
           f"BEFORE finalizing: re-read your answer. For every number or specific fact you "
           f"wrote, confirm it literally appears in FINDINGS above. If it doesn't, remove "
           f"it and say that detail wasn't found in the reviewed excerpt instead.\n\n"
           f"Wrap the final answer in <final_answer>...</final_answer>."
        )
        raw = llm.chat(system=self.system_prompt, user=prompt, max_tokens=1536)
        parsed = ResponseParser.parse(raw)
        return parsed.final_answer or raw

    @staticmethod
    def _parse_decision(raw: str, scratchpad=None) -> AgentDecision:
        parsed = ResponseParser.parse(raw)
        thought = parsed.thought or "Thinking about next step..."
        
        action_data = {}
        #action_str = parsed.action.strip() if parsed.action else ""
        if parsed.action and parsed.action.strip().startswith("{"):
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
                print(f"      ⚠️ Malformed action JSON: {parsed.action!r}")
                #action_data = {}

        if not action_data:
            # Don't blindly repeat document_reader if it already produced a
            # real, usable result — that just re-reads the same file for no
            # new information. Check whether ANY prior step already returned
            # substantial, non-empty, non-"not found" content before deciding
            # to retry vs. finish with what's already gathered.
            has_usable_prior = False
            if scratchpad is not None:
                no_info_markers = (
                    "does not contain", "not provided in", "cannot be found",
                    "no information", "not found in", "no relevant information",
                )
                for s in scratchpad.steps:
                    obs = (s.observation or "").strip()
                    if len(obs) > 100 and not any(m in obs.lower() for m in no_info_markers):
                        has_usable_prior = True
                        break

            if has_usable_prior:
                 print("      ⚠️ No action parsed, but usable data was already "
                       "gathered — finishing instead of repeating a tool call.")
                 return AgentDecision(
                     thought=thought + " (action parsing failed — sufficient data already gathered)",
                     tool_name="synthesizer",
                     tool_input="",
                     is_final=True,
                 )

                    
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

        #print(f"      ⚠️ No action parsed from model output — defaulting to document_reader retry.")
        return AgentDecision(
            thought=thought,
            tool_name=tool if not is_final else "synthesizer",
            tool_input=inp,
            is_final=is_final,
        )
