import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional
from pathlib import Path

if TYPE_CHECKING:
    from loop.scratchpad import Scratchpad
    from tools.base import ToolResult
    from tools.registry import ToolRegistry
    from utils.llm_client import LLMClient


WEB_FALLBACK_CONFIDENCE_THRESHOLD = 0.50
MAX_WEB_FALLBACK_RETRIES = 2

_COMPUTATION_TRIGGER_WORDS = [
    "calculate", "compute", "variance", "market share", "per capita",
    "derive", "derivation", "percentage", "difference", "ratio",
]

@dataclass
class AgentDecision:
    thought: str
    tool_name: str
    tool_input: str
    is_final: bool = False


class BasePattern(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier: 'react', 'selfrag', 'crag'"""

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """The system prompt that shapes this pattern's behaviour."""
        

    @abstractmethod
    def think(
        self,
        query: str,
        scratchpad: "Scratchpad",
        llm: "LLMClient",
        registry: "ToolRegistry",
        plan: Optional[dict] = None,
    ) -> AgentDecision:
        """Produce the next AgentDecision."""

    @abstractmethod
    def post_process(
        self,
        tool_result: "ToolResult",
        query: str,
        scratchpad: "Scratchpad",
        llm: "LLMClient",
        registry: "ToolRegistry",
        tool_name="None",
    ) -> "ToolResult":
        """
        Optionally grade / correct the tool result.
        Return (possibly modified) ToolResult.
        """
        return tool_result

    @abstractmethod
    def synthesize(
        self,
        query: str,
        scratchpad: "Scratchpad",
        llm: "LLMClient",
    ) -> str:
        """Produce the final answer string."""

    @staticmethod
    def _has_usable_prior_step(scratchpad, min_score: float = 0.5) -> bool:
        """A prior step only counts as 'usable' if it's substantial AND
        was actually graded well — length alone isn't evidence of quality,
        just evidence the tool didn't return an empty string."""

        _NO_INFO_MARKERS = (
            "does not contain", "not provided in", "cannot be found",
            "no information", "not found in", "no relevant information",
        )
        if scratchpad is None:
            return False
        for s in scratchpad.steps:
            obs = (s.observation or "").strip()
            if len(obs) <= 100:
                continue
            if any(m in obs.lower() for m in _NO_INFO_MARKERS):
                continue
            score = getattr(s, "confidence", None)
            if score is None:
                score = getattr(s, "grade", None)
            if score is not None and score < min_score:
                continue  # long, on-topic-looking text, but graded low — not usable
            return True
        return False


    @staticmethod
    def _fallback_decision_on_parse_failure(thought: str, scratchpad) -> AgentDecision:
        """Shared by every pattern's _parse_decision when the model's raw
        output couldn't be parsed into a tool call."""
        if BasePattern._has_usable_prior_step(scratchpad):
            print("      ⚠️ No action parsed, but a prior step was both "
                  "substantial and well-graded — finishing instead of repeating a tool call.")
            return AgentDecision(
                thought=thought + " (action parsing failed — sufficient graded data already gathered)",
                tool_name="synthesizer",
                tool_input="",
                is_final=True,
            )

        needs_web = any(getattr(s, "needs_correction", False) for s in (scratchpad.steps if scratchpad else []))
        next_tool = "web_search" if needs_web else "document_reader"
        print(f"      ⚠️ No action parsed from model output — retrying with '{next_tool}'.")
        return AgentDecision(
            thought=thought + f" (action parsing failed — retrying with {next_tool})",
            tool_name=next_tool,
            tool_input="",
            is_final=False,
        )

    @staticmethod
    def _check_verified_computed_result(tool_result, status_key: str) -> bool:
        """A VERIFIED COMPUTED RESULT (deterministic pandas output from
        csv_analyzer) is ground truth — never needs LLM grading or web
        correction. Shared identically by CRAG and Self-RAG."""
        if "VERIFIED COMPUTED RESULT" in (tool_result.content or ""):
            tool_result.metadata[status_key] = "verified_computed"
            tool_result.metadata["grade"] = 1.0
            tool_result.confidence = max(tool_result.confidence, 0.95)
            return True
        return False

    @staticmethod
    def _apply_prior_correction_wrap(tool_result, scratchpad, status_key: str, label: str) -> bool:
        """If the PREVIOUS step was flagged needs_correction, THIS step's
        result is the correction attempt — wrap it with the prior low-
        relevance content for context, clear the flag. Shared by both
        patterns; only the wrapper label ('Web correction' vs 'Correction
        attempt') differs."""
        if scratchpad.steps and getattr(scratchpad.steps[-1], 'needs_correction', False):
            prior = scratchpad.steps[-1]
            prior_snippet = (prior.observation or "")[:500]
            tool_result.content = (
                f"[{label} — NEW information from this hop]\n"
                f"{tool_result.content}\n\n"
                f"[Original low-relevance retrieval — grade {prior.grade:.0%}, for reference]\n"
                f"{prior_snippet}"
            )
            tool_result.metadata[status_key] = "correction_applied"
            tool_result.metadata["corrected"] = True
            tool_result.metadata["needs_correction"] = False
            tool_result.confidence = max(tool_result.confidence, 0.6)
            return True
        return False
    

    @staticmethod
    def _is_trivially_empty(content: str) -> bool:
        """Deterministic, LLM-free check — content this short or containing
        this exact phrase is almost certainly a non-answer, no need to spend
        an LLM grading call finding that out. Previously Self-RAG-only; now
        shared so CRAG gets the same fast, reliable short-circuit instead of
        always paying for an LLM score on empty content."""
        c = (content or "").strip()
        return len(c) < 100 or "no highly relevant" in c.lower()
    
    
    @staticmethod
    def _maybe_flag_web_fallback(
        tool_result, scratchpad, score: float, status_key: str,
        max_retries: int = MAX_WEB_FALLBACK_RETRIES,
        threshold: float = WEB_FALLBACK_CONFIDENCE_THRESHOLD,
    ) -> bool:
        """Single shared confidence gate: whatever pattern is running, however
        it computed `score`, this decides — and mechanically triggers — a
        web_search fallback the SAME way every time. This is what makes
        web_search a normal, always-available tool across react/crag/selfrag
        rather than something wired ad hoc per pattern."""
        already_web = tool_result.metadata.get("tool_used") == "web_search"
        corrections_made = sum(1 for s in scratchpad.steps if getattr(s, "corrected", False))
        if score < threshold and not already_web and corrections_made < max_retries:
            tool_result.metadata["needs_correction"] = True
            tool_result.metadata["corrected"] = True
            tool_result.metadata[status_key] = "needs_web_fallback"
            tool_result.confidence = score
            return True
        return False

    @staticmethod
    def _maybe_flag_calculation_needed(tool_result, scratchpad, query, status_key: str) -> bool:
        """Shared reactive gate, mirroring _maybe_flag_web_fallback: if the
        question needs a numeric computation and this hop's content contains
        raw figures relevant to it, but no computation has happened yet, flag
        needs_calculation so the engine forces a calculator hop next. This
        lets ANY pattern (react/crag/selfrag) trigger a real calculation as
        soon as the numbers it needs actually appear, instead of relying only
        on the static, upfront plan.needs_computation classifier."""
        q_lower = query.lower()
        if not any(w in q_lower for w in _COMPUTATION_TRIGGER_WORDS):
            return False

        already_computed = any(
            "VERIFIED COMPUTED RESULT" in (s.observation or "") or s.tool_used == "calculator"
            for s in scratchpad.steps
        )
        if already_computed:
            return False

        has_raw_numbers = bool(re.search(r'\$?\d[\d,]*\.?\d*\s*(million|billion|%)?', tool_result.content or ""))
        if has_raw_numbers:
            tool_result.metadata["needs_calculation"] = True
            tool_result.metadata[status_key] = f"{tool_result.metadata.get(status_key, '')}+needs_calculation"
            return True
        return False


    def _local_documents_note(self, scratchpad: "Scratchpad", plan: Optional[dict] = None) -> str:
        try:
            from utils.paths import DOCS_DIR  # centralized path, see earlier fix
        except ImportError:
            DOCS_DIR = Path("./documents")

        files = [
            f.name for f in Path(DOCS_DIR).rglob("*")
            if f.is_file() and f.suffix.lower() in {".pdf", ".docx", ".txt", ".md", ".csv"}
        ]
        if not files:
            return ""

        plan_says_needed = plan is None or plan.get("needs_document", True)
        if not plan_says_needed:
            return (
                f"\nLOCALLY AVAILABLE DOCUMENTS (already uploaded, in ./documents):\n"
                f"{file_list}\n\n"
                f"These do not appear relevant to this question based on prior "
                f"analysis — only use document_reader if you have reason to "
                f"believe otherwise.\n"
            )

        file_list = "\n".join(f"  - {name}" for name in files)

        is_first_hop = len(scratchpad.steps) == 0
        mandate = (
            "This is your FIRST action. Local documents exist above — you MUST "
            "select 'document_reader' as your tool for this action, not "
            "web_search, even if you're unsure it will contain the answer. "
            "You may switch to web_search on a later hop if document_reader's "
            "result turns out to be insufficient.\n"
            if is_first_hop else
            "If a previous document_reader attempt failed or was insufficient, "
            "web_search is a reasonable next step.\n"
        )
        return (
            f"{mandate}\n\n"
            f"If the question could plausibly be answered using one of these "
            f"local files — even partially, or if it references 'the document', "
            f"'the file', 'the resume', 'the paper', 'chapter X', or names a "
            f"topic one of these files likely covers — you MUST try "
            f"'document_reader' BEFORE 'web_search'. Only use web_search if "
            f"none of these files are plausibly relevant, or document_reader's "
            f"result was insufficient.\n"
        )



    def _build_think_prompt(
        self,
        query: str,
        scratchpad: "Scratchpad",
        registry: "ToolRegistry",
        extra_instructions: str = "",
        plan: Optional[dict] = None,
    ) -> str:
        valid_tools = ", ".join(f'"{t}"' for t in registry.list_names()) if hasattr(registry, "list_names") else ""

        
        return f"""You are an Agentic RAG Researcher using the {self.name.upper()} pattern.

ORIGINAL QUESTION: {query}

AVAILABLE TOOLS (choose ONLY from this exact list, no other tool names exist):
{registry.descriptions()}
{self._local_documents_note(scratchpad, plan)}

VALID tool values for the "tool" field: {valid_tools}, or "FINISH" if you have enough information.
Do NOT invent tool names like "GeneralKnowledgeTool" or "kb_tool" — they do not exist and will fail.

TOOL INPUT RULES — the "input" field must be a purpose-built value for the
chosen tool, NOT a restatement of your own reasoning or the full original
question:
- For "web_search": a short, direct search-engine query (3-8 words) covering
  ONLY what is still missing. Do not include words like "uploaded documents"
  or "check if" — those describe your reasoning process, not a search query.
  GOOD: "2026 LLM benchmark comparison SOTA models"
  BAD:  "uploaded documents, model architectural dimensions and parameters"
    (this restates your thought process instead of searching for anything)
- For "document_reader": a short topic/keyword phrase describing what to
  look for in the document (e.g. "d_model dimensions", "Table 3
  hyperparameters") — not the full original question verbatim.
- If a previous hop already searched or read something and it FAILED or came
  up empty (see SCRATCHPAD below), do NOT repeat the same input again —
  reformulate it with different, more specific terms, or switch tools.

SCRATCHPAD (what you have found so far):
{scratchpad.context_for_prompt()}

{extra_instructions}

Decide your NEXT action. Respond in this EXACT format:
<thought>Your step-by-step reasoning about what to do next</thought>
<action>{{"tool": "tool_name", "input": "what to search for"}}</action>

The "tool" value MUST be exactly one of: {valid_tools}, FINISH
"""
