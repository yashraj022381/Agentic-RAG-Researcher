from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional
from pathlib import Path

if TYPE_CHECKING:
    from loop.scratchpad import Scratchpad
    from tools.base import ToolResult
    from tools.registry import ToolRegistry
    from utils.llm_client import LLMClient

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
