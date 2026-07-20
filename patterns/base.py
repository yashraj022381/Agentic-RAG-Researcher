from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

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
    ) -> "ToolResult":
        """
        Optionally grade / correct the tool result.
        Return (possibly modified) ToolResult.
        """

    @abstractmethod
    def synthesize(
        self,
        query: str,
        scratchpad: "Scratchpad",
        llm: "LLMClient",
    ) -> str:
        """Produce the final answer string."""


    def _build_think_prompt(
        self,
        query: str,
        scratchpad: "Scratchpad",
        registry: "ToolRegistry",
        extra_instructions: str = "",
    ) -> str:
        valid_tools = ", ".join(f'"{t}"' for t in registry.list_names()) if hasattr(registry, "list_names") else ""
        
        return f"""You are an Agentic RAG Researcher using the {self.name.upper()} pattern.

ORIGINAL QUESTION: {query}

AVAILABLE TOOLS (choose ONLY from this exact list, no other tool names exist):
{registry.descriptions()}

VALID tool values for the "tool" field: {valid_tools}, or "FINISH" if you have enough information.
Do NOT invent tool names like "GeneralKnowledgeTool" or "kb_tool" — they do not exist and will fail.

SCRATCHPAD (what you have found so far):
{scratchpad.context_for_prompt()}

{extra_instructions}

Decide your NEXT action. Respond in this EXACT format:
<thought>Your step-by-step reasoning about what to do next</thought>
<action>{{"tool": "tool_name", "input": "what to search for"}}</action>

The "tool" value MUST be exactly one of: {valid_tools}, FINISH
"""
