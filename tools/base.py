from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict

@dataclass
class ToolResult:
    content: str
    source: str
    confidence: float = 0.7
    metadata: dict = field(default_factory = dict)

    def is_useful(self, threshold: float = 0.4) -> bool:
        return self.confidence >= threshold and bool(self.content.strip())

class BaseTool(ABC):

    @property
    def name(self) -> str:
        """Short identifier, e.g. 'web_search'"""

    @property
    def description(self) -> str:
        """One-line description for the agent to read."""

    @abstractmethod
    def run(self, query: str, context: Optional[str] = None) -> ToolResult:
        """
        Execute the tool.

        Args:
            query: What to look for
            context: Optional prior context to help the tool

        Returns:
            ToolResult
        """

    def __repr__(self):
        return f"<Tool: {self.name}>"
