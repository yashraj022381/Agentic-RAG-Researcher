from typing import Dict, Optional, List
from .base import BaseTool
from .web_search import WebSearchTool
from .knowledge_base import KnowledgeBaseTool
from .document_reader import DocumentReaderTool
from .fact_checker import FactCheckerTool
from .synthesizer import SynthesizerTool


class ToolRegistry:

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._register_defaults()

    def _register_defaults(self):
        for tool in [
            WebSearchTool(),
            KnowledgeBaseTool(),
            DocumentReaderTool(),
            FactCheckerTool(),
            SynthesizerTool(),
        ]:
            self.register(tool)

    def register(self, tool: BaseTool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def all(self) -> List[BaseTool]:
        return list(self._tools.values())

    def descriptions(self) -> str:
        lines = []
        for t in self._tools.values():
            lines.append(f"  * {t.name}: {t.description}")
        return "\n".join(lines)

    def list_names(self) -> list:
        return list(self._tools.keys()) 
