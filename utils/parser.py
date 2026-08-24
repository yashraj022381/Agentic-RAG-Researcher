import re
from dataclasses import dataclass
from typing import Optional

@dataclass
class ParsedResponse:
    thought: str = ""
    action: str = ""
    action_input: str = ""
    observation: str = ""
    final_answer: str = ""
    confidence: float = 0.0
    sources: list = None
    grade: float = 0.0
    raw: str = ""

    def __post_init__(self):
        if self.sources is None:
            self.sources = []

class ResponseParser:

    @staticmethod
    def extract(tag: str, text: str, default: str = "") -> str:
        pattern = rf"<{tag}>(.*?)(?:</{tag}>|$)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else default

    @staticmethod
    def extract_float(tag: str, text: str, default: float = 0.0) -> float:
        raw = ResponseParser.extract(tag, text)
        nums = re.findall(r"\d+\.?\d*", raw)
        if not nums:
            return default
        val = float(nums[0])

        return val / 100.0 if val > 1.0 else val

    
    @classmethod
    def parse(cls, text: str) -> ParsedResponse:
        if not text or not isinstance(text, str):
            return ParsedResponse(raw=text or "")

        p = ResponseParser

        action_raw = p.extract("action", text)
        tool_name = action_raw.strip()
        tool_input = p.extract("action_input", text)

        sources_raw = p.extract("sources", text)
        sources = [s.strip() for s in sources_raw.splitlines() if s.strip()]

        final_answer = p.extract("final_answer", text) or text

        return ParsedResponse(
            thought=p.extract("thought", text),
            action=tool_name,
            action_input=tool_input.strip(),
            observation=p.extract("observation", text),
            final_answer=final_answer,
            confidence=p.extract_float("confidence", text, default=0.5),
            sources=sources,
            grade=p.extract_float("grade", text, default=0.5),
            raw=text,
        )
    
    """
    def parse(cls, text: str) -> ParsedResponse:

        if not text or not isinstance(text, str):
            return ParsedResponse(raw = text or "")
        
        p = ResponseParser

        action_raw = p.extract("action", text)
        if ":" in action_raw:
             tool_name, _, tool_input = action_raw.partition(":")

        else:
            tool_name, tool_input = action_raw, ""

        sources_raw = p.extract("sources", text)
        sources = [s.strip() for s in sources_raw.splitlines() if s.strip()]

        final_answer = p.extract("final_answer", text) or text

        return ParsedResponse(
            thought = p.extract("thought", text),
            action = tool_name.strip(),
            action_input = tool_input.strip(),
            observation = p.extract("observation", text),
            final_answer = final_answer,
            confidence = p.extract_float("confidence", text, default = 0.5),
            sources = sources,
            grade = p.extract_float("grade", text, default = 0.5),
            raw = text,
        )
    """
