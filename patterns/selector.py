import re
from typing import TYPE_CHECKING
from .react import ReActPattern
from .selfrag import SelfRAGPattern
from .crag import CRAGPattern

if TYPE_CHECKING:
    from .base import BasePattern
    from config.settings import Settings


CRAG_KEYWORDS = {
    "verify", "verified", "fact", "facts", "factual",
    "true", "false", "accurate", "accuracy", "correct",
    "incorrect", "wrong", "right", "claim", "claims",
    "debunk", "proof", "prove", "disprove", "check",
    "is it true", "is this true", "are they correct",
}

SELFRAG_KEYWORDS = {
    "compare", "comparision", "versus", "vs", "vs.",
    "evaluate", "evaluation", "assess", "assessment",
    "better", "worse", "best", "worst", "rank", "ranking",
    "grade", "rate", "which is", "what is the difference",
    "pros and cons", "advantages", "disadvantages",
    "similarities", "differences",
}


class PatternSelector:
    
    def __init__(self, settings: "Settings"):
        self.settings = settings
        self.patterns: Dict[str, "BasePattern"] = {}

    def register(self, pattern: "BasePattern"):
        self.patterns[pattern.name] = pattern

    def select_pattern(self, query: str) -> "BasePattern":
        """Select the best pattern for the query."""
        forced = getattr(self.settings, "forced_pattern", None)
        if forced:
            return self._build(forced)
        name = self._detect(query)
        return self._build(name)

    def _detect(self, query: str) -> str:
        q_lower = query.lower()
        words = set(re.findall(r"\w+", q_lower))

        crag_score = len(words & CRAG_KEYWORDS)
        selfrag_score = len(words & SELFRAG_KEYWORDS)

        for phrase in CRAG_KEYWORDS:
            if " " in phrase and phrase in q_lower:
                crag_score += 1
        for phrase in SELFRAG_KEYWORDS:
            if " " in phrase and phrase in q_lower:
                selfrag_score += 1

        # Boost score for specific phrases
        if any(phrase in q_lower for phrase in ["is it true", "fact check", "verify"]):
            crag_score += 2

        # Decision logic
        if selfrag_score > crag_score:
            return "selfrag"
        elif crag_score > 0:
            return "crag"
        else:
            return "react"

    def _build(self, name: str) -> "BasePattern":
        if name == "selfrag":
            return SelfRAGPattern(
                relevance_threshold=self.settings.selfrag_relevance_threshold,
                max_retries=self.settings.selfrag_max_retries,
            )
        elif name == "crag":
            return CRAGPattern(
                score_threshold=self.settings.crag_score_threshold,
                max_corrections=self.settings.crag_max_corrections,
            )
        else:
            return ReActPattern()

    @staticmethod
    def explain(query: str) -> str:
        # Simple static explanation
        return "Pattern selected based on query keywords."    
