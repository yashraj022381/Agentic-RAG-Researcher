import re
from typing import TYPE_CHECKING, Dict, Optional
from .react import ReActPattern
from .selfrag import SelfRAGPattern
from .crag import CRAGPattern

if TYPE_CHECKING:
    from .base import BasePattern
    from config.settings import Settings


CRAG_KEYWORDS = {
    "verify", "verified", "fact", "facts", "factual",
    "false", "accurate", "accuracy", "incorrect",
    "wrong", "right", "claim", "claims", "contradiction",
    "debunk", "proof", "prove", "disprove", "check",
    "is it true", "is this true", "are they correct",
    "exact specifications", "exact dimensions", "installation",
    "specifications for", "precise dimensions",
    "exists", "exist", "missing", "null", "schema", "invalid", "outlier",
    "does the document mention", "is there any mention", "unsupported", "false premise"
}

SELFRAG_KEYWORDS = {
    "compare", "comparision", "versus", "vs", "vs.",
    "evaluate", "evaluation", "assess", "assessment",
    "better", "worse", "best", "worst", "rank", "ranking",
    "which is", "what is the difference",
    "pros and cons", "advantages", "disadvantages",
    "similarities", "step-by-step", "breakdown", "mathematical", "derivation", "formula",
    "exact definition", "technical terms", "verbatim", "cite source", "summarize"
}


class PatternSelector:
    
    def __init__(self, settings: "Settings"):
        self.settings = settings
        self.patterns: Dict[str, "BasePattern"] = {}

    def register(self, pattern: "BasePattern"):
        self.patterns[pattern.name] = pattern

    #def select_pattern(self, query: str) -> "BasePattern":
    def select_pattern(self, query, plan: Optional[dict] = None) -> "BasePattern":
        """Select the best pattern for the query."""
        forced = getattr(self.settings, "forced_pattern", None)
        if forced:
            return self._build(forced)
        if plan and plan.get("pattern") in ("react", "crag", "selfrag"):
            return self._build(plan["pattern"])
        name = self._detect(query)
        return self._build(name) #and self._build(self._detect(query))  # keyword fallback if planning failed
    

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

        extraction_signal = any(re.search(rf'\b{re.escape(w)}\b', q_lower) for w in [
            "extract", "pdf", "document", "chapter", "resume", "excerpt",
            "in the document", "in chapter",
        ]) or any(phrase in q_lower for phrase in [
           "according to", "from the", "in the",
        ])

        
        if extraction_signal:
            selfrag_score = 0

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
