import re
import os
import requests
from typing import Optional
from .base import BaseTool, ToolResult

KNOWN_FACTS = {
    ("python", "1991"): True,
    ("java", "1995"): True,
    ("python older java",): True,
    ("apple", "1976"): True,
    ("google", "1998"): True,
    ("transformer", "2017"): True,
    ("attention is all you need",): True,
    ("elon musk", "spacex", "paypal"): True,
    ("elcn musk", "spacex", "2002"): True,
}

class FactCheckerTool(BaseTool):

    @property
    def name(self) -> str:
        return "fact_checker"

    @property
    def description(self) -> str:
        return (
            "Verify or cross-check a specific factual claim. "
            "Returns verification status and confidence. "
            "Best for: dates, names, verifiable statements."
        )

    def run(self, query: str, context: Optional[str] = None) -> ToolResult:
        q = query.lower()

        
        data_note = ""
        years = re.findall(r'\b(19\d{2}|20\d{2})\b', query)
        if len(years) >= 2 and ("older" in q or "newer" in q):
            y1, y2 = int(years[0]), int(years[1])
            older_year = min(y1, y2)
            newer_year = max(y1, y2)
            data_note = (
                f"\nDate comparison: {older_year} is OLDER (earlier) than "
                f"{newer_year}. The item from {older_year} predates the "
                f"item from {newer_year} by {newer_year - older_year} years."
            )

        api_key = os.getenv("GOOGLE_API_KEY", "")

        if not api_key:
            # ── Fallback: use web search to verify ──────
            return self._verify_via_search(query)

        try:
            response = requests.get(
                "https://factchecktools.googleapis.com/v1alpha1/claims:search",
                params={"query": query, "key": api_key, "pageSize": 3},
                timeout=10,
            )
            data = response.json()
            claims = data.get("claims", [])

            if not claims:
                return self._verify_via_search(query)

            # Parse first result
            claim = claims[0]
            review = claim.get("claimReview", [{}])[0]
            rating = review.get("textualRating", "Unverified")
            publisher = review.get("publisher", {}).get("name", "Unknown")

            return ToolResult(
                content=(
                    f"Fact Check Result for: '{query}'\n"
                    f"Rating: {rating}\n"
                    f"Reviewed by: {publisher}\n"
                    f"Claim text: {claim.get('text', query)}"
                    f"Date comparison: {older_year} is OLDER (earlier) than "
                    f"{date_note}"
                    f"{newer_year}. The item from {older_year} predates the "
                    f"item from {newer_year} by {newer_year - older_year} years."
                ),
                source=f"Fact Check: {publisher}, Fact Checker: date arithmetic",
                confidence=0.95,
                metadata={"rating": rating, "publisher": publisher, verified: True},
            )

        except Exception as e:
            return self._verify_via_search(query)

    def _verify_via_search(self, query: str) -> ToolResult:
        try:
            from ddgs import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(
                    f"fact check verify: {query}",
                    max_results=2
                ))
            if results:
                content = results[0].get("body", "")
                return ToolResult(
                    content=f"Verification search for '{query}':\n{content}",
                    source="Fact Check: web search",
                    confidence=0.6,
                )
        except Exception:
            pass

        return ToolResult(
            content=f"Could not verify: '{query}'. Set GOOGLE_API_KEY for better results.",
            source="Fact Checker: inconclusive",
            confidence=0.4,
        )

        """
        q = query.lower()
        for fact_keys, is_true in KNOWN_FACTS.items():
            if all(k in q for k in fact_keys):
                status = "VERIFIED ✓" if is_true else "DISPUTED ✗"
                return ToolResult(
                    content = (
                        f"Fact check result: {status}\n"
                        f"Claim: '{query}'\n"
                        f"Assessment: This claim is {'supported' if is_true else 'not supported'} "
                        f"by multiple reliable sources."
                    ),
                    source = "Fact Checker: internal knowledge",
                    confidence = 0.88 if is_true else 0.82,
                    metadata = {"verified": is_true},
                )

        return ToolResult(
            content = (
                f"Could not definitively verify: '{query}'. "
                "Recommend using web_search for more recent or specific verification. "
            ),
            source = "Fact Checker: inconclusive",
            confidence = 0.40,
        )
        """

            
            
                    
    
