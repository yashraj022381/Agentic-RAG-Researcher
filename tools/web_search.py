import os
import re
from typing import Optional
from .base import BaseTool, ToolResult

MOCK_WEB = {
    "transformer": (
        "The Transformer architecture was introduced in the 2017 paper "
        "'Attention Is All You Need' by Vaswani et al. at Google Brain. "
        "Key authors: Ashish Vaswani, Noam Shazeer, Niki Parmar, "
        "Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, "
        "and Illia Polosukhin.",
        "https://arxiv.org/abs/1706.03762",
    ),
    "attention": (
        "The Transformer architecture uses self-attention mechanisms. "
        "Introduced in 2017 by Vaswani et al. at Google Brain.",
        "https://arxiv.org/abs/1706.03762",
    ),
    "apple": (
        "Apple Inc. was founded on April 1, 1976, by Steve Jobs, "
        "Steve Wozniak, and Ronald Wayne in Cupertino, California. "
        "Apple was founded 22 years before Google.",
        "https://en.wikipedia.org/wiki/Apple_Inc.",
    ),
    "google": (
        "Google was founded on September 4, 1998, by Larry Page and "
        "Sergey Brin while PhD students at Stanford University.",
        "https://en.wikipedia.org/wiki/Google",
    ),
    "founding": (
        "Apple was founded April 1, 1976 by Steve Jobs and Steve Wozniak. "
        "Google was founded September 4, 1998 by Larry Page and Sergey Brin. "
        "Apple started first, 22 years before Google.",
        "https://en.wikipedia.org/wiki/Founding_stories",
    ),
    "founded": (
        "Apple Inc. founded 1976. Google founded 1998. "
        "Apple is older than Google by 22 years. "
        "Steve Jobs founded Apple. Larry Page and Sergey Brin founded Google.",
        "https://en.wikipedia.org/wiki/Founded",
    ),
    "compare": (
        "Comparing Apple vs Google: Apple was founded in 1976 by Steve Jobs. "
        "Google was founded in 1998 by Larry Page. "
        "Apple started first by 22 years.",
        "https://comparison.example.com",
    ),
    "comparison": (
        "Apple (founded 1976) vs Google (founded 1998). "
        "Apple came first. Both are technology giants.",
        "https://comparison.example.com",
    ),
    "started": (
        "Apple was the first to start, founded in 1976. "
        "Google came later in 1998. Apple predates Google by 22 years.",
        "https://history.example.com",
    ),
    "story": (
        "Apple founding story: Steve Jobs, Steve Wozniak founded Apple 1976. "
        "Google founding story: Larry Page, Sergey Brin founded Google 1998.",
        "https://stories.example.com",
    ),
    "stories": (
        "The founding story of Apple began in 1976 with Steve Jobs. "
        "The founding story of Google began in 1998 with Larry Page. "
        "Apple was founded first.",
        "https://stories.example.com",
    ),
    "python": (
        "Python was created by Guido van Rossum and first released in 1991. "
        "It is a high-level, general-purpose programming language.",
        "https://en.wikipedia.org/wiki/Python_(programming_language)",
    ),
    "java": (
        "Java was created by James Gosling at Sun Microsystems "
        "and released in 1995. Platform-independent language.",
        "https://en.wikipedia.org/wiki/Java_(programming_language)",
    ),
    "older": (
        "Python (1991) is older than Java (1995). "
        "Python was created before Java by 4 years.",
        "https://history.example.com",
    ),
    "created": (
        "Python was created in 1991 by Guido van Rossum. "
        "Java was created in 1995 by James Gosling at Sun Microsystems. "
        "Python is older.",
        "https://languages.example.com",
    ),
    "photosynthesis": (
        "Photosynthesis converts light energy into glucose. "
        "Equation: 6CO₂ + 6H₂O + light → C₆H₁₂O₆ + 6O₂.",
        "https://en.wikipedia.org/wiki/Photosynthesis",
    ),
    "llm": (
        "Large Language Models (LLMs) are built on the Transformer "
        "architecture, pre-trained on massive text corpora using "
        "self-supervised learning.",
        "https://en.wikipedia.org/wiki/Large_language_model",
    ),
    "large": (
        "Large Language Models use Transformer architecture. "
        "Key researchers: Vaswani et al. introduced Transformers in 2017.",
        "https://en.wikipedia.org/wiki/Large_language_model",
    ),
    "language": (
        "Language models use Transformer architecture with self-attention. "
        "GPT, BERT, Claude are all based on Transformers.",
        "https://en.wikipedia.org/wiki/Language_model",
    ),
    "vaccine": (
        "Vaccines train the immune system. mRNA vaccines deliver "
        "genetic instructions for cells to produce a viral protein.",
        "https://www.cdc.gov/vaccines",
    ),
    "climate": (
        "Climate change refers to long-term shifts in global temperatures "
        "driven by human activities since the 1800s.",
        "https://www.un.org/en/climatechange",
    ),
}


class WebSearchTool(BaseTool):
    
    def __init__(self):
        self.api_key = os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            print(" TAVILY_API_KEY not found. Using DuckDuckGo fallback.")
            
    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "Search the internet for current, real-time information. "
            "Best for: recent events, facts, people, dates, comparisons."
        )

    @staticmethod
    def _sanitize_query(query: str) -> str:
        """Defensive cleanup in case upstream parsing leaks raw JSON fragments."""
        q = query.strip()
        m = re.search(r'"input"\s*:\s*"([^"]+)"', q)
        if m:
            return m.group(1)
        q = re.sub(r'^[\'"]?\w+[\'"]?\s*,\s*', '', q)
        q = q.rstrip('}').strip(' \'"')
        return q or query

    def run(self, query: str, context = None) -> ToolResult:
        query = self._sanitize_query(query)
        api_key = os.getenv("TAVILY_API_KEY", "")

        search_query = query

        if not api_key:
            #return self._duckduckgo_search(query)
            return self._duckduckgo_search(query)

        if "compare" in query.lower() or "founding" in query.lower():
            search_query = query + "wikipedia founded date"

        try:
            import requests
            response = requests.post(
                "https://api.tavily.com/search",
                json = {
                    "api_key": api_key,
                    "query": query,
                    "search_depth": "basic",
                    "max_results": 3,
                   # "include_answer": True,
                },
                timeout = 10,
            )
            data = response.json()
            results = data.get("results", [])
            
            if not results:
                return self._fallback(query)

            content = "\n\n".join([
                f"{r.get('title', '')}:{r.get('content', '')}"
                for r in results[:3]
            ])
            source_url = results[0].get("url", "https://tavily.com")

            return ToolResult(
                content = content,
                source = f"Web: {source_url}",
                confidence = 0.85,
                metadata = {"query": query, "num_results": len(results)},
            )
                 
        except Exception as e:
            return self._fallback(query, error = str(e))

           

        
    def _duckduckgo_search(self, query: str) -> ToolResult:

        try:
            from ddgs import DDGS

            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results = 3))

            if not results:
                return self._fallback(query)

            content = "\n\n".join([
                f"{r.get('title', '')}: {r.get('body', '')}"
                for r in results[:3]
            ])

            return ToolResult(
                content=content,
                source=f"Web: DDGS---'{query}'",
                confidence=0.80,
                metadata={"query": query, "engine": "ddgs"},
            )

        except ImportError:
            return self._fallback(
                query,
                error = "Run: pip install ddgs"
            )

        except Exception as e:
            return self._fallback(query, error = str(e))

    def _fallback(self, query: str, error: str = "") -> ToolResult:
        return ToolResult(
            content = (
                f"Web search for '{query}' could not be completed."
                f"{error if error else 'No results found.'}"
        ),
        source = "Web: search-failed",
        confidence = 0.1,
        metadata = {"query": query, "error": error},
            )
            

       
