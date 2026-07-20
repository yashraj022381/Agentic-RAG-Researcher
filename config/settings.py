from dataclasses import dataclass, field
from typing import Optional
import os

@dataclass
class Settings:

    api_key: str = field(default_factory = lambda: os.getenv("GROQ_API_KEY", ""))
    model: str = "llama-3.1-8b-instant"
    max_tokens: int = 500

    forced_pattern: Optional[str] = None

    max_hops: int = 6
    confidence_threshold: float = 0.75
    min_hops: int = 1

    selfrag_relevance_threshold: float = 0.10
    selfrag_max_retries: int = 1

    crag_score_threshold: float = 0.50
    crag_max_corrections: int = 1

    verbose: bool =  False

    def validate(self):
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY is not set!\n"
                "Please create a .env file your Groq API key.\n"
                "Example: GROQ_API_KEY = gsk_xxxxxxxxx"
            )


        if self.max_hops < 1:
            self.max_hops = 6
