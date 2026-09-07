import re
from dataclasses import dataclass, field
from typing import List, Optional
from utils.document_excerpt import extract_relevant_excerpt

@dataclass
class Step:
    hop_number: int
    thought: str = ""
    tool_used: str = ""
    tool_input: str = ""
    observation: str = ""
    confidence: float = 0.5
    pattern: str = ""
    grade: float = 0.0
    corrected: bool = False
    needs_correction: bool = False
    needs_calculation: bool = False

@dataclass
class Scratchpad:
    query: str
    steps: List[Step] = field(default_factory=list)

    def add_step(self, step: Step):
        self.steps.append(step)

    @property
    def hop_count(self) -> int:
        """Return number of hops (steps) performed."""
        return len(self.steps)

    def last_confidence(self) -> float:
        valid_steps = [
            s for s in self.steps
            if hasattr(s, 'confidence') and s.confidence is not None
        ]
        if not valid_steps:
            return 0.75
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for i, step in enumerate(valid_steps, start = 1):
            weight = i
            weighted_sum += float(step.confidence) * weight
            total_weight += weight
           
        return round(weighted_sum / total_weight, 2) if total_weight > 0 else 0.75

    def all_observations(self, max_chars: int = 8000, per_step_chars: int = 3000) -> str:
        print("      [DEBUG] scratchpad.py all_observations version: v50-page-anchor")
        page_match = re.search(r'\bpage\s*(\d+)\b', self.query, re.IGNORECASE)


        parts = []
        for step in self.steps:
            obs = step.observation or ""
            if not obs:
                continue
            step_cap = 4500 if step.tool_used == "document_reader" else per_step_chars

            if len(obs) > step_cap:
                if page_match and step.tool_used == "document_reader":
                    page_num = page_match.group(1)
                    page_pattern = re.compile(rf'\[Page\s*{page_num}\b', re.IGNORECASE)
                    pos_match = page_pattern.search(obs)
                    if pos_match:
                        start = max(0, pos_match.start() - 200)
                        end = min(len(obs), pos_match.start() + step_cap)
                        obs = obs[start:end]
                    else:
                        obs = extract_relevant_excerpt(obs, self.query, window_chars=step_cap, max_total_chars=step_cap)
                else:
                    obs = extract_relevant_excerpt(
                        
                        obs, self.query, window_chars=per_step_chars, max_total_chars=per_step_chars,
                    )
            parts.append(obs)
        
        combined = "\n\n[...]\n\n".join(parts)
        
        if len(combined) > max_chars:
            #combined = combined[:max_chars]
            combined = extract_relevant_excerpt(
                combined, self.query, window_chars=max_chars, max_total_chars=max_chars,
            )           
        return combined
        
    def context_for_prompt(self, max_chars: int = 4000) -> str:
        parts = []
        for step in self.steps:
            tool_label = f" (tool: {step.tool_used})" if step.tool_used else ""
            conf_label = f" [confidence: {step.confidence:.2f}]" if step.confidence is not None else ""
            parts.append(f"Hop {step.hop_number}{tool_label}{conf_label}: {step.observation}")
        combined = "\n\n".join(parts)
        if len(combined) > max_chars:
            combined = combined[-max_chars:]  # keep the most recent context, trim the oldest
        return combined

    
@dataclass
class ResearchResult:
    query: str
    final_answer: str
    pattern_used: str
    tool_used: Optional[str] = None  
    confidence: float = 0.0
    hops: int = 0
    sources: List[str] = field(default_factory=list)
    api_calls: int = 0
    steps: List[Step] = field(default_factory=list)

    def __post_init__(self):
        if isinstance(self.pattern_used, list):
            self.pattern_used = "+".join(str(p) for p in self.pattern_used)
