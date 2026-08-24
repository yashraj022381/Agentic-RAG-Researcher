from dataclasses import dataclass, field
from typing import List, Optional

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
            #if hasattr(step, 'confidence') and step.confidence is not None:
            weight = i
            weighted_sum += float(step.confidence) * weight
            total_weight += weight
            #total += float(step.confidence)
            #count += 1
        return round(weighted_sum / total_weight, 2) if total_weight > 0 else 0.75

    def all_observations(self, max_chars: int = 6000) -> str:
        combined = "\n".join([step.observation for step in self.steps if step.observation])
        if len(combined) > max_chars:
            combined = combined[:max_chars]
        return combined
        
    def context_for_prompt(self, max_chars: int = 4000) -> str:
        #return "\n\n".join([f"Hop {step.hop_number}: {step.observation}" for step in self.steps])
        #parts = [f"Hop {step.hop_number}: {step.observation}" for step in self.steps]
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
