import json
import time
from pathlib import Path
from datetime import datetime, timezone
 
LOG_DIR = Path("./logs")
LOG_DIR.mkdir(exist_ok=True)
 
RESEARCH_LOG = LOG_DIR / "research_log.jsonl"
LLM_CALL_LOG = LOG_DIR / "llm_calls.jsonl"
 
# Approximate USD cost per 1M tokens (input, output). VERIFY against Groq's
# current pricing page before relying on these for real budgeting — they are
# placeholders to make relative cost comparisons possible, not billing-accurate.
MODEL_PRICING = {
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "default": {"input": 0.50, "output": 0.70},
}
 
 
def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = MODEL_PRICING.get(model, MODEL_PRICING["default"])
    cost = (input_tokens / 1_000_000) * pricing["input"] + (output_tokens / 1_000_000) * pricing["output"]
    return round(cost, 6)
 
 
def log_llm_call(model: str, input_tokens: int, output_tokens: int, elapsed_seconds: float, purpose: str = ""):
    """Called once per raw LLM API call (think/synthesize/grade)."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "estimated_cost_usd": estimate_cost(model, input_tokens, output_tokens),
        "purpose": purpose,
    }
    try:
        with open(LLM_CALL_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass  # logging must never break the main request
    return record
 
 
def log_research_call(
    query: str,
    pattern_used: str,
    hops: int,
    api_calls: int,
    confidence: float,
    elapsed_seconds: float,
    total_tokens: int,
    estimated_cost_usd: float,
    sources_count: int = 0,
):
    """Called once per completed researcher.research() call."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "pattern_used": pattern_used,
        "hops": hops,
        "api_calls": api_calls,
        "confidence": confidence,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "total_tokens": total_tokens,
        "estimated_cost_usd": estimated_cost_usd,
        "sources_count": sources_count,
    }
    try:
        with open(RESEARCH_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass
    return record
 
 
def read_all_research_logs() -> list:
    if not RESEARCH_LOG.exists():
        return []
    records = []
    with open(RESEARCH_LOG, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records
 
 
def read_all_llm_call_logs() -> list:
    if not LLM_CALL_LOG.exists():
        return []
    records = []
    with open(LLM_CALL_LOG, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records
