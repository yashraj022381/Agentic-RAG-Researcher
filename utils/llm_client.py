import os
import time
from typing import Optional
from groq import Groq

try:
    from groq import RateLimitError
except ImportError:
    RateLimitError = Exception

try:
    from groq import APIConnectionError
except ImportError:
    APIConnectionError = Exception

from utils.cost_tracker import log_llm_call

# Errors worth retrying with backoff — transient network/server issues,
# not things like bad requests or auth failures which retrying won't fix.
_RETRYABLE_ERRORS = (RateLimitError, APIConnectionError)


class LLMClient:

    def __init__(self, api_key: str = None, model: str = "llama-3.1-8b-instant", max_tokens: int = 2048):
        if api_key is None:
            api_key = os.getenv("GROQ_API_KEY")

        self.client = Groq(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.call_count = 0

        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0

    def reset_usage_counters(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0

    def chat(
        self,
        system: str,
        user: str,
        history: Optional[list] = None,
        max_tokens: Optional[int] = None,
        retries: int = 4,
        purpose: str = "",
    ) -> str:

        messages = []
        if history:
            messages.extend(history)

        messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        tokens_for_this_call = max_tokens if max_tokens is not None else self.max_tokens

        last_error = None
        for attempt in range(retries):
            try:
                start = time.time()
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=tokens_for_this_call,
                    messages=messages,
                    temperature=0.7,
                )
                elapsed = time.time() - start
                self.call_count += 1

                usage = getattr(response, "usage", None)
                input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
                output_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

                self.total_input_tokens += input_tokens
                self.total_output_tokens += output_tokens

                call_record = log_llm_call(
                    model=self.model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    elapsed_seconds=elapsed,
                    purpose=purpose,
                )
                self.total_cost_usd += call_record["estimated_cost_usd"]

                return response.choices[0].message.content.strip()

            except _RETRYABLE_ERRORS as e:
                last_error = e
                wait = min(2 ** attempt, 10)
                time.sleep(wait)
                continue

            except Exception:
                raise

        raise RuntimeError(f"LLM call failed after {retries} retries: {last_error}")

    def grade(self, prompt: str) -> float:
        reply = self.chat(
            system=(
                "You are a strict grader. "
                "Respond with ONLY a decimal number between 0.0 and 1.0. "
                "Nothing else. No words, no explanation."
            ),
            user=prompt,
            purpose="grade",
        )
        try:
            score = float(reply.strip())
            return max(0.0, min(1.0, score))
        except ValueError:
            return 0.5
