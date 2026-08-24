import os
import time
import json as _json
from typing import Optional
from groq import Groq
from utils.cost_tracker import log_llm_call

try:
    from groq import RateLimitError
except ImportError:
    RateLimitError = Exception

try:
    from groq import APIConnectionError
except ImportError:
    APIConnectionError = Exception


try:
    from groq import BadRequestError
except ImportError:
    BadRequestError = Exception

_RETRYABLE_ERRORS = (RateLimitError, APIConnectionError)

# ... inside chat(), in the retry loop, alongside the existing except blocks:
  

# Errors worth retrying with backoff — transient network/server issues,
# not things like bad requests or auth failures which retrying won't fix.
#_RETRYABLE_ERRORS = (RateLimitError, APIConnectionError)


class LLMClient:

    def __init__(self, api_key: str = None, model: str = "openai/gpt-oss-20b", max_tokens: int = 2048):#"groq/compound-mini", max_tokens: int = 2048):#"openai/gpt-oss-120b", max_tokens: int = 2048):

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
        system,
        user,
        history = None,
        max_tokens = None,
        retries = 4,
        purpose = "",
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

                content = response.choices[0].message.content
                if not content or not content.strip():
                    # Groq occasionally returns a 200 with empty content (no exception) —
                    # treat it the same as a transient failure and retry rather than
                    # silently propagating an empty string downstream.
                    last_error = RuntimeError("Empty response content from LLM")
                    wait = min(2 ** attempt, 10)
                    time.sleep(wait)
                    continue
                
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

                return content.strip() #response.choices[0].message.content.strip()

            except (RateLimitError, APIConnectionError) as e:
                if isinstance(e, RateLimitError):
                    error_str = str(e)
                    if "tokens per day" in error_str or "TPD" in error_str:
                        raise RuntimeError(f"Daily token quota exhausted: {error_str}") from e
                last_error = e
                wait = min(2 ** attempt, 10)
                time.sleep(wait)
                continue

            except BadRequestError as e:
                # gpt-oss occasionally emits a native tool-call attempt instead of
                # plain text, even with no tools array in the request — Groq rejects
                # it, but the rejection body contains the model's actual intended
                # output. Recover it and reshape it into the plain-text action format
                # this project's parser expects, instead of losing the turn entirely.
                error_body = getattr(e, "body", None) or {}
                failed_gen = None
                try:
                    failed_gen = error_body.get("error", {}).get("failed_generation")
                except Exception:
                    pass

                if failed_gen and purpose == "think":
                    try:
                        parsed = _json.loads(failed_gen)
                        args = parsed.get("arguments", {})
                        tool = args.get("tool", "")
                        tool_input = args.get("input", "")
                        # Reshape into the <action>{...}</action> format ResponseParser expects
                        recovered = (
                            f"<thought>Recovered from a native tool-call attempt.</thought>\n"
                            f'<action>{{"tool": "{tool}", "input": "{tool_input}"}}</action>'
                        )
                        self.call_count += 1
                        return recovered
                    except Exception:
                        pass

                # Couldn't recover — treat as a normal retryable failure
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
