# Postmortem: Building and Debugging an Agentic RAG Researcher

## What this project is

A multi-pattern agentic research system that auto-selects between three reasoning strategies — **ReAct** (iterative reason-and-act), **Self-RAG** (self-reflective retrieval with confidence grading), and **CRAG** (corrective retrieval with automatic web-search fallback) — depending on the shape of the incoming query. It runs a multi-hop loop (up to 5 hops) across tools including web search, a local vector-store knowledge base, a document reader, and a fact-checker, tracked through a scratchpad that records every hop's thought, tool call, observation, and confidence score.

The interesting part of this project isn't the architecture — it's what happened when I actually started running it against real queries. What follows is a log of the real bugs that surfaced, why they happened, how I found and fixed each one, and what I'd do differently next time. I'm writing this up because debugging an agentic LLM system surfaces failure modes that don't show up in typical software engineering, and I think that's more valuable to talk about than the initial build.

---

## Bug log

### 1. Hop counter silently stuck at 1, no matter how many hops actually ran

**Symptom:** `API calls: 5` but `Hops: 1`, every single run, regardless of query complexity.

**Root cause:** The display layer read the wrong attribute name off the result object:
```python
getattr(result, 'total_hops', 1)
```
The dataclass field was actually named `hops`, not `total_hops`. Because `getattr` was called with a default value, the typo never raised an exception — it silently fell back to `1` and displayed that as if it were real data, every time.

**Fix:** `getattr(result, 'hops', 1)` — one-word change.

**Lesson:** `getattr(obj, name, default)` is a footgun when `name` might be wrong, because a typo degrades silently into a plausible-looking wrong value instead of crashing loudly. I now prefer direct attribute access (`result.hops`) for internal dataclasses I control — if the field name is wrong, I want an immediate `AttributeError`, not a quietly wrong number that looks like real output for weeks.

---

### 2. Raw JSON fragments leaking into "sources," corrupting displayed citations

**Symptom:** Sources showed garbage like:
```
→ Web: DDGS---'"web_search", "input": "technologies used in large language models"}'
```

**Root cause:** The response parser assumed an old `tool_name: input` string format and split the LLM's actual JSON action (`{"tool": "web_search", "input": "..."}`) on the **first colon**:
```python
tool_name, _, tool_input = action_raw.partition(":")
```
This chopped a valid JSON object into two garbage halves. `tool_input` ended up holding a malformed fragment of the JSON itself, which then got passed straight through to the search tool and echoed back into the "source" field.

**Fix:** Stop partitioning on `:` entirely. Pass the raw `<action>` string through untouched and let each pattern's own `json.loads()` handle it properly (with a `try/except` around malformed JSON, defaulting to an empty dict rather than crashing).

**Lesson:** When an LLM's structured output format changes (even one you designed), every downstream parser needs to change with it. This bug existed because one part of the prompt evolved (from `tool: input` strings to JSON) but the parser wasn't updated to match — a version-skew problem, just between a prompt and its own parser instead of between two services.

---

### 3. LLM inventing tool names that don't exist in the registry

**Symptom:** The model's own reasoning text said things like *"I will use the `GeneralKnowledgeTool`"* and *"I will add documents using `kb_tool.add_documents(docs)`"* — neither of which exist in the actual tool registry (`web_search`, `knowledge_base`, `document_reader`, `fact_checker`, `synthesizer`).

**Root cause:** The prompt showed the model a list of tool descriptions but never explicitly constrained the `"tool"` field to *only* those exact values, and never penalized deviation. The model pattern-matched on plausible-sounding tool names instead.

**Fix:** Two layers of defense — (1) explicitly enumerate valid tool names in the prompt and forbid inventing new ones, and (2) a code-level fallback in the engine loop: if `registry.get(tool_name)` returns `None`, fall back to `web_search` and **update `tool_name` to reflect what actually ran**, rather than letting the hallucinated name flow into the sources list.

**Lesson:** Prompting alone is a soft constraint — LLMs will still occasionally invent plausible values even when told not to. Anything the agent picks that gets *executed* (a tool call, a file path, a function name) needs a hard validation layer in code, not just an instruction in the prompt. I now think of prompts as guidance and code as the actual guardrail.

---

### 4. LLM confidently doing wrong arithmetic and "correcting" a true fact into a false one

**Symptom:** CRAG's fact-checking pattern was explicitly given the rule "smaller year = older" and the correct data (Python 1991, Java 1995) — and still concluded *"Java is older than Python because 1995 is older than 1991."* The correction mechanism actively broke a correct answer.

**Root cause:** I was asking the LLM to perform a simple numeric comparison as part of a longer free-form reasoning chain, and trusting it to get that arithmetic right just because the rule was stated. LLMs are unreliable at exactly this kind of "small logical operation buried in a paragraph" task, even when explicitly told the rule.

**Fix:** Stop asking the model to do the comparison. Extract candidate years deterministically with a regex, compute the actual comparison in Python, and inject the **verified conclusion** into the prompt as a stated fact the model must report — not something it needs to derive:
```python
verified_note = (
    f"VERIFIED FACT (computed programmatically — do NOT recalculate): "
    f"{older} ({year_facts[older]}) is OLDER than {newer} ({year_facts[newer]})..."
)
```

**Lesson:** This was the single most important realization in the whole project: **for anything with a deterministic right answer, compute it in code and hand the LLM the answer — don't ask the LLM to derive it.** Free-text reasoning is genuinely bad at reliable arithmetic and logical comparison, no matter how explicitly you state the rule. This generalizes way beyond dates — any time an agent needs to compare, count, or calculate something, that's a signal to reach for a deterministic helper function, not a longer prompt.

---

### 5. Tool descriptions leaking into hallucinated fake research pipelines

**Symptom:** For an unrelated query about earthquakes, the model generated an entire fabricated sequence of fake tool calls and invented documents attributed to real institutions:
```python
kb_tool.add_documents([
    {"title": "Earthquake Causes", "source": "United States Geological Survey (USGS)", "text": "..."},
    ...
])
```
None of this had actually happened — the knowledge base was empty the whole time.

**Root cause:** The knowledge base tool's own "empty" message contained literal code syntax:
```python
content = "Knowledge base is empty. Add documents using: kb_tool.add_documents(docs)"
```
This string flowed into the model's own context on every hop (as an observation) and into the final synthesis prompt. Since it looked like real, callable code sitting in the model's own "research findings," the model pattern-matched onto it and began narrating a fictional pipeline built around that exact phrase — inventing plausible institutional sources to go with it.

**Fix:** Rewrite every tool-status message shown to the model as plain, non-code natural language (`"No documents have been added to the internal knowledge base yet."`), and add an explicit anti-fabrication instruction to every synthesis prompt forbidding code syntax and narrated "research processes" in the final answer.

**Lesson:** Anything shown to the model — including your own internal status messages, error strings, and tool descriptions — is part of its context and can be imitated. This is a self-inflicted prompt-injection-shaped bug: I wasn't defending against an external attacker, I was defending against my own code's error messages accidentally teaching the model a bad pattern.

---

### 6. Confidently merging two different real people who share a name

**Symptom:** Asking about a real individual by name pulled back an unrelated real person's LinkedIn/resume data (different institution, different degree, different job history) and presented it as fact with 85% confidence — while a resume for the *actual* intended person existed locally the whole time.

**Root cause:** No verification step ever cross-checked a web-sourced identity against ground truth already available locally. The system had no concept of "this might be the wrong person with the same name" — it just reported whatever the top search result said.

**Fix:** Broadened local-document matching so ambiguous short names still route to the local file when there's a real word-overlap with a known filename, and added an explicit disambiguation instruction to synthesis prompts: if findings describe multiple distinct entities (different people, or an organization mistaken for a person) sharing a name, state the ambiguity rather than merging their attributes.

**Lesson:** This is the failure mode I'd flag as most important if this were a production system rather than a demo — an agent confidently attaching one real person's biography to a different real person's name isn't just "a wrong answer," it's a real accuracy/reputational risk. Any system that resolves named entities against open-web search needs an explicit identity-verification step, not an implicit trust in whatever ranks first.

---

### 7. A documented CLI flag that silently did nothing

**Symptom:** `--forced-pattern selfrag` and `--forced-pattern crag` both produced `Pattern Used: REACT` — the flag was accepted, parsed, and stored in settings, but had zero effect on behavior.

**Root cause:** `PatternSelector.select_pattern()` never actually checked `self.settings.forced_pattern` — it unconditionally ran keyword-based auto-detection every time. A secondary, related bug: several of the auto-detection keyword sets contained multi-word phrases (`"is it true"`, `"pros and cons"`) that could structurally never match, because the detector split queries into single-word tokens and did a set intersection — multi-word strings can't appear in a set of individual words, so those phrases were silently dead weight in the keyword lists the whole time.

**Fix:** Check `forced_pattern` first and short-circuit if set. Separately, add substring matching for any keyword phrase containing a space, since single-word set intersection can't catch them.

**Lesson:** A flag that's parsed and stored but never *read* anywhere is invisible in normal testing — the code runs fine, nothing crashes, and the output just quietly ignores your input. This is why the eval harness (below) mattered: an automated check comparing "requested pattern" against "pattern actually used" caught this in one run, where dozens of manual test queries hadn't.

---

### 8. A tool that crashed on almost every query that wasn't a date comparison

**Symptom:** `fact_checker: error` in the sources list on nearly every query.

**Root cause:** Three independent bugs stacked in one file: (1) `re.findall(...)` was called with no `import re` anywhere in the file; (2) variables (`older_year`, `newer_year`) were only defined inside an `if` branch but referenced unconditionally afterward, causing `UnboundLocalError` on any query that wasn't a date comparison; (3) a malformed dict literal (`{"publisher": publisher, verified: True}`) used a bare undefined identifier instead of a string key.

**Fix:** Add the missing import, guard the year-comparison note behind the same condition that creates it, and fix the dict literal to use a proper string key.

**Lesson:** A tool's `except Exception` catch-all in the engine loop is a double-edged sword — it kept the whole system from crashing, but it also silently hid three real, independent bugs behind a generic `"error"` label for a long time. Broad exception handling at the orchestration layer is good for resilience, but it means bugs inside individual tools need their own explicit tests, because the orchestrator will never surface them clearly on its own.

---

### 9. Answers cut off mid-sentence, leaking raw unclosed XML-like tags into output

**Symptom:** Final answers occasionally just stopped — mid-word, no closing punctuation, sometimes with a fragment of the prompt's own instruction text bleeding into what should have been the answer.

**Root cause:** Long observations plus prompt overhead pushed generations past `max_tokens`, so the response got truncated before reaching the closing `</final_answer>` tag. The parser's tag-extraction regex required the closing tag to be present (`<tag>(.*?)</tag>`), so a truncated response matched nothing, fell back to a default, and the caller then dumped the entire raw, half-finished, tag-less response as the "answer."

**Fix:** Two changes — make the tag extractor tolerant of a missing closing tag (match through end-of-string as a fallback), and give the synthesis step (which tends to produce the longest output) an explicit, generous `max_tokens` rather than relying on a shared default sized for shorter reasoning steps.

**Lesson:** Truncation isn't just "the answer got shorter" — if your parsing logic assumes well-formed output, truncation can turn into a much worse failure (leaking raw prompt scaffolding into user-facing text) than the truncation itself. Parsers for LLM output should always have a degrade-gracefully path for the "model didn't finish" case, since it's a certainty, not an edge case, at scale.

---

## Building an eval harness (and what it taught me about testing LLM systems)

After enough rounds of "paste terminal output, find bug, fix, repeat," I built a small eval suite: 15 test cases in JSONL, each asserting something concrete — required/forbidden substrings in the answer, expected pattern routing, hop/api_call consistency, confidence bounds, expected tool usage, and crash behavior on adversarial/empty input. A runner script executes each case against the live system and writes a timestamped JSON report.

This caught real things that manual spot-checking had missed, including bug #7 above (the forced-pattern flag) on the very first run. But it also taught me something about testing systems with a live web-search step and a non-zero-temperature LLM: **the same test can flip between pass and fail across runs with zero code changes**, purely from what search happens to return and how the model happens to phrase things that particular time. A single pass/fail boolean is the wrong abstraction for that kind of test — a "did this fact get mentioned across 5 runs" pass-rate metric is more honest than a hard gate, and I'd build that in from the start next time rather than retrofitting it.

**Current suite status: 15/15 passing.**

---

## How I'd summarize this in an interview

- *"Tell me about a bug you found hard to track down."* → The CRAG date-arithmetic bug (#4): the model had the correct rule and correct data stated explicitly, and still got the comparison backwards. Diagnosing it meant realizing the fix wasn't a better prompt — it was accepting that LLMs shouldn't be trusted to do deterministic computation inline, and moving that logic into code instead.
- *"Tell me about a subtle bug caused by your own code, not the model."* → The `getattr` hop-counter bug (#1) and the knowledge-base description leak (#5) — both were self-inflicted: a silent-default typo, and an internal status message that accidentally taught the model a bad pattern to imitate.
- *"How do you think about testing AI/LLM systems differently from regular software?"* → The eval harness section above — deterministic assertions work for structural things (routing, hop counts, crash safety), but content correctness against live retrieval needs a pass-rate model, not a pass/fail one.
- *"Tell me about a correctness issue with real-world stakes."* → The entity-confusion bug (#6) — confidently attaching one real person's biography to a different person's name under time pressure is a genuine trust/safety issue, not just a wrong-answer bug, and it's the one I'd prioritize fixing first in any similar system going forward.

---

## What I'd do differently starting over

1. Build the eval harness *first*, even a minimal 5-test version, before generating any real output to eyeball — several of these bugs would have been caught in one run instead of several rounds of manual back-and-forth.
2. Treat every tool's error/status string as untrusted content the model will see and potentially imitate — write them as if they're part of the prompt, because they are.
3. Default to "compute it in code, tell the model the answer" for anything with a deterministic ground truth, rather than trusting free-text reasoning to get there.
4. Add structured logging from day one instead of parsing terminal output by eye — most of these bugs were diagnosed by re-running queries and reading scrollback, which doesn't scale past a handful of debugging sessions.
