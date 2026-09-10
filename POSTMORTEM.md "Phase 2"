# Postmortem: Building and Debugging an Agentic RAG Researcher

## What this project is

A multi-pattern agentic research system that auto-selects between three reasoning strategies — **ReAct** (iterative reason-and-act), **Self-RAG** (self-reflective retrieval with confidence grading), and **CRAG** (corrective retrieval with automatic web-search fallback) — depending on the shape of the incoming query. It runs a multi-hop loop across tools including web search, a document reader (with OCR fallback for scanned PDFs), a CSV analyzer that executes real pandas computations instead of asking the LLM to do arithmetic, and a calculator, tracked through a scratchpad that records every hop's thought, tool call, observation, and confidence score.

This postmortem covers two phases. **Phase 1** is the original build-and-stabilize pass. **Phase 2** is a much longer, harder debugging arc that started when I built a 9-case Gemini-generated test suite specifically designed to probe pattern selection, tool routing, and multi-hop reasoning under adversarial conditions (false premises, schema mismatches, cross-source synthesis, verbatim citation) — and kept finding real bugs every time I thought it was done. I'm keeping both phases in one document because the contrast is the point: Phase 1 was "make it work," Phase 2 was "make it work *reliably*, under the specific ways agentic systems fail that don't show up until you stress-test routing logic across dozens of adversarial cases."

---

## Phase 1: Bug log (original build-and-stabilize pass)

### 1. Hop counter silently stuck at 1, no matter how many hops actually ran

**Symptom:** `API calls: 5` but `Hops: 1`, every single run, regardless of query complexity.

**Root cause:** The display layer read the wrong attribute name off the result object:
```python
getattr(result, 'total_hops', 1)
```
The dataclass field was actually named `hops`, not `total_hops`. Because `getattr` was called with a default value, the typo never raised an exception — it silently fell back to `1` and displayed that as if it were real data, every time.

**Fix:** `getattr(result, 'hops', 1)` — one-word change.

**Lesson:** `getattr(obj, name, default)` is a footgun when `name` might be wrong, because a typo degrades silently into a plausible-looking wrong value instead of crashing loudly. I now prefer direct attribute access for internal dataclasses I control — if the field name is wrong, I want an immediate `AttributeError`, not a quietly wrong number that looks like real output for weeks.

---

### 2. Raw JSON fragments leaking into "sources," corrupting displayed citations

**Symptom:** Sources showed garbage like `→ Web: DDGS---'"web_search", "input": "..."}'`.

**Root cause:** The response parser assumed an old `tool_name: input` string format and split the LLM's actual JSON action on the **first colon**, chopping a valid JSON object into two garbage halves.

**Fix:** Stop partitioning on `:` entirely. Pass the raw `<action>` string through untouched and let each pattern's own `json.loads()` handle it, with a `try/except` defaulting to an empty dict rather than crashing.

**Lesson:** When an LLM's structured output format changes, every downstream parser needs to change with it — a version-skew problem, just between a prompt and its own parser instead of two services.

---

### 3. LLM inventing tool names that don't exist in the registry

**Symptom:** The model's own reasoning referenced `GeneralKnowledgeTool` and `kb_tool.add_documents(docs)` — neither exists in the registry.

**Root cause:** The prompt showed tool descriptions but never hard-constrained the `"tool"` field to those exact values.

**Fix:** Two layers — explicitly enumerate valid tool names and forbid inventing new ones in the prompt, plus a code-level fallback: if `registry.get(tool_name)` returns `None`, fall back to `web_search` and update `tool_name` to reflect what actually ran, rather than letting the hallucinated name flow into sources.

**Lesson:** Prompting alone is a soft constraint. Anything the agent picks that gets *executed* needs a hard validation layer in code — prompts are guidance, code is the guardrail.

---

### 4. LLM confidently doing wrong arithmetic and "correcting" a true fact into a false one

**Symptom:** Given the explicit rule "smaller year = older" and correct data (Python 1991, Java 1995), the system still concluded *"Java is older than Python because 1995 is older than 1991."*

**Root cause:** Trusting the LLM to perform a numeric comparison correctly inside a longer free-form reasoning chain, just because the rule was stated.

**Fix:** Stop asking the model to do the comparison. Extract candidate years deterministically with regex, compute the comparison in Python, and inject the **verified conclusion** into the prompt as a stated fact the model must report, not derive.

**Lesson:** The single most important realization of Phase 1, and one that Phase 2 ended up re-learning at much larger scale (see the CSV-analyzer work below): **for anything with a deterministic right answer, compute it in code and hand the LLM the answer — don't ask the LLM to derive it.**

---

### 5–9. (Condensed)

A knowledge-base tool's internal status message accidentally taught the model a bad citation-formatting pattern it then imitated in real answers; a `--forced-pattern` CLI flag was parsed and stored but never actually read by the selector, so it silently did nothing; a fact-checker tool crashed on almost every non-date-comparison query from three independent bugs stacked in one file (missing `import re`, a variable only defined inside an `if` branch but referenced unconditionally after, and a malformed dict literal using a bare identifier instead of a string key); and truncated LLM responses that missed a closing `</final_answer>` tag caused the parser to fall through to dumping the entire raw, half-finished response — including leaked prompt scaffolding — as the "answer," fixed by making the tag extractor tolerant of a missing closing tag and giving synthesis calls a more generous `max_tokens`.

**Common thread across Phase 1:** most of these were bugs I could find by eyeballing terminal output on a handful of manual queries. That stopped being true in Phase 2.

---

## Phase 2: Debugging an adversarial, multi-pattern test suite

Phase 2 started with a 9-case test suite (later grown to 15) built specifically to probe multi-hop reasoning: false-premise detection, schema mismatches, cross-source synthesis requiring both a local document and live web data, verbatim citation compliance, and complex derived computation. Running this suite surfaced a category of bugs Phase 1 never touched: **routing and orchestration bugs**, where every individual tool worked correctly in isolation, but the *decision* about which pattern or tool to use — made before any tool ever ran — was wrong, and the wrongness only showed up as a cascade several hops later.

### 10. The pattern classifier silently stopped being able to return a confident answer at all

**Symptom:** Every query, regardless of how clearly it matched CRAG, Self-RAG, or ReAct's signature, deferred to a slower, less reliable LLM-based fallback classifier — for weeks, without ever visibly failing.

**Root cause:** A tie-breaking fix introduced this line:
```python
best_score = max(scores, key=scores.get)   # returns a KEY, not a value!
tied_patterns = [p for p, s in scores.items() if s == best_score]
```
`max(dict, key=...)` returns a dictionary *key* (a string like `"crag"`), not the numeric score. The next line then compared an `int` to that `str` — always `False` — so `tied_patterns` was permanently empty, which triggered a "defer, this is ambiguous" fallback on every single call. The fast, deterministic classifier had been dead code for a long time and nothing crashed to reveal it.

**Fix:** `best_score = max(scores.values())`.

**Lesson:** A single-character-class-of-bug (`scores` vs `scores.values()`) can silently disable an entire subsystem while every individual test still "passes" in the sense of not crashing — it just quietly takes the slower, worse path every time. This is the routing-layer equivalent of bug #1's `getattr` typo, but with a much bigger blast radius, because it sat upstream of every other decision in the system.

---

### 11. A CSV precheck shortcut bypassed the entire reasoning loop for computation-heavy queries

**Symptom:** Queries that explicitly asked to "engineer a variance column" and "compute per capita, step-by-step" were answered by a fast local shortcut that couldn't actually perform multi-column derived arithmetic — it just apologized that the raw numbers weren't present, even though they were.

**Root cause:** The condition gating whether a query skipped the full agentic loop in favor of a fast CSV-only shortcut checked `needs_document` and `needs_web`, but never checked `needs_computation`:
```python
skip_precheck = bool(forced_pattern) or (plan and (plan.get("needs_both") or plan.get("pattern") in ("crag", "selfrag")))
```
Any query needing real multi-step computation fell through the shortcut instead of reaching the loop where a calculator or a proper derived-metrics tool could actually help.

**Fix:** Add `plan.get("needs_computation")` to the skip condition, and — separately — build an actual deterministic derived-arithmetic operation into the CSV tool (vectorized pandas subtraction/division across every row in one call) rather than relying on the loop to force 14 individual calculator calls for a 7-row, 2-metric dataset.

**Lesson:** A "fast path" shortcut is a routing decision like any other, and it needs the same completeness checking as the main router — an incomplete gate condition doesn't fail loudly, it just silently routes some fraction of real queries around your actual reasoning system.

---

### 12. Word-boundary regex silently failed on the exact column names it was written to catch

**Symptom:** A detector meant to recognize "engineer a **Spend_Variance** column" as a computation request never fired for that exact phrase.

**Root cause:** `\bvariance\b` in regex treats underscores as word characters — so `"spend_variance"` is one unbroken token as far as `\b` is concerned, and `variance` embedded inside it can never match a word-boundary pattern.

**Fix:** Normalize underscores to spaces before matching (`query.lower().replace('_', ' ')`), the same technique already used elsewhere in the codebase for column-name comparisons, just not consistently applied to this one detector.

**Lesson:** A fix pattern that's correct in one place in a codebase doesn't automatically propagate to structurally identical code elsewhere — worth a deliberate grep for the same anti-pattern (`\b...\b` against raw, un-normalized text containing underscored identifiers) rather than assuming one fix closes the whole class of bug.

---

### 13. A generic column name turned "the company's crypto bonus" into a false CSV match

**Symptom:** A query with nothing to do with any spreadsheet ("the company's 2026 Cryptocurrency Staking Bonus program") kept triggering an unnecessary CSV-analysis detour.

**Root cause:** The query's possessive `"company's"` tokenizes to the bare word `"company"`, which happened to be an actual column name in one of the uploaded CSVs. A single coincidental shared word was treated as sufficient evidence the query was about that spreadsheet.

**Fix:** Maintain a blocklist of overly generic column-name words (`name`, `date`, `company`, `status`, `id`, etc.) that don't count as a match on their own, and — for compound, multi-word column names — require overlap on *more than one* of the column's constituent words before treating it as a real signal, not just one common word.

**Lesson:** This exact failure mode (a single generic word coincidentally matching) recurred at least three separate times against three different generic words (`"table"`, `"actual"`, `"company"`) before I fixed the underlying pattern instead of blocklisting words one at a time as they surfaced. The real fix wasn't "add another word to the blocklist" — it was raising the evidence bar structurally for any signal derived from a single shared token.

---

### 14. Tool input silently carried over from an abandoned tool selection into the wrong tool

**Symptom:** A web search inexplicably searched for the literal string `"142.0 - 86.9"` — the search engine parsed it as something resembling an IP address and a version number, returning completely irrelevant results.

**Root cause:** An earlier override block built a calculator expression and stored it as a pending `override_tool_input`. A *later* override block then reassigned the tool selection away from calculator to `web_search` — but nothing cleared the now-stale `override_tool_input`, so the centralized input-assignment logic picked it up as if it belonged to the new tool:
```python
if override_tool_input is not None:
    tool_input = override_tool_input   # still holds the abandoned calculator expression
```

**Fix:** Reset `override_tool_input = None` any time a later block reassigns the tool selection away from whatever the pending input was built for.

**Lesson:** State that's set once and read much later, across multiple conditional branches that can each reassign the "current plan," is exactly where staleness bugs live. The fix that actually held was making tool-input construction happen once, centrally, based on the *final* resolved tool — not scattered across every branch that might change the selection.

---

### 15. Scratchpad truncation silently deleted the one page the question was actually about

**Symptom:** A model confidently stated *"the excerpt does not include page 4"* — while page 4's full text was sitting, verbatim, in that exact hop's own tool observation just a few lines earlier in the same debug log.

**Root cause:** Multi-hop context aggregation truncated each step's observation to a fixed character budget using plain `text[:N]` slicing. For a six-page PDF where the relevant content sat on page four, the first N characters were pages one through three — page four never made it into what synthesis actually saw, despite genuinely being retrieved.

**Fix:** Replace blind truncation with query-aware excerpting — when the query names a specific page number, anchor directly to that page's literal marker in the document text; otherwise fall back to keyword-anchored windowing (reusing an excerpting helper that already existed for a different purpose) rather than always keeping only the start of the text.

**Lesson:** A model correctly reporting what it can't see is *worse* than a model incorrectly guessing, because it's confidently, plausibly wrong in a way that looks like careful honesty. Any context-aggregation step that truncates retrieved content needs to be truncating *toward relevance*, never toward document position — "keep the first N characters" is almost never the right default for anything longer than a couple of paragraphs.

---

### 16. The model tried to call a tool during the answer-writing step, with no tools available

**Symptom:** A "final answer" came back as either a raw, leaked JSON tool-call blob (`{"action": "search", "parameters": {...}}`) shown directly to the user, or a hard API-level error (`Tool choice is none, but model called a tool`) that surfaced as an ugly stack-trace-style string instead of any answer at all.

**Root cause:** The synthesis step has no tools registered — its whole job is to write prose from what's already been gathered — but the model's underlying instinct to keep researching didn't switch off just because the calling context changed. Nothing checked whether the "final answer" it produced actually looked like an answer before showing it to the user.

**Fix:** Detect both failure shapes — JSON-looking output, and the specific API error string — and retry once with an explicit "no tools are available, you already have everything you need, write plain prose" instruction. If that fails too, fall back to a clearly-labeled raw-findings dump rather than either raw JSON or a bare error message.

**Lesson:** "The model won't call a tool if none are registered" is not a safe assumption — check what actually comes back, not just what's structurally supposed to be possible.

---

### 17. Deployment: the wrong package on PyPI, and a system-vs-Python dependency confusion

**Symptom:** `pip install tesseract-ocr` failed to compile from source on Streamlit Community Cloud with a missing `leptonica` header.

**Root cause:** `tesseract-ocr` on PyPI is not the OCR engine — it's an unrelated, effectively abandoned package that happens to share a name with the real tool. The actual Tesseract binary is a system-level program, not something `pip` installs at all; it needs to come from `apt`, and the Python side only needs a thin wrapper (`pytesseract`) to shell out to it.

**Fix:** `pytesseract` (plus `pdf2image`, `pypdf`) in `requirements.txt`; the actual `tesseract-ocr` and `poppler-utils` binaries in a separate `packages.txt`, which Streamlit Cloud auto-detects and installs via `apt-get`.

**Lesson:** "Same name on PyPI" is not evidence of "the right package" — worth checking a package's actual purpose before trusting the name, especially for anything that wraps a system binary rather than being pure Python.

---

### 18. A stale, un-deleted file kept re-triggering a platform-side outage on every single deploy

**Symptom:** An identical `apt-get` failure (an expired upstream Debian security-mirror snapshot, entirely outside my control) recurred across five consecutive deploy attempts over more than a day, unchanged, despite "removing" the file that triggered it multiple times.

**Root cause:** The file removal never actually reached the remote repository — verified by the deploy log still showing `"Apt dependencies were installed from .../packages.txt"` on every attempt, which is only possible if the file was still present. Local edits and commits weren't consistently making it to what GitHub (and therefore the deploy pipeline) actually had.

**Fix:** Verify directly on the GitHub web UI that a file is actually gone, not just locally — and only then treat "still broken" as evidence of a genuinely new problem, rather than re-diagnosing the same already-fixed issue repeatedly.

**Lesson:** The single most time-costly pattern across this whole project, at both the code and deployment level, was mistaking "I made the edit" for "the edit is what's actually running." Every layer of this stack (local files, git, GitHub, the deploy pipeline, the running container) is a place a change can silently fail to propagate — and the fix is always the same: add a cheap, unambiguous way to verify from the *output* that a specific change is actually live (a version-string print statement in code; checking the file listing directly in GitHub's UI for deployment), rather than re-describing the intended fix and hoping.

---

## Building an eval harness (and what it taught me about testing LLM systems)

The Phase 1 eval suite (15 JSONL test cases asserting required/forbidden substrings, expected pattern routing, hop/api_call consistency, confidence bounds) caught real things manual spot-checking missed, including the dead `--forced-pattern` flag on its first run.

Phase 2 extended this with cases specifically designed to be adversarial about *routing*, not just output correctness: a query with a false premise the system needs to detect rather than hallucinate an answer to; a schema-validation case where a requested column might not exist; a cross-source case requiring both a local document and live web data to be genuinely synthesized together, not just concatenated; a verbatim-citation case with zero tolerance for paraphrase drift.

The biggest lesson from Phase 2's testing process specifically: **the same test can look completely different across two runs with zero code changes, purely from an LLM provider's own intermittent failure modes** — in this case, a specific model occasionally returning an empty completion under load, unrelated to anything in the surrounding application code. Distinguishing "this is a real, reproducible logic bug" from "this is provider-side noise that happens to recur" became its own skill: the tell was whether a failure's *symptom* changed in a way that traced to a specific code path (a bug), versus an *identical* failure recurring on objectively correct inputs across multiple otherwise-successful runs (noise, not worth chasing further).

**Current suite status: 15/15 passing**, across all three patterns, with routing, tool sequencing, and content correctness independently verified per case.

---

## How I'd summarize this in an interview

- *"Tell me about a bug you found hard to track down."* → The classifier tie-break bug (#10): `max(dict, key=...)` returning a key instead of a value silently disabled an entire fast-path subsystem, and every symptom of it looked like "the LLM fallback made a slightly odd choice" rather than "the deterministic classifier that should have run first never actually ran." Diagnosing it meant not trusting that a subsystem was reachable just because nothing crashed.
- *"Tell me about a bug that was worse than a simple wrong answer."* → The scratchpad truncation bug (#15) — a model that confidently, articulately explains why it *can't* answer something is more dangerous than one that's visibly wrong, because it reads as careful honesty instead of a retrieval bug.
- *"How do you think about testing AI/LLM systems differently from regular software?"* → The eval-harness section above — deterministic assertions work for structural things (routing, hop counts, tool sequences), but content correctness against live retrieval needs a pass-rate model and a way to distinguish real bugs from provider-side intermittency, not a single pass/fail gate.
- *"Tell me about the most time-costly mistake in this whole project."* → Not a specific bug — the meta-pattern in #18: repeatedly re-describing a fix without a cheap way to verify it had actually landed in the running system. The single highest-leverage change I made partway through Phase 2 was adding version-string print statements to every file under active iteration, specifically so the next debug log could prove or disprove "did this edit actually take effect" in one glance instead of re-diagnosing from scratch.

---

## What I'd do differently starting over

1. Build the adversarial test suite *before* declaring any routing logic "done," not after — nearly every Phase 2 bug was a routing decision that looked correct against the handful of cases it was originally written for, and only broke against cases specifically designed to probe edge conditions.
2. Add a cheap "prove this code is actually running" mechanism (version-string logging) from the start for any file under active, iterative debugging — the cost of a single print statement is nothing next to the cost of re-diagnosing an already-fixed bug multiple times because an edit silently didn't land.
3. Treat any context-truncation logic as a correctness bug waiting to happen by default — "keep the first N characters" needs active justification, not just convenience.
4. Separate "is this tool call's *input* correct" from "is this tool call's *selection* correct" as two independent things to test — several of the hardest bugs here were cases where the right tool got selected, but stale or misbuilt input from an earlier, abandoned decision silently flowed into it.
5. For deployment specifically: verify every "I fixed it" claim against the actual remote source of truth (GitHub's file listing, not a local `git status`) before spending more time on a hypothesis that the underlying problem has changed.
