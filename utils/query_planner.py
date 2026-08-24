import json
import re
from patterns.selector import PatternSelector
from utils.query_analysis import is_complex_query
from utils.taxonomy_classifier import best_guess_pattern, second_ask_words, reformulate_words, _needs_computation

_plan_cache = {}

def plan_query(query: str, llm, available_documents: list[str]) -> dict:
    """One upfront LLM call that decides what this query actually needs,
    instead of letting each hop re-decide blindly. Returns a plan the loop
    can follow, with keyword-heuristic fallbacks if parsing fails."""
    doc_list = "\n".join(f"  - {d}" for d in available_documents) or "  (none)"
    prompt = f"""Analyze this research question and produce a plan.

QUESTION: {query}

LOCALLY AVAILABLE DOCUMENTS:
{doc_list}

PATTERN GUIDE (pick exactly one, based on what the question is actually asking you to DO):

"react" — straightforward step-by-step retrieval/extraction, or lookup-then-search
  tasks with no explicit verification/correction requirement. Use when the question
  just wants information gathered and reported.
  Example: "What does Section 3 say about X, and what's the latest news on Y?"

"crag" — the question implies retrieved information might be WRONG, INCOMPLETE, or
  needs CORRECTING/REFORMULATING if the first attempt fails. Use for:
  - Fallback execution: "if the document doesn't have X, find it another way"
  - Specific technical extraction: the question asks for a narrow, highly specific
    technical detail (exact dimensions, specifications, installation instructions,
    precise values) from ONE named local document — even without explicit "if not
    found" phrasing. This kind of detail is easy for a document to simply not
    contain, so treat it as implicitly needing correction/fallback if the first
    retrieval comes up empty or off-topic.
  - Query reformulation: the question is phrased ambiguously and may need retrying
    with different terms
  - Premise verification: the question ASSUMES something is true and asks you to
    verify that assumption using local data before answering
  Example: "Assuming the dataset shows X, calculate Y" (verify the assumption first)
  Example: "Find the value of X in the doc; if not found, retry with different terms"
  Example: "According to the [Document], what are the exact specifications/dimensions
    for [narrow technical thing]?" — a document is named, but the requested detail is
    specific enough that it may not actually be covered; corrective fallback to web
    search is appropriate if the excerpt doesn't contain it.

"selfrag" — the question demands the model GRADE its own answer's reliability or
  OR asks to COMPARE/evaluate two or more distinct things against each other
  (even without explicit 'verify' language — e.g. 'compare the founding stories of 
  X and Y' still qualifies, since a fair comparison requires checking claims 
  about both sides) before accepting a result. Use for:
  - Strict verification: "confirm this is exactly correct, don't guess"
  - Mathematical grounding: numeric/formula answers that must be checked against the
    source, not approximated
  - Numerical verification: computed statistics that must be validated as accurate,
    not estimated
  Example: "Verify the exact formula matches what's in the paper, do not approximate"
  Example: "Confirm this statistic is computed correctly from the dataset, not estimated"
  "CONCRETE EXAMPLE — read carefully: 'Calculate X from the data, then check if "
  "that matches external reports' is REACT. The word 'check' here means "
  "'look up and compare casually' — it is NOT asking you to strictly validate "
  "or grade anything. Compare this to 'strictly verify X matches exactly' which "   
  "IS selfrag. The presence of the word 'check' or 'matches' alone, without an "
  "explicit strictness/rigor word, should NOT push you toward selfrag or crag — "
  "default to react for these casual-comparison phrasings.\n"
  "CONCRETE EXAMPLE: 'Extract X from the document, and compare it with external "
  "findings' is REACT, not selfrag — even though it contains the word 'compare'. "
  "The task's actual shape is EXTRACT-THEN-SEARCH, which is react's core "
  "pattern. Reserve selfrag specifically for when the question's primary ask "
  "IS the comparison/evaluation itself (e.g. 'which of these two approaches is "
  "better'), not when comparison is just the second step after extraction.\n"


  

"IMPORTANT DISTINCTION: a casual 'then check if X matches Y' or 'cross-reference "
"with Z' is REACT — it's just an extra lookup step, not a rigor requirement. "
"Only classify as 'selfrag' when the language demands STRICTNESS explicitly — "
"words like 'strictly', 'exactly', 'precisely', 'not estimated', 'not "
"approximate', 'numerically verify'. A CSV-based premise check ('assuming X, "
"verify by counting') without that strict language is 'crag' (premise "
"verification), not 'selfrag'.\n"


Answer these about what's needed:
1. Does answering this need information from a LOCAL DOCUMENT (pdf/docx/txt/md)? (true/false)
2. Does it need a LOCAL DATA FILE (CSV) requiring computation, not just document text? (true/false)
3. Does it need CURRENT/WEB information (recent events, benchmarks, external facts not in any local file)? (true/false)
4. Does it need BOTH local content AND web search to fully answer? (true/false)
5. Which pattern fits best per the guide above: "react", "crag", or "selfrag"?
6. Estimated hops needed (1 = single lookup, 2 = local+web, 3+ = multi-part verification)
7. Does the question explicitly ask you to VERIFY, CONFIRM, or check the ACCURACY/PREMISE of something computed from data (not just retrieve/calculate a value)? (true/false)

Respond with ONLY this JSON, nothing else:
{{"needs_document": true/false, "needs_data": true/false, "needs_web": true/false, "needs_both": true/false, "pattern": "react|crag|selfrag", "estimated_hops": <int>}}
"""

    cache_key = query.strip().lower()
    if cache_key in _plan_cache:
        return _plan_cache[cache_key]

    try:
        raw = llm.chat(
            system="You are a research query planner. Respond with only valid JSON, no explanation.",
            user=prompt,
            max_tokens=200,
        )
        
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            plan = json.loads(match.group(0))
            plan.setdefault("needs_document", False)
            plan.setdefault("needs_web", False)
            plan.setdefault("needs_both", False)
            plan.setdefault("pattern", "react")
            plan.setdefault("estimated_hops", 2)
            _plan_cache[cache_key] = plan
            plan.setdefault("needs_verification", False)
            return plan
    except Exception as e:
        print(f"      ⚠️ Query planning failed ({e}) — falling back to heuristics.")

    # Fallback: reuse existing keyword-based signals if the LLM plan fails
    #from utils.query_analysis import is_complex_query

    #fallback_selector = PatternSelector.__new__(PatternSelector)
    #fallback_pattern = fallback_selector._detect(query)

    
    q_lower = query.lower()
    explicit_web_phrases = ["search the web", "search online", "look up online", "find online"]
    needs_web_fallback = (
        any(w in q_lower for w in second_ask_words)
        or any(w in q_lower for w in reformulate_words)
        or any(p in q_lower for p in explicit_web_phrases)
    )
    needs_document_fallback = bool(available_documents)
    guessed_pattern = best_guess_pattern(query, available_documents)
    
    return {
        "needs_document": needs_document_fallback,
        "needs_web": needs_web_fallback,
        "needs_both": needs_document_fallback and needs_web_fallback,
        "pattern": guessed_pattern,
        "estimated_hops": 2 if needs_document_fallback and needs_web_fallback else 1,
        "needs_computation": _needs_computation(q_lower),
        
    }
