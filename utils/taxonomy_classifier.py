import re
from pathlib import Path
from utils.paths import DOCS_DIR
from utils.csv_analyzer import load_csv_schema


_DOCUMENT_SIGNAL_WORDS = [
    "pdf", "document", "the paper", "the file", "the resume", "chapter",
    "attached", "earnings report", "10-k", "10-q", "the report",
    "uploaded", "the docx", "the doc",
]

_DATA_SIGNAL_WORDS = [
    "csv", "dataset", "passenger", "pclass", "train.csv", "test.csv",
    "spreadsheet",
    "per department", "per row", "each department", "each row",
]

# Generic phrasing patterns that imply "answer this from a local document"
# without naming any specific file — generalizes across ANY uploaded doc.
_DOCUMENT_PHRASE_PATTERNS = [
    r"according to the \w+",
    r"in the (document|paper|report|file|pdf|resume)",
    r"mentioned (on|in) page",
    r"from the (attached|uploaded)",
    r"the (document|paper|report) states",
]

strict_math_words = ["mathematical formula", "do not approximate", "mathematical grounding"]
strict_words = [
    "strictly verify", "strictly", "exactly", "exact", "precisely", "precise",
    "verbatim", "step-by-step", "step by step",
]
numeric_verify_words = ["numerically verify", "not estimated", "numerical verification"]
premise_words = ["assuming", "premise"]
reformulate_words = ["reformulate", "search terms", "if it's not there", "if not there"]
compare_words = ["compare", "versus", " vs ", "check if that matches", "cross-reference"]
second_ask_words = [
    "and what", "and also", "as well as", "latest", "online",
    "current", "recent", "benchmark",
]

# Structural/schema-risk words: the question asks to filter/group/list BY a
# specific named field, which carries real risk that field simply doesn't
# exist in the actual data — the tabular-data equivalent of "might not be
# in the document, correct if so."
schema_risk_words = [
    "assigned", "grouped by", "categorized by", "labeled",
]


# "verbatim" is an unambiguous, rare word that virtually always means
# "quote this exactly, don't paraphrase" — a strong, standalone Self-RAG
# signal (strict grounding/citation verification) that deserves to clear
# MIN_CONFIDENT_SCORE on its own, rather than relying on the LLM planner
# to apply query_planner.py's own prompt guidance correctly every time.
citation_verbatim_words = ["verbatim", "word for word", "exact wording", "exact quote"]


# Sequential multi-step instruction chains ("read X, identify Y, search Z,
# and calculate W") are react's core signature, even with zero strict/
# premise/reformulate language. Detected by counting action verbs, since
# exact wording varies too much to enumerate every phrasing.
_ACTION_VERBS = [
    "read", "identify", "search", "find", "calculate", "compute",
    "extract", "compare", "summarize", "look up", "check", "locate",
    "determine", "list", "retrieve",
]

_STRONG_DOCUMENT_WORDS = [
    "pdf", "the paper", "the resume", "chapter", "10-k", "10-q", "the docx", "the doc",
]

_WEAK_DOCUMENT_WORDS = [
    "document", "the file", "attached", "earnings report", "the report", "uploaded",
]

# "exact <detail> for/of/on <named thing>" — a narrow, single-value
# extraction from one document (e.g. "exact policy and payout rate for
# X"). Matches on SENTENCE STRUCTURE, not specific nouns, so it
# generalizes to any narrow-detail-from-one-doc query. The {0,40} cap
# keeps it from over-matching across an entire long sentence.
_NARROW_EXTRACTION_PATTERN = re.compile(
    r"\bexact\s+[a-z][a-z\s\-]{0,40}?\s+(?:for|of|on|regarding|about)\b",
    re.IGNORECASE,
)

_DOCUMENT_PREMISE_PATTERN = re.compile(
    r'\b(according to|based on|per|as (?:stated|listed|described) in)\s+(?:the\s+)?'
    r'[a-z0-9\s\-]{0,40}?\b(pdf|document|paper|handbook|report|text|book|manual|contract|agreement|file|section)\b',
    re.IGNORECASE,
)

_CSV_ROW_LOOKUP_PATTERN = re.compile(
    r'\b(find|get|filter|show|retrieve|look\s?up|lookup)\b.*?'
    r'\b(id|passenger|customer|row|record)\b.*?\b\d+\b',
    re.IGNORECASE,
)

_RECENT_YEAR_PATTERN = re.compile(r'\b20(2[4-9]|3\d)\b')

_EXPLICIT_WEB_PHRASES = ["search the web", "search online", "look up online", "find online"]


_COMPUTATION_WORDS = ["calculate", "compute", "variance", "market share", "per capita", "derive", "derivation"]

_MULTI_ROW_WORDS = ["for each", "each department", "per department", "engineer a", "for all rows", "each row"]

_EXTERNAL_KNOWLEDGE_WORDS = [
    "historical", "modern", "current", "recent", "state of the art",
    "state-of-the-art", "sota", "today", "nowadays", "contemporary",
]

_GENERIC_STEM_WORDS = {
    "document", "file", "files", "report", "paper", "doc", "docx",
    "text", "sample", "large", "com", "attached", "uploaded",
}


_GENERIC_COLUMN_WORDS = {
    "name", "names", "date", "dates", "id", "type", "status", "category",
    "region", "location", "company", "age", "years", "amount", "value",
    "code", "description", "notes", "email", "phone", "address",
    "currency", "level", "number", "total",
}

# A keyword-layer match is only trusted when it clears this bar. Below it,
# classify_taxonomy() defers to the (smarter, but slower/less reliable)
# LLM planner instead of guessing — see classify_taxonomy()'s docstring.
MIN_CONFIDENT_SCORE = 3


def _count_action_verbs(q: str) -> int:
    return sum(1 for v in _ACTION_VERBS if re.search(rf'\b{re.escape(v)}\b', q))


def _detect_needs_document(q: str, needs_data: bool, available_documents: list = None) -> bool:
    """File-agnostic document-intent detection: generic words, generic
    phrasing patterns, and actual filenames on disk — never hardcoded
    titles, so this adapts to whatever the user has actually uploaded.
    `needs_data` is passed in rather than recomputed here, so it's
    calculated exactly once per call to classify_taxonomy()."""

    #needs_data = _detect_needs_data(q, available_documents)

    if any(w in q for w in _STRONG_DOCUMENT_WORDS):
        return True

    if any(w in q for w in _WEAK_DOCUMENT_WORDS) and not needs_data:
        return True

    #if any(re.search(p, q) for p in _DOCUMENT_PHRASE_PATTERNS):
    #F    return True

    if any(re.search(p, q) for p in _DOCUMENT_PHRASE_PATTERNS) and not needs_data:
        return True

    if available_documents:
        # Word-based (not substring) matching — a filename stem word must
        # appear as a WHOLE WORD in the query, not merely as a substring
        # of some other word (e.g. filename stem "churn" must not match
        # merely because "churn_risk_score" contains "churn" as a
        # substring — that produced false-positive needs_document hits).
        q_words = set(re.findall(r"\w+", q))
        for name in available_documents:
            if name.lower().endswith('.csv'):
                continue 
            stem_words = {
                w for w in re.findall(r"\w+", Path(name).stem.lower().replace("_", " ").replace("-", " "))
                if len(w) > 3 and w not in _GENERIC_STEM_WORDS
            }
            if stem_words and (stem_words & q_words):
                return True
    return False

def _detect_needs_data(q: str, available_documents: list = None) -> bool:
    print("      [DEBUG] _detect_needs_data version: v48-no-early-return")
    if any(w in q for w in _DATA_SIGNAL_WORDS):
        return True
    if available_documents:
        q_words = set(re.findall(r"\w+", q))
        for name in available_documents:
            if not name.lower().endswith(".csv"):
                continue
            schema = load_csv_schema(Path(DOCS_DIR) / name)
            if not schema:
                continue
            for col in schema.get("columns", []):
                col_words = {
                    w for w in re.findall(r"\w+", col.lower().replace("_", " "))
                    if len(w) > 3 and w not in _GENERIC_COLUMN_WORDS
                }
                if not col_words: 
                    #return True
                    continue
                overlap = col_words & q_words

                required = 2 if len(col_words) >= 2 else 1
                if len(overlap) >= required:
                    print(f"      [DEBUG] needs_data=True via column '{col}' in '{name}' "
                          f"(overlap={overlap}, required={required})")
                    return True
    print(f"      [DEBUG] needs_data=False")
    return False


def _score_pattern(q: str, needs_document: bool, needs_data: bool) -> dict:
    """Shared scoring logic used both by classify_taxonomy() (which may
    defer to the LLM planner below MIN_CONFIDENT_SCORE) and by
    best_guess_pattern() (which never defers — used as query_planner.py's
    last-resort fallback when the LLM call itself has failed)."""
    scores = {"react": 0, "crag": 0, "selfrag": 0}

    if any(w in q for w in numeric_verify_words):
        scores["selfrag"] += 3
    if any(w in q for w in strict_math_words):
        scores["selfrag"] += 3
    if any(w in q for w in strict_words):
        scores["selfrag"] += 3
    if any(w in q for w in citation_verbatim_words):
        scores["selfrag"] += 3
        

    if any(w in q for w in premise_words):
        scores["crag"] += 2
    if any(w in q for w in reformulate_words):
        scores["crag"] += 3
    if needs_data and any(w in q for w in schema_risk_words):
        scores["crag"] += 3

    # A genuine computation/derivation signal takes priority over the
    # narrow-extraction CRAG signal, which is meant for "this specific
    # fact might simply not be IN the document" cases, not "compute this
    # exact number and verify it" cases. Deliberately narrower than
    # strict_words as a whole — it must NOT include bare "exact"/
    # "exactly"/"precisely"/"verbatim", since those are also the trigger
    # words for _NARROW_EXTRACTION_PATTERN itself; including them here
    # would make this guard fire every time the pattern matches at all,
    # permanently disabling the narrow-extraction CRAG signal (this broke
    # Test 2.1 in an earlier draft of this function).
    has_computation_signal = (
        "step-by-step" in q
        or "step by step" in q
        or any(w in q for w in strict_math_words)
        or any(w in q for w in numeric_verify_words)
        or re.search(r'\b(compute|calculate|derive|derivation)\b', q)
    )
    
    has_external_reference_signal = (
        any(w in q for w in _EXTERNAL_KNOWLEDGE_WORDS)
        or bool(_RECENT_YEAR_PATTERN.search(q))
        or any(w in q for w in _EXPLICIT_WEB_PHRASES)
    )
    
    if needs_document and _NARROW_EXTRACTION_PATTERN.search(q) and not has_computation_signal:
        scores["crag"] += 3
    has_citation_signal = any(w in q for w in citation_verbatim_words) or any(w in q for w in strict_words)
    document_premise_match = needs_document and _DOCUMENT_PREMISE_PATTERN.search(q)
    if document_premise_match and not has_computation_signal:
        scores["crag"] += 3
    if needs_data and _CSV_ROW_LOOKUP_PATTERN.search(q) and not has_computation_signal:
        scores["crag"] += 3
        
    if any(w in q for w in compare_words):
        scores["react"] += 1
    if any(w in q for w in second_ask_words):
        scores["react"] += 2

    action_verb_count = _count_action_verbs(q)
    if needs_document and action_verb_count >= 3:
        scores["react"] += 3

    if (needs_document or needs_data) and has_external_reference_signal:
        scores["react"] += 3

    return scores


def _needs_computation(q: str) -> bool:
    return any(w in q for w in _COMPUTATION_WORDS)

def _needs_multi_row_computation(q: str) -> bool:
    return any(w in q for w in _MULTI_ROW_WORDS)


def classify_taxonomy(query: str, available_documents: list = None) -> dict:
    """
    Weighted, multi-signal classification. Checked BEFORE the LLM planner
    as a fast, deterministic signal — but only trusted when a real signal
    clears MIN_CONFIDENT_SCORE. Returns None (deferring to plan_query())
    when no signal fires, or when the best score is too weak to trust,
    rather than forcing a low-confidence guess.
    """
    print("      [DEBUG] taxonomy_classifier.py version: v46-compound-overlap-fix")
    q = query.lower()

    #needs_data = any(w in q for w in _DATA_SIGNAL_WORDS)
    needs_data = _detect_needs_data(q, available_documents)
    needs_document = _detect_needs_document(q, needs_data, available_documents)

    scores = _score_pattern(q, needs_document, needs_data)

    if max(scores.values()) < MIN_CONFIDENT_SCORE:
        return None

    best_score = max(scores.values())
    tied_patterns = [p for p, s in scores.items() if s == best_score]
    
    if not tied_patterns:
        # Should be structurally impossible, but never let this function crash
        # the whole request — defer to the LLM planner instead.
        return None
    
    if len(tied_patterns) > 1:
        # A genuine tie between two patterns at the confidence floor is exactly
        # the "too weak to trust" case this function is meant to defer on —
        # picking one arbitrarily (by dict order) is a silent low-confidence
        # guess, not a real classification.
        return None

    best = tied_patterns[0]
    
    needs_web = (
        any(w in q for w in second_ask_words)
        or any(w in q for w in reformulate_words)
        or any(p in q for p in _EXPLICIT_WEB_PHRASES)
        or ((needs_document or needs_data) and (
            any(w in q for w in _EXTERNAL_KNOWLEDGE_WORDS)
            or bool(_RECENT_YEAR_PATTERN.search(q))
        ))
    )
    

    return {
        "pattern": best,
        "needs_document": needs_document,
        "needs_data": needs_data,
        "needs_web": needs_web,
        "needs_computation": _needs_computation(q),
        "category": f"{best}_scored",
    }


def best_guess_pattern(query: str, available_documents: list = None) -> str:
    """Never defers — used only as query_planner.py's last-resort fallback
    when the LLM planner call itself has failed, so a rough signal-based
    guess is used instead of blindly defaulting to 'react' regardless of
    query content."""
    q = query.lower()
    #needs_data = any(w in q for w in _detect_needs_data)
    needs_data = _detect_needs_data(q, available_documents)
    needs_document = _detect_needs_document(q, needs_data, available_documents)
    scores = _score_pattern(q, needs_document, needs_data)
    if max(scores.values()) == 0:
        return "react"
    best_score = max(scores.values())
    tied = [p for p, s in scores.items() if s == best_score]
    return tied[0] if len(tied) == 1 else "react" 
    #return max(scores, key=scores.get)
