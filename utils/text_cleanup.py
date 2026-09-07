import re
 
_LEAKED_TAGS = ("thought", "action", "observation", "final_answer")
 
# Punctuation that indicates a sentence (or list item, or heading) actually
# finished cleanly.
_SENTENCE_ENDERS = ('.', '!', '?', '"', "'", ')', ']', ':', '```')
 
 
def _looks_truncated(text: str) -> bool:
    """Heuristic: a long answer that doesn't end in normal closing
    punctuation was very likely cut off by a token limit, not intentionally
    written that way."""
    if len(text) < 200:
        # Short answers legitimately might not end in punctuation
        # (e.g. a single number or short phrase) — don't touch these.
        return False
    tail = text.rstrip()
    return not tail.endswith(_SENTENCE_ENDERS)
 
 
def _trim_to_last_complete_sentence(text: str) -> str:
    """Cut a truncated answer back to the last point where a sentence (or
    bullet/heading) actually finished, dropping the dangling fragment."""
    # Find the last occurrence of any sentence-ending punctuation followed
    # by whitespace or end-of-string.
    best_cut = -1
    for ender in _SENTENCE_ENDERS:
        idx = text.rfind(ender)
        if idx > best_cut:
            best_cut = idx
 
    if best_cut == -1 or best_cut < len(text) * 0.5:
        # No good cut point found, or it would remove more than half the
        # answer — better to leave it as-is than mangle it further.
        return text
 
    return text[:best_cut + 1].rstrip()
 
 
def clean_answer_text(text: str) -> str:
    if not text:
        return text
 
    cleaned = text
 
    for tag in _LEAKED_TAGS:
        cleaned = re.sub(rf'</?{tag}>', '', cleaned, flags=re.IGNORECASE)
 
    cleaned = re.sub(r'```[a-zA-Z]*\n?', '', cleaned)
    cleaned = re.sub(r'```', '', cleaned)
 
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    cleaned = "\n".join(line.rstrip() for line in cleaned.split("\n"))
    cleaned = cleaned.strip()
 
    if _looks_truncated(cleaned):
        trimmed = _trim_to_last_complete_sentence(cleaned)
        if trimmed.strip():
            cleaned = trimmed.strip()
            cleaned += "\n\n*(Response may have been cut short — try asking again if this seems incomplete.)*"
 
    return cleaned
