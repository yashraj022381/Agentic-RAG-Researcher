import re
 
 
def extract_relevant_excerpt(
    content: str,
    query: str,
    window_chars: int = 3000,
    max_total_chars: int = 16000,
    hits_per_keyword: int = 2,
) -> str:
    """
    Return up to max_total_chars of `content`, prioritizing windows of text
    around occurrences of the query's keywords rather than just the start
    of the document.
    """
    if not content:
        return content
 
    if len(content) <= max_total_chars:
        return content
 
    query_words = sorted(
        {
            w for w in re.findall(r'\w+', query.lower())
            if len(w) > 3 or w.isdigit()
        },
        key=len,
        reverse=True,  # longer/more specific words first (better signal)
    )
 
    if not query_words:
        return content[:max_total_chars]
 
    lower_content = content.lower()
    hit_positions = []
 
    for word in query_words:
        found_for_this_word = 0
        start_search = 0
        while found_for_this_word < hits_per_keyword:
            idx = lower_content.find(word, start_search)
            if idx == -1:
                break
            # Skip if this hit is too close to one we already have, to avoid
            # grabbing overlapping windows for the same passage repeatedly.
            if not any(abs(idx - p) < window_chars for p in hit_positions):
                hit_positions.append(idx)
                found_for_this_word += 1
            start_search = idx + 1
 
    if not hit_positions:
        # No keyword hits anywhere — fall back to the start of the document,
        # which is no worse than the previous blind-truncation behavior.
        return content[:max_total_chars]
 
    hit_positions.sort()
 
    excerpts = []
    total_len = 0
    for pos in hit_positions:
        start = max(0, pos - window_chars // 3)
        end = min(len(content), pos + (window_chars * 2) // 3)
        excerpt = content[start:end]
        excerpts.append(excerpt)
        total_len += len(excerpt)
        if total_len >= max_total_chars:
            break
 
    # Always include the very start too (title/abstract often carries
    # useful framing context even if the specific answer is elsewhere).
    intro = content[:800]
    combined = intro + "\n\n[...]\n\n" + "\n\n[...]\n\n".join(excerpts)
 
    return combined[:max_total_chars]
