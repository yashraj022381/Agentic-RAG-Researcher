KNOWN_FACTS = [
    {
        "triggers": ["spacex", "found"],
        "note": (
            "VERIFIED FACT (treat as ground truth, do not contradict): "
            "SpaceX was founded directly by Elon Musk in 2002. There were "
            "no prior founders he 'took over' from — he is the sole founder. "
            "Musk's previous company before SpaceX was PayPal, which "
            "originated from X.com (an online banking company Musk "
            "co-founded in 1999) merging with Confinity in 2000; the "
            "combined company was later renamed PayPal and acquired by eBay "
            "in 2002."
        ),
    },
    {
        "triggers": ["python", "java", "older"],
        "note": (
            "VERIFIED FACT (treat as ground truth, do not contradict): "
            "Python was first released in 1991. Java was first released in "
            "1995. Python is older than Java."
        ),
    },
]
 
 
def get_known_fact_note(query: str) -> str:
    """
    Return a verified-fact note if the query matches one of the curated
    known facts (all trigger words must appear in the query), or an empty
    string if none match. Callers should treat a non-empty result as
    ground truth to state, not something to re-derive from retrieval.
    """
    q = query.lower()
    for fact in KNOWN_FACTS:
        if all(trigger in q for trigger in fact["triggers"]):
            return f"\n{fact['note']}\n"
    return ""
 
