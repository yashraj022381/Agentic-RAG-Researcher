import re

def is_complex_query(query: str) -> bool:
    q = query.lower()
    signals = 0
    if len(re.findall(r'\b(and|,)\b', q)) >= 3:
        signals += 1
    if "list all" in q or "list the" in q or re.search(r'\ball\b.*\b(values|dimensions|parameters|steps)\b', q):
        signals += 1
    if len(re.findall(r'\$[^$]+\$', query)) >= 3:
        signals += 1
    if len(re.findall(r'\b(why|how|what|explain|compare)\b', q)) >= 2:
        signals += 1

   
    external_ref_words = ["historical", "historical accounts", "real-world", "according to reports",
                         "cross-reference", "compare with", "actual accounts", "in reality"]
    data_words = ["csv", "dataset", "passengers", "pclass", "survival rate"]
    if any(w in q for w in external_ref_words) and any(w in q for w in data_words):
        signals += 2  # strong, decisive signal on its own
        
    return signals >= 2
