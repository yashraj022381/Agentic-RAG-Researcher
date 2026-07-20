import re
from pathlib import Path
from typing import Optional
 
 
def has_local_documents(docs_folder: str = "./documents") -> bool:
    folder = Path(docs_folder)
    if not folder.exists():
        return False
    return any(f.is_file() for f in folder.glob("*"))
 
 
def read_local_document_text(path: Path, max_chars: int = 2000) -> str:
    """Best-effort plain-text read of a local document. For PDFs/DOCX this
    does real extraction where the library is available; falls back to an
    empty string (never raises) so callers can treat 'no content' uniformly."""
    try:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            try:
                from pypdf import PdfReader
                reader = PdfReader(str(path))
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
                return text[:max_chars]
            except ImportError:
                return ""
        elif suffix == ".docx":
            try:
                import docx
                doc = docx.Document(str(path))
                text = "\n".join(p.text for p in doc.paragraphs)
                return text[:max_chars]
            except ImportError:
                return ""
        else:
            return path.read_text(errors="ignore")[:max_chars]
    except Exception:
        return ""
 
 
def find_relevant_document(
    query: str,
    llm,
    docs_folder: str = "./documents",
    max_files_to_check: int = 5,
    preview_chars: int = 600,
) -> Optional[Path]:
    """
    Content-based relevance check across local documents. Returns the path
    of the first document the LLM judges plausibly relevant to the query,
    or None if no documents exist or none seem relevant.
 
    This is intentionally permissive ("plausibly relevant", not "certainly
    relevant") — a false positive here just means document_reader gets tried
    and comes back with a low-confidence result, which is cheap. A false
    negative means a real local document gets silently skipped, which is the
    failure mode this function exists to prevent.
    """
    folder = Path(docs_folder)
    if not folder.exists():
        return None
 
    files = sorted([f for f in folder.glob("*") if f.is_file()])[:max_files_to_check]
    if not files:
        return None
 
    previews = []
    for f in files:
        text = read_local_document_text(f, max_chars=preview_chars)
        if text.strip():
            previews.append((f, text))
 
    if not previews:
        return None
 
    # Single-document shortcut: still worth a relevance check rather than
    # blind trust, since a query might genuinely be unrelated to the one
    # document on file (e.g. "what's the capital of France" with a resume
    # sitting in documents/).
    listing = "\n\n".join(
        f"[DOCUMENT {i + 1}: {f.name}]\n{text}"
        for i, (f, text) in enumerate(previews)
    )
 
    prompt = (
        f"QUERY: {query}\n\n"
        f"Below are short previews of locally available documents.\n\n{listing}\n\n"
        f"Does ANY of these documents plausibly contain information relevant "
        f"to answering the query — e.g. it discusses the same topic, or "
        f"mentions a person/entity named in the query? Respond in EXACTLY "
        f"this format:\n"
        f"VERDICT: RELEVANT or NOT_RELEVANT\n"
        f"DOCUMENT: <exact filename from above, or NONE>"
    )
 
    verdict_text = llm.chat(
        system=(
            "You judge document relevance for a retrieval system. Be "
            "inclusive: if a document plausibly relates to the query's "
            "topic or names an entity mentioned in the query, mark it "
            "relevant. Only say NOT_RELEVANT if there is truly no plausible "
            "connection."
        ),
        user=prompt,
    )
 
    if "VERDICT: RELEVANT" not in verdict_text.upper():
        return None
 
    match = re.search(r'DOCUMENT:\s*(.+)', verdict_text, re.IGNORECASE)
    filename = match.group(1).strip() if match else ""
 
    if not filename or filename.upper() == "NONE":
        return None
 
    for f, _ in previews:
        if f.name.lower() in filename.lower() or filename.lower() in f.name.lower():
            return f
 
    # Verdict said relevant but the filename didn't parse cleanly against
    # what we offered — fall back to the first preview rather than losing
    # the positive signal entirely.
    return previews[0][0]
 
 
def cross_check_identity(web_content: str, local_doc_content: str, llm, person_name: str = "") -> dict:
    """
    Ask the LLM whether a web-sourced result and a local (trusted) document
    describe the SAME person, checking for substantive overlap (institution,
    field of study, employer, skills) rather than just the name matching.
 
    Returns:
        {"match": bool, "reasoning": str}
    """
    if not web_content.strip() or not local_doc_content.strip():
        return {"match": True, "reasoning": "Insufficient content to compare; skipping verification."}
 
    name_note = f" The name in question is '{person_name}'." if person_name else ""
 
    prompt = (
        f"LOCAL DOCUMENT (verified, trusted, about a known person):\n{local_doc_content[:1500]}\n\n"
        f"WEB RESULT (unverified — may describe a completely different person "
        f"who simply shares the same name):\n{web_content[:1500]}\n\n"
        f"Do these two texts describe the SAME real person?{name_note} "
        f"Check for substantive overlap: institution/university, field of study, "
        f"employer, job title, and specific skills — not just whether the name matches, "
        f"since many different people can share a common name.\n\n"
        f"Respond in EXACTLY this format:\n"
        f"VERDICT: MATCH or MISMATCH\n"
        f"REASON: <one sentence explaining why>"
    )
 
    verdict_text = llm.chat(
        system=(
            "You are a strict identity verifier. Be skeptical of surface-level name "
            "matches — assume two people sharing a name are DIFFERENT people unless "
            "there is real overlapping detail (institution, employer, field, skills)."
        ),
        user=prompt,
    )
 
    is_match = "VERDICT: MATCH" in verdict_text.upper()
 
    reason_match = re.search(r'REASON:\s*(.+)', verdict_text, re.IGNORECASE | re.DOTALL)
    reason = reason_match.group(1).strip() if reason_match else verdict_text.strip()
 
    return {"match": is_match, "reasoning": reason}
 
 
def build_mismatch_warning(local_filename: str, reasoning: str) -> str:
    """Standard warning text to prepend to an answer when identity verification fails."""
    return (
        f"⚠️ **Identity verification warning**: information below may describe a "
        f"DIFFERENT person than the one in your local document '{local_filename}'. "
        f"{reasoning}\n\n---\n\n"
    )
