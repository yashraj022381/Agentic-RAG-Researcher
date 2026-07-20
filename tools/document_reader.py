import os
import re
from pathlib import Path
from typing import Optional
from .base import BaseTool, ToolResult
from pypdf import PdfReader

MOCK_DOCS = {
    "attention_paper": {
        "title": "Attention Is All You Need (Vaswani et al., 2017)",
        "content": (
            "Abstract: We propose a new simple network architecture, the Transformer, "
            "based solely on attention mechanisms. The Transformer is the first "
            "transduction model relying entirely on self-attention to compute "
            "representations of its input and output without using RNNs or CNNs. "
            "The model achieves 28.4 BLEU on WMT 2014 English-to-German translation."
        ),
    },
    "rag_paper": {
        "title": "Retrieval-Augmented Generation (Lewis et al., 2020)",
        "content": (
            "RAG models combine pre-trained parametric memory (the LLM) "
            "with non-parametric memory (a dense vector index of Wikipedia). "
            "For generation, the model conditions on both the input and the "
            "retrieved documents. RAG outperforms parametric-only seq2seq models."
        ),
    },
}

SUPPORTED = {".pdf", ".docx", ".doc", ".txt", ".md", ".csv"}

MAX_CONTENT_CHARS = 60000
 
 
class DocumentReaderTool(BaseTool):
 
    def __init__(self, docs_folder: str = "./documents"):
        """
        Reads files from the given folder.
        Default folder: D:\\agentic_rag_researcher\\documents\\
        """
        self.docs_folder = Path(docs_folder)
        self.docs_folder.mkdir(parents=True, exist_ok=True)
 
    @property
    def name(self) -> str:
        return "document_reader"
 
    @property
    def description(self) -> str:
        return (
            "Read and extract content from uploaded documents (PDF, DOCX, TXT). "
            "Best when you need detail from a specific known file or report."
        )
 
    # ── Main run() method ─────────────────────────────────────────────────────
    def run(
        self,
        query: str,
        context: Optional[str] = None,
        file_path: Optional[str] = None
    ) -> ToolResult:
        
        q = query.lower()
 
        # ── List all available files ─────────────────────────────
        available_files = [
            f for f in self.docs_folder.rglob("*")
            if f.is_file() and f.suffix.lower() in SUPPORTED
        ]
 
        if not available_files:
            return ToolResult(
                content=(
                    f"No documents found in '{self.docs_folder.resolve()}'. "
                    f"Please add PDF, DOCX, or TXT files to that folder."
                ),
                source="Document: folder empty",
                confidence=0.0,
            )

        matched = None

        if file_path:
            candidate = Path(file_path)
            if candidate.is_file() and candidate in available_files:
                matched = candidate
 
        # ── Try to match by filename or content keywords ──────────
        if matched is None:
            matched = self._find_best_match(q, available_files)
 
        if not matched:
            names = [f.name for f in available_files]
            return ToolResult(
                content=(
                    f"No document matching '{query}' found.\n"
                    f"Available files: {', '.join(names)}"
                ),
                source="Document: not found",
                confidence=0.1,
            )
 
        # ── Read the matched file ─────────────────────────────────
        try:
            content = self._read_file(matched)
            if not content.strip():
                return ToolResult(
                    content=f"File '{matched.name}' appears to be empty.",
                    source=f"Document: {matched.name}",
                    confidence=0.2,
                )
 
            # Limit to first 3000 chars for prompt safety
            truncated = content[:MAX_CONTENT_CHARS]
            if len(content) > MAX_CONTENT_CHARS:
                truncated += f"\n\n[... {len(content) - MAX_CONTENT_CHARS} more characters truncated ...]"
 
            return ToolResult(
                content=truncated,
                source=f"Document: {matched.name}",
                confidence=0.93,
                metadata={
                    "file":      str(matched.resolve()),
                    "size_chars": len(content),
                    "extension": matched.suffix,
                },
            )
 
        except Exception as e:
            return ToolResult(
                content=f"Error reading '{matched.name}': {str(e)}",
                source=f"Document: {matched.name}",
                confidence=0.3,
            )
 
    # ── File matching ─────────────────────────────────────────────────────────
    def _find_best_match(self, query: str, files: list) -> Optional[Path]:
        """
        Find the best matching file for the query.
        Checks: exact name, partial name, keywords.
        """
        q = query.lower()
        query_words = set(q.split())

        string_triggers = {
            "resume", "cv", "summarise", "summarize",
            "summary", "pdf", "doc", "docx", "report",
            "uploaded", "my_file", "the file"
        }

        generic_triggers = {
            "document", "doc", "pdf", "file", "report", "paper", "the", "my", "uploaded"
        }
        
        if query_words & generic_triggers & string_triggers and len(files):
            for f in files:
                name = f.stem.lower().replace("_","").replace("-","")
                if any(word in name for word in query_words):
                    return f

            if len(files) == 1:
                return files[0]
            
            return files[0]
 
        best_file, best_score  = None, 0
 
        for f in files:
            name_words = set(
                f.stem.lower().replace("_", " ").replace("-", " ").split()
            )
            
            score = len(query_words & name_words)
 
            # Exact substring match scores higher
            if f.stem.lower() in q:
                score += 10
 
            # Suffix match (e.g. query mentions "pdf")
            if f.suffix.lstrip(".") in q:
                score += 3
 
            if score > best_score:
                best_score = score
                best_file  = f
 
        # If nothing scored, try any file if query is generic
        if best_score == 0:
            generic = {"file", "document", "doc", "report", "paper", "read"}
            if query_words & generic:
                return files[0]  # return first file available
 
        return best_file if best_score > 0 else None
    
        #if not files:
        #    return None
        #return files[0]
 
    # ── File readers ──────────────────────────────────────────────────────────
    def _read_file(self, path: Path) -> str:
        ext = path.suffix.lower()
 
        if ext in (".txt", ".md", ".csv"):
            return self._read_text(path, encoding="utf-8")
 
        elif ext == ".pdf":
            return self._read_pdf(path)
 
        elif ext in (".docx", ".doc"):
            return self._read_docx(path)
 
        else:
            return self._read_text(path)
 
    def _read_text(self, path: Path) -> str:
        """Read plain text / markdown / CSV."""
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return path.read_text(encoding="latin-1", errors="ignore")
 
    def _read_pdf(self, path: Path) -> str:
        """Read PDF using PyPDF2."""
        try:
            text_parts = []
            reader = PdfReader(str(path))
            #with open(path, "rb") as f:
             #   reader = PyPDF2.PdfReader(f)
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(
                        f"[Page {i + 1}]\n{page_text}"
                    )
            return "\n\n".join(text_parts) if text_parts else ""
 
        except ImportError:
            return (
                "pypdf not installed. Run:\n"
                "  pip install pypdf\n"
                "Then restart and try again."
            )
        except Exception as e:
            return f"PDF read error: {str(e)}"
 
    def _read_docx(self, path: Path) -> str:
        """Read DOCX using python-docx."""
        try:
            import docx
            doc = docx.Document(str(path))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n\n".join(paragraphs)
 
        except ImportError:
            return (
                "python-docx not installed. Run:\n"
                "  pip install python-docx\n"
                "Then restart and try again."
            )
        except Exception as e:
            return f"DOCX read error: {str(e)}"
 
    # ── Helper: list all available documents ──────────────────────────────────
    def list_documents(self) -> list:
        """Return a list of all documents in the folder."""
        return [
            {
                "name":      f.name,
                "type":      f.suffix.upper().lstrip("."),
                "size_kb":   round(f.stat().st_size / 1024, 1),
                "path":      str(f.resolve()),
            }
            for f in self.docs_folder.rglob("*")
            if f.is_file() and f.suffix.lower() in SUPPORTED
        ]





