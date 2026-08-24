import os
from typing import Optional
from .base import BaseTool, ToolResult

KB_CHUNKS = [
    {
        "id": "kb-001",
        "text": (
            "The Attention mechanism allows each word in a sequence to attend "
            "to every other word. This was the key innovation in the Transformer "
            "model (Vaswani et al., 2017). Self-attention computes Query, Key, "
            "Value matrices to weigh relevance."
        ),
        "keywords": ["attention", "transformer", "self-attention", "llm", "ai"],
        "confidence": 0.9,
    },
    {
        "id": "kb-002",
        "text": (
            "Retrieval-Augmented Generation (RAG) combines a retriever "
            "(finds relevant documents) with a generator (LLM that answers). "
            "The retriever uses vector embeddings to find semantically similar chunks."
        ),
        "keywords": ["rag", "retrieval", "vector", "embeddings", "generation"],
        "confidence": 0.92,
    },
    {
        "id": "kb-003",
        "text": (
            "ReAct (Reason + Act) is a prompting framework where the model "
            "interleaves reasoning traces and actions. It follows: "
            "Thought → Action → Observation → Thought → … until done."
        ),
        "keywords": ["react", "reason", "act", "thought", "action", "observation"],
        "confidence": 0.88,
    },
    {
        "id": "kb-004",
        "text": (
            "Self-RAG trains a model to reflect on its own retrieval quality. "
            "It uses special tokens: [Retrieve], [IsRel], [IsSup], [IsUse] "
            "to decide when to retrieve and whether to use retrieved docs."
        ),
        "keywords": ["self-rag", "self", "retrieve", "isrel", "issup", "isuse", "grade"],
        "confidence": 0.87,
    },
    {
        "id": "kb-005",
        "text": (
            "CRAG (Corrective RAG) adds a lightweight retrieval evaluator. "
            "If retrieved documents score below a threshold, CRAG falls back "
            "to web search to supplement or replace the retrieved context."
        ),
        "keywords": ["crag", "corrective", "evaluator", "fallback", "web", "threshold"],
        "confidence": 0.89,
    },
    {
        "id": "kb-006",
        "text": (
            "Multi-hop reasoning requires the agent to answer sub-questions "
            "sequentially. Answer from hop 1 feeds into hop 2. Example: "
            "'Who founded the company that makes the M1 chip?' → "
            "Hop1: M1 chip maker = Apple → Hop2: Apple founder = Steve Jobs."
        ),
        "keywords": ["multi-hop", "hop", "sequential", "sub-question", "chain"],
        "confidence": 0.91,
    },
    {
        "id": "kb-007",
        "text": (
            "Agentic RAG adds autonomous decision-making on top of standard RAG. "
            "The agent decides WHICH tool to call, WHEN to stop, and WHETHER "
            "its answer is good enough — without human intervention in the loop."
        ),
        "keywords": ["agentic", "agent", "autonomous", "decision", "loop", "tool"],
        "confidence": 0.93,
    },
]

class KnowledgeBaseTool(BaseTool):

    def __init__(self):
        self._seed_if_empty()

    def _seed_if_empty(self):
        try:
            import chromadb
            from chroma.utils import embedding_functions

            client = chromadb.PersistentClient(path="./chroma_db")
            emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            collection = client.get_or_create_collection(
                name="knowledge_base",
                embedding_function=emb_fn,
            )
            if collection.count() == 0:
                documents = [chunk["text"] for chunk in KB_CHUNKS]
                metadatas = [{"source": chunk["id"], "confidence": chunk["confidence"]} for chunk in KB_CHUNKS]
                ids = [chunk["id"] for chunk in KB_CHUNKS]
                collection.add(documents=documents, metadatas=metadatas, ids=ids)
                print(f"✅ Seeded knowledge base with {len(documents)} built-in reference chunks.")

        except ImportError:
            pass

        except Exception as e:
            print(f"⚠️ KB auto-seed failed: {e}")
            

    @property
    def name(self) -> str:
        return "knowledge_base"

    @property
    def description(self) -> str:
        return (
            "Search internal knowledge base / vector store for domain-specific "
            "content about RAG, AI architectures, and research topics."
            "Best for company documents, research papers, internal data."
        )

    def run(self, query: str, context: Optional[str] = None) -> ToolResult:

        try:
            import chromadb
            from chromadb.utils import embedding_functions

            client = chromadb.PersistentClient(path = "./chroma_db")

            emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"  # free, local model
            )

            # Get or create your collection
            collection = client.get_or_create_collection(
                name="knowledge_base",
                embedding_function=emb_fn,
            )

            # ── Check if collection is empty ─────────────
            if collection.count() == 0:
                return ToolResult(
                    content="No documents have been added to the internal knowledge base yet.",
                    source="KB: empty",
                    confidence=0.1,
                )

            # ── Search ───────────────────────────────────
            results = collection.query(
                query_texts=[query],
                n_results=min(3, collection.count()),
            )

            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            if not documents:
                return ToolResult(
                    content=f"No relevant chunks found for '{query}'.",
                    source="KB: no match",
                    confidence=0.2,
                )

            # Convert distance to confidence (lower distance = better)
            confidence = max(0.1, 1.0 - (distances[0] / 2.0))

            content = "\n\n".join(documents[:3])
            source_id = metadatas[0].get("source", "KB chunk") if metadatas else "KB chunk"

            return ToolResult(
                content=content,
                source=f"KB: {source_id}",
                confidence=round(confidence, 2),
                metadata={
                    "num_results": len(documents),
                    "distances": distances,
                },
            )

        except ImportError:
            return ToolResult(
                content="ChromaDB not installed. Run: pip install chromadb sentence-transformers",
                source="KB: not configured",
                confidence=0.0,
            )
        except Exception as e:
            return ToolResult(
                content=f"Knowledge base error: {str(e)}",
                source="KB: error",
                confidence=0.0,
            )

    def add_documents(self, documents: list, metadatas: list = None):

        import chromadb
        from chromadb.utils import embedding_functions

        client = chromadb.PersistentClient(path="./chroma_db")
        emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        collection = client.get_or_create_collection(
            name="knowledge_base",
            embedding_function=emb_fn,
        )
        ids = [f"doc-{i}" for i in range(len(documents))]
        collection.add(
            documents=documents,
            metadatas=metadatas or [{}] * len(documents),
            ids=ids,
        )
        print(f"✅ Added {len(documents)} documents to knowledge base.")


