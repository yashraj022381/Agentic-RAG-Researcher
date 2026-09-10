# Agentic-RAG-Researcher

- A multi-pattern agentic research system that auto-selects between most suitable three reasoning strategies — ReAct, Self-RAG, and CRAG — based on the shape of the           incoming question, then runs a multi-hop tool-use loop to gather, verify, and synthesize an answer from local documents, structured data, and live web search. 

- Built with a modular architecture: pattern selection → multi-hop research loop → tool registry → synthesis, with full scratchpad tracing and live confidence scores.

- Built to answer the kind of multi-hop, adversarial questions that break naive single-pass RAG systems: false premises that should be caught rather than hallucinated past,   schema mismatches in tabular data, cross-source questions that genuinely need both a local document and live web data synthesized together, and verbatim-citation            requirements with zero tolerance for paraphrase drift.


## Why three patterns instead of one
   Pattern	        Core idea	                                           Shines on
   ReAct            Interleaved reasoning + tool calls, one	             Multi-hop lookups, cross-source synthesis
                    sub-question per hop.                                (local doc + web).
   Self-RAG	        Self-reflective grading of every retrieval,          Strict verification, exact computation,
                    with automatic re-query on low relevance.            verbatim citation.
   CRAG	            Explicit false-premise / schema-                     Queries that might be based on something that isn't actually true                                                       mismatch detection before trusting a retrieval.      or doesn't exist in the data.
   

- Pattern selection is automatic: a fast, deterministic keyword-and-structure classifier handles confident cases immediately; anything genuinely ambiguous defers to an LLM-   based planner instead of guessing. You can also force a specific pattern via CLI flag or the web UI for testing.
  
## Features

- **Automatic pattern selection**
  - Pattern selection is automatic: a fast, deterministic keyword-and-structure classifier handles confident cases immediately; anything genuinely ambiguous defers to an        LLM-based planner instead of guessing. You can also force a specific pattern via CLI flag or the web UI for testing.
  - Chooses between ReAct (iterative reason-and-act), Self-RAG (self-reflective retrieval with relevance grading), and CRAG (corrective retrieval with web fallback) based on
    query characteristics. Can also be forced via CLI or UI.

- **Document-first grounding**  
  - Prioritises local documents (PDF, DOCX, TXT, CSV) with OCR support. Only reaches for web search when local content is insufficient.

- **Multi-hop research loop**  
  - Configurable number of reasoning hops (default 5–6). Every hop records thought, tool call, observation, and confidence.

- **Rich toolset**  
  - `document_reader` – extracts relevant excerpts from uploaded files  
  - `csv_analyzer` – schema-aware analysis and deterministic computation on CSV data  
  - `knowledge_base` – ChromaDB vector store  
  - `web_search` – DuckDuckGo-backed search  
  - `fact_checker` – verification helpers  
  - `calculator` – deterministic arithmetic  
  - `synthesizer` – final answer generation  

- **Streamlit web application**  
  Three tabs: Ask a question (live hop-by-hop trace), Documents (upload / manage), Analytics (usage & cost).

- **CLI interface**  
  Single-query, interactive REPL, or demo mode with forced-pattern and verbose options.

- **Cost & usage tracking**  
  Logs estimated token cost, latency, hops, and pattern usage.


## Architecture Overview
Query
↓
Pattern Selector  (auto / forced: react | selfrag | crag)
↓
Research Loop (engine + scratchpad)
├── Think → choose tool + input
├── Execute tool (registry)
├── Post-process / grade / correct
└── Repeat until confidence or max hops
↓
Synthesizer → Final answer + sources + metrics
textKey directories:

| Directory     | Responsibility                                      |
|---------------|-----------------------------------------------------|
| `agent/`      | Core orchestrator (`AgenticRAGResearcher`)          |
| `patterns/`   | ReAct, Self-RAG, CRAG implementations + selector    |
| `loop/`       | Multi-hop engine and scratchpad                     |
| `tools/`      | Tool registry and individual tools                  |
| `config/`     | Settings dataclass (model, thresholds, hops, etc.)  |
| `utils/`      | LLM client, display, query planning, cost tracking… |
| `webapp/`     | Streamlit frontend                                  |
| `documents/`  | Local document store (PDF, DOCX, TXT, CSV)          |
| `eval/`       | Evaluation harness and test queries                 |


## Tech Stack

- **LLM**: Groq (default model configurable)
- **Embeddings / Vector store**: sentence-transformers + ChromaDB
- **Document processing**: pypdf, python-docx, pdf2image, pytesseract, Pillow
- **Data**: pandas
- **Web search**: ddgs (DuckDuckGo)
- **UI**: Streamlit
- **Other**: python-dotenv, rich, requests


## Prerequisites

- Python 3.10+
- A free [Groq API key](https://console.groq.com/)
- A free [Tavily_API_key](https://www.tavily.com/)
- (Optional) Tesseract OCR if you want scanned-PDF support


  ## Installation
      ```bash
     git clone https://github.com/yashraj022381/Agentic-RAG-Researcher.git
     cd Agentic-RAG-Researcher

     python -m venv .venv
     source .venv/bin/activate          # Windows: .venv\Scripts\activate

     pip install -r requirements.txt

  
     Create a .env file in the project root:
     env
     GROQ_API_KEY=gsk_your_key_here


## Usage

  1. Command-line interface
     # Single query (auto pattern)
     python main.py --query "What technology powers large language models?"

     # Force a specific pattern
     python main.py --query "Verify: Is Python older than Java?" --forced-pattern crag

     # Interactive mode
     python main.py --interactive

     # Demo queries + verbose scratchpad
     python main.py --verbose


     Available flags:
     
     Flag              Description                     Default
     --query           Run a single research query     –
     --interactive     Start REPL                      –
     --forced-pattern  auto / react / selfrag / crag   auto
     --max_hops        Maximum reasoning hops          5
     --verbose         Print full scratchpad trace     off


  2. Streamlit web application

     streamlit run webapp/app.py

     - Ask a question – live hop-by-hop reasoning, metrics, sources
     - Documents – upload / delete PDF, DOCX, TXT, CSV (max 60 MB each)
     - Analytics – query history, pattern distribution, estimated cost

  Uploaded files are stored in the documents/ folder and are automatically available to the agent.


## Configuration
  All runtime settings live in config/settings.py (dataclass). Important fields:
  
  Setting                      Default                Description  
  model                        openai/gpt-oss-120b    Groq model name
  max_tokens                   500                    Default generation limit
  max_hops                     6                      Maximum reasoning steps
  confidence_threshold         0.75                   Stop early when confidence is high
  selfrag_relevance_threshold  0.10                   Self-RAG relevance gate
  crag_score_threshold         0.50                   CRAG correction trigger
  forced_pattern               None                   Override automatic selection  


## Supported Document Types

  Type     Extension     Notes 
  PDF      .pdf          Text extraction + OCR fallback
  Word     .docx         Native parsing
  Text     .txt          Plain text
  CSV      .csv          Schema detection + deterministic analytics  


## Example Queries
