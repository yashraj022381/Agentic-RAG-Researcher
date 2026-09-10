# Agentic-RAG-Researcher

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Groq](https://img.shields.io/badge/Powered%20by-Groq-orange)](https://groq.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![ChromaDB](https://img.shields.io/badge/Vector%20Store-ChromaDB-green)](https://www.trychroma.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/yashraj022381/Agentic-RAG-Researcher?style=social)](https://github.com/yashraj022381/Agentic-RAG-Researcher)
[![GitHub forks](https://img.shields.io/github/forks/yashraj022381/Agentic-RAG-Researcher?style=social)](https://github.com/yashraj022381/Agentic-RAG-Researcher)

- A multi-pattern agentic research system that auto-selects between most suitable three reasoning strategies — ReAct, Self-RAG, and CRAG — based on the shape of the           incoming question, then runs a multi-hop tool-use loop to gather, verify, and synthesize an answer from local documents, structured data, and live web search. 

- Built with a modular architecture: pattern selection → multi-hop research loop → tool registry → synthesis, with full scratchpad tracing and live confidence scores.

- Built to answer the kind of multi-hop, adversarial questions that break naive single-pass RAG systems: false premises that should be caught rather than hallucinated past,   schema mismatches in tabular data, cross-source questions that genuinely need both a local document and live web data synthesized together, and verbatim-citation            requirements with zero tolerance for paraphrase drift.


### 🚀 Live Demo

   [![Try the App](https://img.shields.io/badge/Try%20Live%20Demo-Click%20Here-brightgreen?style=for-the-badge&logo=streamlit)](https://agentic-rag-researcher-kr8cf3gfgtsayrbobehhc8.streamlit.app/)


📖 See POSTMORTEM.md for the real debugging story behind this project — 18 concrete bugs, root causes, and what I'd do          differently.

## Screenshots

  <!-- Drop your own screenshots into assets/screenshots/ with these exact filenames and they'll render automatically below.    Recommended: crop to the relevant panel, ~1200px wide, PNG. -->
  Ask — live reasoning trace	                 Documents tab
  Show Image	                                Show Image
  
  Analytics tab	                            Pattern auto-selection in action
  Show Image	                               Show Image

## Why three patterns instead of one

   Pattern	        Core idea	                                           Shines on
   ReAct            Interleaved reasoning + tool calls, one	             Multi-hop lookups, cross-source synthesis
                    sub-question per hop.                                (local doc + web).
   Self-RAG	        Self-reflective grading of every retrieval,          Strict verification, exact computation,
                    with automatic re-query on low relevance.            verbatim citation.
   CRAG	            Explicit false-premise / schema-                     Queries that might be based on something that isn't actually true                                                       mismatch detection before trusting a retrieval.      or doesn't exist in the data.
   

## Architecture

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
  

  agentic_rag_researcher/
  ├── main.py                  # CLI entry point
  ├── webapp/app.py            # Streamlit web UI (Ask / Documents / Analytics tabs)
  ├── agent/researcher.py      # Top-level orchestration: routing, precheck, pattern dispatch
  ├── patterns/                # react.py, selfrag.py, crag.py, base.py, selector.py
  ├── loop/                    # engine.py (the multi-hop loop), scratchpad.py
  ├── tools/                   # document_reader, csv_tool, web_search, calculator, registry
  ├── utils/                   # taxonomy_classifier, query_planner, csv_analyzer, llm_client
  ├── config/settings.py       # Tuneable knobs (model, max_hops, forced_pattern)
  ├── documents/               # Drop PDFs / DOCX / TXT / CSVs here
  ├── eval/                    # 15-case regression suite + runner
  └── POSTMORTEM.md            # The real debugging journey
  

 - The loop, in brief: each hop, the active pattern decides on a tool call (or FINISH); the engine validates and, where needed, overrides that choice against the query's       actual requirements (e.g. forcing csv_analyzer before calculator if raw data hasn't been gathered yet, or reactively forcing a web_search correction when CRAG flags a       low-confidence retrieval); the tool runs; the pattern grades the result and updates the scratchpad; the loop repeats until confidence and requirement checks are             satisfied, then synthesizes a final answer from the full scratchpad.

 - Tools are shared, not pattern-specific. web_search and calculator are available to all three patterns via a shared confidence-threshold gate (_maybe_flag_web_fallback /     reactive calculation flagging) — each pattern keeps its own distinct grading logic (CRAG's premise checks, Self-RAG's relevance scoring), but none of them are limited in    which tools they can reach for.
   

## Key design decisions

  - Deterministic computation, not LLM arithmetic. The CSV tool executes real pandas operations (aggregations, top-N, derived columns, missing-value counts) and hands the       model a VERIFIED COMPUTED RESULT string to report verbatim — the model is never asked to do math it might get wrong.
    
  - Query-aware context excerpting, not blind truncation. When aggregating multi-hop context for synthesis, content is windowed around query keywords (or an explicit page       number, if the question names one) rather than just keeping the first N characters — a multi-page document's answer often isn't on page one.
    
  - A relevance floor before trusting any tool. Both the CSV router and the document router require more than a single coincidental shared word before treating a file as        relevant to a query — a generic column name like Name or Company shouldn't make an unrelated spreadsheet look like a match.
  - 
  - Graceful degradation on synthesis failure. If the final LLM call fails or returns something that isn't a real answer (a leaked tool-call attempt, empty content), the        system falls back to a clearly-labeled, cleaned dump of the raw findings rather than showing an error or garbage text.

     
## Features

  i) **Automatic pattern selection**
    - Pattern selection is automatic: a fast, deterministic keyword-and-structure classifier handles confident cases               immediately; anything genuinely ambiguous defers to an        LLM-based planner instead of guessing. You can also            force a specific pattern via CLI flag or the web UI for testing.
    - Chooses between ReAct (iterative reason-and-act), Self-RAG (self-reflective retrieval with relevance grading), and           CRAG (corrective retrieval with web fallback) based on query characteristics. Can also be forced via CLI or UI.

 ii) **Document-first grounding**  
    - Prioritises local documents (PDF, DOCX, TXT, CSV) with OCR support. Only reaches for web search when local content is        insufficient.

iii) **Multi-hop research loop**  
    - Configurable number of reasoning hops (default 5–6). Every hop records thought, tool call, observation, and confidence.

 iv) **Rich toolset**  
    - `document_reader` – extracts relevant excerpts from uploaded files  
    - `csv_analyzer` – schema-aware analysis and deterministic computation on CSV data  
    - `knowledge_base` – ChromaDB vector store  
    - `web_search` – DuckDuckGo-backed search  
    - `fact_checker` – verification helpers  
    - `calculator` – deterministic arithmetic  
    - `synthesizer` – final answer generation  

  v) **Streamlit web application**  
    - Three tabs: Ask a question (live hop-by-hop trace), Documents (upload / manage), Analytics (usage & cost).

 vi) **CLI interface**  
    - Single-query, interactive REPL, or demo mode with forced-pattern and verbose options.

vii) **Cost & usage tracking**  
    - Logs estimated token cost, latency, hops, and pattern usage.


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
 
  i)  ``` bash
   git clone https://github.com/yashraj022381/Agentic-RAG-Researcher.git
   cd Agentic-RAG-Researcher
   pip install -r requirements.txt


 ii) System dependency (optional, for OCR on scanned PDFs): install Tesseract and Poppler separately — these are system           binaries, not Python packages:
    - macOS: brew install tesseract poppler
    - Ubuntu/Debian: sudo apt-get install tesseract-ocr poppler-utils
    - Streamlit Community Cloud: already handled via packages.txt in this repo

iii) python -m venv .venv
     source .venv/bin/activate          # Windows: .venv\Scripts\activate
  
 iv) Create a .env file in the project root:
     env
     GROQ_API_KEY=gsk_your_key_here


## Usage

  1. Command-line interface

     (i) Web UI (recommended):
        streamlit run webapp/app.py

     (ii) CLI -- Single query (auto pattern):
        python main.py --query "What technology powers large language models?"

     (iii) Force a specific pattern:
        python main.py --query "Verify: Is Python older than Java?" --forced-pattern crag

     (iv) CLI -- Interactive mode:
        python main.py --interactive

     (v) Force a specific pattern:
        python main.py --query "..." --forced-pattern react
        python main.py --query "..." --forced-pattern crag
        python main.py --query "..." --forced-pattern selfrag

     (vi) Demo queries + verbose scratchpad:
        python main.py --verbose

     (vii) Run the regression suite:
        python eval/run_eval.py


     Available flags:
     
     Flag              Description                     Default
     --query           Run a single research query     –
     --interactive     Start REPL                      –
     --forced-pattern  auto / react / selfrag / crag   auto
     --max_hops        Maximum reasoning hops          5
     --verbose         Print full scratchpad trace     off


  3. Streamlit web application

     streamlit run webapp/app.py

     - Ask a question – live hop-by-hop reasoning, metrics, sources
     - Documents – upload / delete PDF, DOCX, TXT, CSV (max 60 MB each)
     - Analytics – query history, pattern distribution, estimated cost

  Uploaded files are stored in the documents/ folder and are automatically available to the agent.


## Configuration
  All runtime settings live in config/settings.py (dataclass). Important fields:
  
  Setting                       Default                Description  
  model                         openai/gpt-oss-120b    Groq model name
  max_tokens                    500                    Default generation limit
  max_hops                      6                      Maximum reasoning steps
  confidence_threshold          0.75                   Stop early when confidence is high
  selfrag_relevance_threshold   0.10                   Self-RAG relevance gate
  crag_score_threshold          0.50                   CRAG correction trigger
  forced_pattern                None                   Override automatic selection  


## Supported Document Types

  Type     Extension     Notes 
  PDF      .pdf          Text extraction + OCR fallback
  Word     .docx         Native parsing
  Text     .txt          Plain text
  CSV      .csv          Schema detection + deterministic analytics  

## Known limitations
  - Synthesis occasionally fails on an empty completion from the underlying model provider under load — the system retries automatically and falls back to a labeled raw-        findings summary rather than losing the gathered research, but this is a known upstream intermittency, not something fully eliminable client-side.
    
  - Derived multi-column CSV computation (e.g. "engineer a variance column") currently detects and computes common patterns (subtraction, per-capita division) but isn't a       general-purpose formula engine.
    
  - OCR quality depends on the source scan; heavily degraded scans may still produce noisy extracted text even with Tesseract configured correctly.


## Example Queries

   Test 1.1: Historical Context vs. Real-Time Web Comparison
   "Extract the core architectural dimensions and parameters of the model from the uploaded document, and search the web for current 2026 SOTA LLM benchmarks to construct       a performance comparison table."

   Test 1.2: Cross-Source Multi-Hop Reasoning
   "Read the financial results from the attached Q2 earnings PDF, identify our top competitor's revenue mentioned on page 4, search the web to find that competitor's actual     Q2 2026 reported revenue, and calculate our market share variance."

   Test 2.1: False Premise & Non-Existent Feature Detection
   "Based on the attached employee handbook PDF, what is the exact policy and payout rate for the company's 2026 Cryptocurrency Staking Bonus program?"

   Test 2.2: Structured Data Schema Validation
   "Filter the attached CSV dataset to show the top 5 customers with the highest churn_risk_score and list their assigned Account Manager names."

   Test 3.1: Complex Mathematical Derivation & Code Verification
   "Extract the quarterly budget and actual spend per department from the attached financial document, engineer a Spend_Variance column, and compute the exact spend per         capita for each department step-by-step."

   Test 3.2: Verbatim Citation & Zero-Hallucination Compliance
   "Summarize the liability limitations in Section 8 of the attached contract PDF. Provide exact line citations and list the three legal exceptions verbatim."


## License
  MIT — see LICENSE.
   

## Acknowledgements

  - Groq for fast inference.
  - ChromaDB & sentence-transformers for local retrieval.
  - The broader open-source RAG / agentic community for inspiration on ReAct, Self-RAG, and CRAG patterns.
