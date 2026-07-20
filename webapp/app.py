import sys
import re
import time
from pathlib import Path
 
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
 
import streamlit as st
 
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass
 
from agent.researcher import AgenticRAGResearcher
from config.settings import Settings
from utils.cost_tracker import read_all_research_logs
 
DOCUMENTS_DIR = PROJECT_ROOT / "documents"
DOCUMENTS_DIR.mkdir(exist_ok=True)
 
ALLOWED_UPLOAD_TYPES = ["pdf", "docx", "txt", "csv"]
MAX_UPLOAD_MB = 10
 
 
st.set_page_config(page_title="Agentic RAG Researcher", page_icon="🔎", layout="wide")
st.title("🔎 Agentic RAG Researcher")
st.caption("Multi-hop · ReAct / Self-RAG / CRAG auto-selection · live scratchpad tracing")
 
tab_ask, tab_docs, tab_analytics = st.tabs(["💬 Ask a question", "📁 Documents", "📊 Analytics"])
 
 
def _extract_document_name(sources: list) -> str:
    """Pull a clean document filename out of the sources list, if this
    answer came from a local document, for prominent display alongside
    the pattern used — instead of it only being visible inside sources."""
    for s in sources or []:
        s = str(s)
        m = re.search(r'Document:\s*([^(]+?)(?:\s*\(|$)', s)
        if m:
            return m.group(1).strip()
    return ""
 
 
# ─────────────────────────────────────────────────────────────────────────
# TAB 1: Live query with hop-by-hop streaming
# ─────────────────────────────────────────────────────────────────────────
with tab_ask:
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input(
            "Your question",
            placeholder="e.g. What causes earthquakes and how are they measured?",
        )
    with col2:
        forced_pattern = st.selectbox("Pattern", ["auto", "react", "selfrag", "crag"])
 
    run_clicked = st.button("Run research", type="primary", use_container_width=True)
 
    if run_clicked and query.strip():
        settings = Settings(
            forced_pattern=None if forced_pattern == "auto" else forced_pattern,
            max_hops=5,
            verbose=False,
        )
        researcher = AgenticRAGResearcher(settings=settings)
 
        st.divider()
        st.markdown("#### Live reasoning trace")
        hop_area = st.container()
 
        def on_hop(step):
            with hop_area:
                conf_pct = f"{step.confidence:.0%}"
                conf_color = "🟢" if step.confidence >= 0.8 else ("🟡" if step.confidence >= 0.5 else "🔴")
                st.markdown(
                    f"**Hop {step.hop_number}** · tool: `{step.tool_used}` "
                    f"· confidence: {conf_color} {conf_pct}"
                    + (" · *corrected*" if getattr(step, "corrected", False) else "")
                )
                if step.thought:
                    st.caption(step.thought)
                with st.expander(f"Observation from hop {step.hop_number}", expanded=False):
                    st.text((step.observation or "")[:1500])
                st.markdown("---")
 
        with st.spinner("Researching..."):
            start = time.time()
            try:
                result = researcher.research(query, on_hop=on_hop)
                error = None
            except Exception as e:
                result = None
                error = str(e)
            elapsed = time.time() - start
 
        if error:
            st.error(f"Research failed: {error}")
        else:
            st.divider()
            st.markdown("#### Final Answer")
            st.markdown(result.final_answer)
 
            doc_name = _extract_document_name(result.sources)
            used_document = result.pattern_used == "document_reader" and bool(doc_name)
 
            if used_document:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Tool used", "DOCUMENT_READER")
                m2.metric("Document used", doc_name)
                m3.metric("Confidence", f"{result.confidence:.0%}")
                m4.metric("Time", f"{elapsed:.1f}s")
                if forced_pattern != "auto":
                    st.caption(
                        f"📄 You selected **{forced_pattern.upper()}**, but this was "
                        f"answered directly from your uploaded document "
                        f"**{doc_name}** — more reliable than a forced pattern's "
                        f"web-based answer for this content."
                    )
                else:
                    st.caption(
                        f"📄 This answer was grounded in your uploaded document "
                        f"**{doc_name}** — not web search."
                    )
            else:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Pattern used", result.pattern_used.upper())
                m2.metric("Hops", result.hops)
                m3.metric("Confidence", f"{result.confidence:.0%}")
                m4.metric("Time", f"{elapsed:.1f}s")
                if forced_pattern != "auto":
                    st.caption(
                        f"🔧 Pattern forced to **{forced_pattern.upper()}** — no "
                        f"local document answered this, so you're seeing this "
                        f"pattern's real multi-hop behavior."
                    )
 
            if result.sources:
                with st.expander(f"Sources ({len(result.sources)})"):
                    for s in result.sources:
                        st.write(f"- {s}")
 
    elif run_clicked:
        st.warning("Enter a question first.")
 
 
# ─────────────────────────────────────────────────────────────────────────
# TAB 2: Document upload / management
# ─────────────────────────────────────────────────────────────────────────
with tab_docs:
    st.subheader("Manage local documents")
    st.caption(
        "Files uploaded here are saved into the project's documents/ folder, "
        "the same place document_reader and the CSV analyzer read from."
    )
 
    uploaded_files = st.file_uploader(
        "Upload one or more documents",
        type=ALLOWED_UPLOAD_TYPES,
        accept_multiple_files=True,
        help=f"Accepted types: {', '.join(ALLOWED_UPLOAD_TYPES)}. Max {MAX_UPLOAD_MB} MB each.",
    )
 
    if uploaded_files:
        saved_count = 0
        for uploaded_file in uploaded_files:
            size_mb = uploaded_file.size / (1024 * 1024)
            if size_mb > MAX_UPLOAD_MB:
                st.error(f"'{uploaded_file.name}' is {size_mb:.1f} MB — exceeds the {MAX_UPLOAD_MB} MB limit, skipped.")
                continue
 
            dest_path = DOCUMENTS_DIR / uploaded_file.name
            try:
                with open(dest_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                saved_count += 1
            except Exception as e:
                st.error(f"Failed to save '{uploaded_file.name}': {e}")
 
        if saved_count:
            st.success(f"Saved {saved_count} file(s) to documents/.")
 
    st.divider()
    st.markdown("#### Current documents")
 
    existing_files = sorted([f for f in DOCUMENTS_DIR.glob("*") if f.is_file()])
 
    if not existing_files:
        st.info("No documents uploaded yet.")
    else:
        for f in existing_files:
            size_kb = f.stat().st_size / 1024
            col_name, col_size, col_delete = st.columns([4, 1, 1])
            col_name.write(f"📄 {f.name}")
            col_size.caption(f"{size_kb:.0f} KB")
            if col_delete.button("🗑️ Delete", key=f"delete_{f.name}"):
                try:
                    f.unlink()
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not delete '{f.name}': {e}")
 
 
# ─────────────────────────────────────────────────────────────────────────
# TAB 3: Analytics dashboard
# ─────────────────────────────────────────────────────────────────────────
with tab_analytics:
    st.subheader("Usage & Cost Analytics")
 
    if st.button("🔄 Refresh"):
        st.rerun()
 
    records = read_all_research_logs()
 
    if not records:
        st.info("No research queries logged yet.")
    else:
        import pandas as pd
 
        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
 
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total queries", len(df))
        c2.metric("Total est. cost", f"${df['estimated_cost_usd'].sum():.4f}")
        c3.metric("Avg hops", f"{df['hops'].mean():.1f}")
        c4.metric("Avg latency", f"{df['elapsed_seconds'].mean():.1f}s")
 
        st.markdown("#### Pattern usage distribution")
        st.bar_chart(df["pattern_used"].value_counts())
 
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Confidence over time")
            st.line_chart(df.set_index("timestamp")["confidence"])
        with col_b:
            st.markdown("#### Latency over time (seconds)")
            st.line_chart(df.set_index("timestamp")["elapsed_seconds"])
 
        with st.expander("Raw log data"):
            st.dataframe(df, use_container_width=True)
