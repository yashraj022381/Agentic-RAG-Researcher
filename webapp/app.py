import sys
import re
import time
import pandas as pd
from pathlib import Path
from utils.paths import DOCS_DIR

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
from utils.cost_tracker import read_all_research_logs, clear_research_logs

DOCUMENTS_DIR = PROJECT_ROOT / DOCS_DIR
DOCUMENTS_DIR.mkdir(exist_ok=True)

ALLOWED_UPLOAD_TYPES = ["pdf", "docx", "txt", "csv"]
MAX_UPLOAD_MB = 60


st.set_page_config(page_title="Agentic RAG Researcher", page_icon="🔎", layout="wide")
st.title("🔎 Agentic RAG Researcher")
st.caption("Multi-hop · ReAct / Self-RAG / CRAG auto-selection · live scratchpad tracing")

tab_ask, tab_docs, tab_analytics = st.tabs(["💬 Ask a question", "📁 Documents", "📊 Analytics"])


def _extract_document_name(sources: list) -> str:
    """Pull a clean document filename out of the sources list, so it can be
    shown prominently next to the tool/pattern info instead of only
    appearing buried in the Sources section below."""
    IGNORE_VALUES = {"not found", "folder empty"}
    for s in sources or []:
        s = str(s)
        m = re.search(r'Document:\s*([^(]+?)(?:\s*\(|$)', s)
        if m:
            name = m.group(1).strip()
            if name.lower() in IGNORE_VALUES:
                continue
            return name
    return ""


def _sort_sources(sources: list) -> list:
    """Group sources by type (Document, Calculator, Web, other) rather than
    showing them in whatever order hops happened to run in — makes the
    Sources list scannable regardless of which tool ran first."""
    def _rank(s: str) -> int:
        s_low = str(s).lower()
        if s_low.startswith("document"):
            return 0
        if s_low.startswith("calculator"):
            return 1
        if s_low.startswith("web"):
            return 2
        return 3
    return sorted(sources or [], key=lambda s: (_rank(s), str(s)))


def _calculator_results(steps: list) -> list:
    """Pull out any calculator step's raw, verified computation — kept
    separate from the prose final answer so the user can double-check the
    exact expression and result independent of how the synthesis step
    chose to describe it in words."""
    return [
        s for s in (steps or [])
        if getattr(s, "tool_used", "") == "calculator" and getattr(s, "observation", "")
    ]


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

    if "research_running" not in st.session_state:
        st.session_state.research_running = False

    run_clicked = st.button(
        "Run research", type="primary", width="stretch",
        disabled=st.session_state.research_running,
    )

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
                    # Plain text, not markdown — avoids "$...$" being
                    # misread as LaTeX math delimiters and swallowing
                    # numbers between them.
                    st.text((step.observation or "")[:4000])
                st.markdown("---")

        try:
            with st.spinner("Researching..."):
                start = time.time()
                try:
                    result = researcher.research(query, on_hop=on_hop)
                    error = None
                except Exception as e:
                    import traceback
                    print("      [DEBUG] FULL TRACEBACK:")
                    traceback.print_exc()
                    result = None
                    error = str(e)
                elapsed = time.time() - start
        finally:
            st.session_state.research_running = False

        if error:
            st.error(f"Research failed: {error}")
        else:
            st.divider()
            st.markdown("#### Final Answer")
            # Escape literal $ so markdown never misreads paired dollar
            # signs as math-mode delimiters, which otherwise silently
            # swallows the numbers between them.
            escaped_answer = result.final_answer.replace("$", "\\$")
            st.markdown(escaped_answer)

            # ---- Verified calculation callout, shown separately from
            # prose whenever a calculator hop actually ran, so exact
            # figures are visible regardless of how the synthesis step
            # chose to phrase them. ----
            calc_steps = _calculator_results(result.steps)
            if calc_steps:
                with st.container(border=True):
                    st.markdown("**🧮 Verified calculation**")
                    for s in calc_steps:
                        st.code(s.observation, language=None)

            st.divider()

            # ---- Unified metrics row — same 5 slots regardless of
            # which path (document precheck vs. full agentic loop)
            # answered the query. ----
            doc_name = _extract_document_name(result.sources)
            used_document = bool(doc_name)

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Tool used", (result.tool_used or "unknown").upper())
            m2.metric("Pattern", str(result.pattern_used).upper())
            m3.metric("Hops", result.hops)
            m4.metric("Confidence", f"{result.confidence:.0%}")
            m5.metric("Time", f"{elapsed:.1f}s")

            if used_document:
                st.caption(f"📄 Document used: **{doc_name}**")

            # ---- Simplified, always-consistent grounding message. ----
            pattern_label = str(result.pattern_used).upper()
            if used_document:
                caption = (
                    f"📄 This answer was grounded in your uploaded document "
                    f"**{doc_name}**, via the **{pattern_label}** pattern — not web search."
                )
            else:
                caption = (
                    f"🌐 This answer relied on **{(result.tool_used or 'web_search').upper()}** "
                    f"rather than a local document, via the **{pattern_label}** pattern."
                )
            if forced_pattern != "auto":
                caption += f" (Pattern was manually forced to **{forced_pattern.upper()}**.)"
            st.caption(caption)

            # ---- Sources, grouped and sorted (Document → Calculator →
            # Web → other) instead of raw hop order. ----
            if result.sources:
                real_sources = _sort_sources(
                    [s for s in result.sources if "not found" not in str(s).lower()]
                )
                if real_sources:
                    with st.expander(f"Sources ({len(real_sources)})"):
                        for s in real_sources:
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
    st.toast("Refreshed")

    records = read_all_research_logs()

    if not records:
        st.info("No research queries logged yet. Ask a question in the first tab, then come back here.")
    else:
        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")

        # Older log rows may have pattern_used stored as a list (a
        # historical bug, fixed upstream) — normalize to string so
        # PyArrow can render the dataframe without erroring.
        df["pattern_used"] = df["pattern_used"].apply(
            lambda x: "+".join(x) if isinstance(x, list) else str(x)
        )

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
            st.dataframe(df, width="stretch")

    st.divider()
    with st.expander("⚠️ Clear analytics data"):
        st.warning(
            "This permanently deletes ALL logged query history "
            "(logs/research_log.jsonl and logs/llm_calls.jsonl). "
            "This cannot be undone."
        )
        confirm_clear = st.checkbox(
            "I understand — clear all analytics data",
            key="confirm_clear_analytics",
        )
        if st.button("🗑️ Clear all analytics data", disabled=not confirm_clear):
            success = clear_research_logs()
            if success:
                st.success("Analytics data cleared.")
            else:
                st.error("Could not fully clear analytics data — check file permissions on the logs/ folder.")
            st.rerun()
