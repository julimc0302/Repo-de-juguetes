"""
app.py — Streamlit UI for the Research Copilot RAG system.

Run with:
    streamlit run app.py
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Research Copilot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state defaults ────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "strategy_usage" not in st.session_state:
    st.session_state.strategy_usage = {1: 0, 2: 0, 3: 0, 4: 0}
if "last_chunks" not in st.session_state:
    st.session_state.last_chunks = []
if "last_citations" not in st.session_state:
    st.session_state.last_citations = []


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_catalog() -> list[dict]:
    catalog_path = Path("catalog.json")
    if catalog_path.exists():
        with open(catalog_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def reload_catalog():
    """Bust the catalog cache so it reloads."""
    load_catalog.clear()


STRATEGY_DESCRIPTIONS = {
    1: "Delimiters — Uses XML-style <CONTEXT> / <QUESTION> tags to separate inputs.",
    2: "JSON Output — Forces a structured JSON reply (answer, sources, confidence, key_findings).",
    3: "Few-Shot — Provides 2 example Q&A pairs to guide the response style.",
    4: "Chain-of-Thought — Asks the model to reason step-by-step through each source.",
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📚 Research Copilot")
    st.caption("RAG over 21 AI/Finance papers")
    st.divider()

    chunk_size = st.selectbox(
        "Chunk size (tokens)",
        options=[256, 1024],
        index=0,
        help="Smaller chunks = more precise retrieval; larger chunks = more context per chunk.",
    )

    strategy = st.selectbox(
        "Prompting strategy",
        options=[1, 2, 3, 4],
        format_func=lambda x: f"Strategy {x} — {['Delimiters', 'JSON Output', 'Few-Shot', 'Chain-of-Thought'][x - 1]}",
        index=0,
    )
    st.caption(STRATEGY_DESCRIPTIONS[strategy])

    st.divider()

    # Status
    catalog = load_catalog()
    total_papers = len(catalog)
    total_chunks_256 = sum(p.get("chunk_count_256", 0) for p in catalog)
    total_chunks_1024 = sum(p.get("chunk_count_1024", 0) for p in catalog)

    st.metric("Papers indexed", total_papers)
    col_a, col_b = st.columns(2)
    col_a.metric("Chunks (256)", total_chunks_256)
    col_b.metric("Chunks (1024)", total_chunks_1024)

    st.divider()

    if st.button("🔄 Re-index PDFs", use_container_width=True):
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key or api_key.startswith("sk-..."):
            st.error("Set OPENAI_API_KEY in .env before indexing.")
        else:
            with st.spinner("Indexing PDFs… this may take a few minutes."):
                result = subprocess.run(
                    [sys.executable, "ingest.py"],
                    capture_output=True,
                    text=True,
                    cwd=str(Path(__file__).parent),
                )
            if result.returncode == 0:
                from rag import clear_collections_cache
                clear_collections_cache()
                reload_catalog()
                st.success("Indexing complete!")
                st.rerun()
            else:
                st.error("Indexing failed.")
                error_output = result.stderr or result.stdout or "No output captured"
                st.code(error_output[-2000:])


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_chat, tab_papers, tab_dashboard = st.tabs(["💬 Chat", "📄 Papers", "📊 Dashboard"])


# ── Tab 1: Chat ───────────────────────────────────────────────────────────────
with tab_chat:
    st.header("Ask about the research")

    # Clear conversation button
    col1, col2 = st.columns([8, 2])
    with col2:
        if st.button("🗑 New conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_chunks = []
            st.session_state.last_citations = []
            st.rerun()

    # Render existing messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("citations"):
                with st.expander("📎 Ver fuentes (APA citations)"):
                    for cite in msg["citations"]:
                        st.markdown(f"- {cite}")
                if msg.get("chunks"):
                    with st.expander("🔍 Chunks recuperados"):
                        for i, chunk in enumerate(msg["chunks"], 1):
                            meta = chunk.get("metadata", {})
                            st.markdown(
                                f"**[{i}]** {meta.get('title', 'Unknown')} "
                                f"*(chunk {meta.get('chunk_index', '?')}, "
                                f"dist={chunk.get('distance', 0):.3f})*"
                            )
                            st.text(chunk.get("text", "")[:500] + ("…" if len(chunk.get("text", "")) > 500 else ""))

    # Chat input
    if prompt := st.chat_input("Ask a question about the papers…"):
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key or api_key.startswith("sk-..."):
            st.error("Configure OPENAI_API_KEY in .env to use the chat.")
        elif total_papers == 0:
            st.warning("No papers indexed yet. Use 'Re-index PDFs' in the sidebar.")
        else:
            # Display user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Build chat history (exclude metadata fields for OpenAI)
            chat_history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
                if m["role"] in ("user", "assistant")
            ]

            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    try:
                        from rag import ask as rag_ask
                        result = rag_ask(
                            question=prompt,
                            chunk_size=chunk_size,
                            strategy=strategy,
                            chat_history=chat_history,
                        )
                        answer = result["answer"]
                        citations = result["citations"]
                        chunks = result["chunks"]

                        st.markdown(answer)

                        if citations:
                            with st.expander("📎 Ver fuentes (APA citations)"):
                                for cite in citations:
                                    st.markdown(f"- {cite}")
                        if chunks:
                            with st.expander("🔍 Chunks recuperados"):
                                for i, chunk in enumerate(chunks, 1):
                                    meta = chunk.get("metadata", {})
                                    st.markdown(
                                        f"**[{i}]** {meta.get('title', 'Unknown')} "
                                        f"*(chunk {meta.get('chunk_index', '?')}, "
                                        f"dist={chunk.get('distance', 0):.3f})*"
                                    )
                                    st.text(
                                        chunk.get("text", "")[:500]
                                        + ("…" if len(chunk.get("text", "")) > 500 else "")
                                    )

                        # Save to session state
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": answer,
                                "citations": citations,
                                "chunks": chunks,
                            }
                        )
                        st.session_state.strategy_usage[strategy] += 1

                    except Exception as exc:
                        st.error(f"Error: {exc}")
                        st.session_state.messages.append(
                            {"role": "assistant", "content": f"Error: {exc}"}
                        )


# ── Tab 2: Papers ─────────────────────────────────────────────────────────────
with tab_papers:
    st.header("Indexed Papers")

    if not catalog:
        st.info("No papers indexed yet. Run 'Re-index PDFs' from the sidebar.")
    else:
        df = pd.DataFrame(catalog)

        # Ensure required columns exist
        for col in ["title", "authors", "year", "pages", "chunk_count_256", "chunk_count_1024", "filename"]:
            if col not in df.columns:
                df[col] = "—"

        # Filters
        col_search, col_year = st.columns([3, 2])
        with col_search:
            search_text = st.text_input("Search papers", placeholder="keyword…")
        with col_year:
            valid_years = sorted([y for y in df["year"].dropna().unique() if isinstance(y, (int, float)) and y > 0])
            if valid_years:
                year_range = st.slider(
                    "Filter by year",
                    min_value=int(min(valid_years)),
                    max_value=int(max(valid_years)),
                    value=(int(min(valid_years)), int(max(valid_years))),
                )
            else:
                year_range = (0, 9999)

        # Apply filters
        filtered = df.copy()
        if search_text:
            mask = (
                filtered["title"].str.contains(search_text, case=False, na=False)
                | filtered["authors"].str.contains(search_text, case=False, na=False)
            )
            filtered = filtered[mask]
        if valid_years:
            filtered = filtered[
                filtered["year"].between(year_range[0], year_range[1])
            ]

        display_cols = ["title", "authors", "year", "pages", "chunk_count_256", "chunk_count_1024"]
        st.dataframe(
            filtered[display_cols].rename(
                columns={
                    "title": "Title",
                    "authors": "Authors",
                    "year": "Year",
                    "pages": "Pages",
                    "chunk_count_256": "Chunks (256)",
                    "chunk_count_1024": "Chunks (1024)",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.divider()
        st.subheader("Paper detail")
        selected_title = st.selectbox(
            "Select a paper to preview",
            options=["— select —"] + list(df["title"].tolist()),
        )
        if selected_title and selected_title != "— select —":
            paper_row = df[df["title"] == selected_title].iloc[0]
            st.markdown(f"**Title:** {paper_row['title']}")
            st.markdown(f"**Authors:** {paper_row['authors']}")
            st.markdown(f"**Year:** {paper_row['year']}")
            st.markdown(f"**Pages:** {paper_row['pages']}")
            st.markdown(f"**File:** `{paper_row['filename']}`")

            # Show first chunk as abstract
            try:
                from rag import load_collections
                collections = load_collections()
                col_256 = collections.get(256)
                if col_256:
                    results = col_256.get(
                        where={"filename": {"$eq": paper_row["filename"]}},
                        include=["documents", "metadatas"],
                        limit=1,
                    )
                    if results["documents"]:
                        st.subheader("First chunk (abstract excerpt)")
                        st.text(results["documents"][0])
            except Exception:
                pass


# ── Tab 3: Dashboard ──────────────────────────────────────────────────────────
with tab_dashboard:
    st.header("System Dashboard")

    if not catalog:
        st.info("No papers indexed yet.")
    else:
        df = pd.DataFrame(catalog)

        # Top metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Papers", len(df))
        m2.metric("Total Chunks (256)", int(df.get("chunk_count_256", pd.Series([0])).sum()))
        m3.metric("Total Chunks (1024)", int(df.get("chunk_count_1024", pd.Series([0])).sum()))

        st.divider()

        # Papers by year
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Papers by year")
            year_df = (
                df[df["year"] > 0]["year"]
                .value_counts()
                .sort_index()
                .reset_index()
            )
            year_df.columns = ["Year", "Count"]
            year_df["Year"] = year_df["Year"].astype(str)
            st.bar_chart(year_df.set_index("Year"))

        with col_right:
            st.subheader("Chunks per paper (256-token)")
            if "chunk_count_256" in df.columns:
                chunks_df = df[["title", "chunk_count_256"]].copy()
                chunks_df["short_title"] = chunks_df["title"].str[:40]
                chunks_df = chunks_df.sort_values("chunk_count_256", ascending=False)
                st.bar_chart(chunks_df.set_index("short_title")["chunk_count_256"])

        st.divider()

        # Strategy usage
        st.subheader("Prompting strategy usage in this session")
        strategy_names = {
            1: "Delimiters",
            2: "JSON Output",
            3: "Few-Shot",
            4: "Chain-of-Thought",
        }
        usage_data = {
            strategy_names[k]: v
            for k, v in st.session_state.strategy_usage.items()
        }
        usage_df = pd.DataFrame(
            list(usage_data.items()), columns=["Strategy", "Times Used"]
        )
        st.dataframe(usage_df, use_container_width=True, hide_index=True)
        st.bar_chart(usage_df.set_index("Strategy"))
