"""
streamlit_app.py
----------------
Research Copilot — RAG interface for 20 academic papers on criminal governance
and extortion in Latin America.

Flow:
  1. API Key gate  — user enters OpenAI key (stored in session state only)
  2. Index gate    — auto-detects ChromaDB; builds it if missing (first run)
  3. Main app      — Chat / Papers / Compare / About

Run locally:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Research Copilot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state defaults ─────────────────────────────────────────────────
for _k, _v in {
    "api_key": "",
    "api_key_validated": False,
    "index_ready": False,
    "messages": [],
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — API KEY GATE
# ══════════════════════════════════════════════════════════════════════════

def _validate_api_key(key: str) -> tuple[bool, str]:
    key = key.strip()
    if not key:
        return False, "La clave no puede estar vacía."
    if not key.startswith("sk-"):
        return False, "Una API key de OpenAI debe comenzar con 'sk-'."
    try:
        from openai import OpenAI, AuthenticationError
        OpenAI(api_key=key).models.list()
        return True, ""
    except AuthenticationError:
        return False, "API key inválida. Verifica que sea correcta."
    except Exception as exc:
        return False, f"Error al verificar: {exc}"


def _try_load_env_key():
    """Load key from local .env if present (dev convenience, never committed)."""
    if st.session_state.api_key_validated:
        return
    try:
        from dotenv import dotenv_values
        key = dotenv_values(ROOT / ".env").get("OPENAI_API_KEY", "").strip()
        if key and not key.startswith("sk-..."):
            st.session_state.api_key = key
            st.session_state.api_key_validated = True
    except Exception:
        pass


def render_api_key_gate() -> bool:
    """Returns True if a valid key is in session state."""
    if st.session_state.api_key_validated:
        return True

    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown("## 📚 Research Copilot")
        st.markdown(
            "Asistente de investigación para **20 artículos académicos** sobre "
            "crimen organizado, extorsión y gobernanza criminal en América Latina."
        )
        st.divider()
        st.markdown("### 🔑 Ingresa tu OpenAI API Key")
        st.caption(
            "La clave se guarda solo en memoria (sesión). Nunca se escribe en disco "
            "ni en el código. "
            "Obtén la tuya en [platform.openai.com/api-keys](https://platform.openai.com/api-keys)."
        )
        with st.form("api_key_form"):
            key_input = st.text_input(
                "API Key", type="password", placeholder="sk-...",
                label_visibility="collapsed",
            )
            if st.form_submit_button("Iniciar Research Copilot →", type="primary",
                                     use_container_width=True):
                with st.spinner("Verificando…"):
                    ok, msg = _validate_api_key(key_input)
                if ok:
                    st.session_state.api_key = key_input.strip()
                    st.session_state.api_key_validated = True
                    st.rerun()
                else:
                    st.error(f"❌ {msg}")
        st.divider()
        st.caption("💡 **Desarrollo local:** crea `.env` con `OPENAI_API_KEY=sk-...`")
    return False


# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — INDEX GATE
# ══════════════════════════════════════════════════════════════════════════

def _index_exists() -> bool:
    """Check if either ChromaDB collection has at least one document."""
    try:
        from src.vectorstore import (
            CHROMA_PERSIST_DIR, COLLECTION_SMALL,
            get_chroma_client, get_or_create_collection,
        )
        client = get_chroma_client(CHROMA_PERSIST_DIR)
        col = get_or_create_collection(client, COLLECTION_SMALL)
        return col.count() > 0
    except Exception:
        return False


def _build_index(api_key: str, status_placeholder):
    """Build both ChromaDB collections. Streams progress to status_placeholder."""
    from src.ingestion import load_papers
    from src.chunking import chunk_papers
    from src.vectorstore import (
        CHROMA_PERSIST_DIR, COLLECTION_SMALL, COLLECTION_LARGE,
        get_chroma_client, get_or_create_collection, index_chunks,
    )
    from openai import OpenAI

    oa = OpenAI(api_key=api_key)
    chroma = get_chroma_client(CHROMA_PERSIST_DIR)

    status_placeholder.info("📂 Leyendo los 20 PDFs…")
    papers = load_papers(verbose=False)

    for strategy, col_name in [("small", COLLECTION_SMALL), ("large", COLLECTION_LARGE)]:
        label = "256 tokens" if strategy == "small" else "1024 tokens"
        status_placeholder.info(f"✂️ Chunking ({label})…")
        chunks = chunk_papers(papers, strategy=strategy)

        col = get_or_create_collection(chroma, col_name)
        status_placeholder.info(
            f"🔢 Generando embeddings para {len(chunks)} chunks ({label})…  \n"
            "Esto toma ~2-3 minutos la primera vez."
        )
        index_chunks(chunks, col, show_progress=False, openai_client=oa)

    status_placeholder.success("✅ Índice construido. ¡Listo para usar!")


def render_index_gate() -> bool:
    """
    Returns True if the index is ready.
    On first run (no index), shows a build button.
    """
    if st.session_state.index_ready:
        return True

    if _index_exists():
        st.session_state.index_ready = True
        return True

    # Index missing — show setup screen
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown("## ⚙️ Primera configuración")
        st.markdown(
            "El índice vectorial no existe todavía. "
            "Hay que procesar los **20 PDFs** y generar sus embeddings.  \n"
            "Esto ocurre **una sola vez** (~2-3 min) y luego se guarda."
        )
        st.info(
            "📌 Se usará tu API key para llamar a `text-embedding-3-small`.  \n"
            "Costo aproximado: **< $0.05 USD** por indexación completa."
        )
        status = st.empty()
        if st.button("🚀 Construir índice ahora", type="primary", use_container_width=True):
            with st.spinner("Construyendo índice… no cierres esta pestaña."):
                try:
                    _build_index(st.session_state.api_key, status)
                    st.session_state.index_ready = True
                    st.rerun()
                except Exception as exc:
                    st.error(f"Error al construir el índice: {exc}")
    return False


# ══════════════════════════════════════════════════════════════════════════
# CACHED RESOURCES
# ══════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def get_chroma_client_cached():
    from src.vectorstore import get_chroma_client, CHROMA_PERSIST_DIR
    return get_chroma_client(CHROMA_PERSIST_DIR)


@st.cache_data(show_spinner=False)
def load_papers_metadata():
    json_path = ROOT / "papers" / "papers.json"
    if not json_path.exists():
        return []
    with open(json_path, encoding="utf-8") as f:
        return json.load(f).get("papers", [])


def get_openai_client():
    from openai import OpenAI
    return OpenAI(api_key=st.session_state.api_key)


# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.title("⚙️ Configuración")

        strategy = st.selectbox(
            "Estrategia de prompts",
            options=["delimiters", "json", "fewshot", "cot"],
            format_func=lambda x: {
                "delimiters": "1 — Delimitadores",
                "json": "2 — Salida JSON",
                "fewshot": "3 — Few-Shot",
                "cot": "4 — Chain-of-Thought",
            }[x],
            key="strategy",
        )
        chunk_strategy = st.radio(
            "Tamaño de chunks",
            options=["small", "large"],
            format_func=lambda x: "256 tokens" if x == "small" else "1024 tokens",
            horizontal=True,
            key="chunk_strategy",
        )
        top_k = st.slider("Top-k fragmentos", 1, 10, 5, key="top_k")

        st.divider()
        masked = "sk-…" + st.session_state.api_key[-4:] if len(st.session_state.api_key) > 6 else "—"
        st.caption(f"🔑 API Key activa: `{masked}`")
        if st.button("🔒 Cerrar sesión", use_container_width=True):
            for k in ["api_key", "api_key_validated", "index_ready", "messages"]:
                st.session_state[k] = "" if k == "api_key" else (False if k != "messages" else [])
            st.rerun()

        st.divider()
        st.caption("Research Copilot v0.1.0")

    return strategy, chunk_strategy, top_k


# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — CHAT
# ══════════════════════════════════════════════════════════════════════════

def render_chat_tab(strategy, chunk_strategy, top_k):
    st.header("💬 Chat con los Papers")
    st.caption(
        "Haz preguntas sobre crimen organizado, extorsión y gobernanza criminal "
        "en América Latina."
    )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📎 Fuentes", expanded=False):
                    for s in msg["sources"]:
                        st.markdown(
                            f"**{s['title']}** — {', '.join(s['authors'][:2])} ({s['year']})  \n"
                            f"Relevancia: `{s['score']:.3f}` | {s.get('venue','')}"
                        )

    if prompt := st.chat_input("¿Cuál es tu pregunta de investigación?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Buscando en los papers…"):
                answer, sources = _run_rag(prompt, strategy, chunk_strategy, top_k)
            st.markdown(answer)
            if sources:
                with st.expander("📎 Fuentes", expanded=True):
                    for s in sources:
                        st.markdown(
                            f"**{s['title']}** — {', '.join(s['authors'][:2])} ({s['year']})  \n"
                            f"Relevancia: `{s['score']:.3f}` | {s.get('venue','')}"
                        )
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})

    if st.session_state.messages:
        if st.button("🗑️ Limpiar conversación"):
            st.session_state.messages = []
            st.rerun()


def _run_rag(question, strategy, chunk_strategy, top_k):
    from src.retrieval import search
    from src.generation import generate_answer
    chroma = get_chroma_client_cached()
    oa = get_openai_client()
    try:
        chunks = search(question, top_k=top_k, strategy=chunk_strategy,
                        chroma_client=chroma, openai_client=oa)
    except RuntimeError as e:
        return f"⚠️ Error en la búsqueda: {e}", []
    result = generate_answer(question=question, chunks=chunks, strategy=strategy, client=oa)
    sources = [{"title": c.title, "authors": c.authors, "year": c.year,
                "venue": c.venue, "score": c.score} for c in chunks]
    return result["answer"], sources


# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — PAPER BROWSER
# ══════════════════════════════════════════════════════════════════════════

def render_papers_tab():
    st.header("📄 Explorador de Papers")
    papers = load_papers_metadata()
    if not papers:
        st.warning("No se encontró papers.json.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        years = sorted({p["year"] for p in papers if p.get("year")})
        sel_years = st.multiselect("Año", years)
    with c2:
        topics = sorted({t for p in papers for t in (p.get("topics") or [])})
        sel_topics = st.multiselect("Tema", topics)
    with c3:
        q = st.text_input("Título / autor", "").lower()

    filtered = papers
    if sel_years:
        filtered = [p for p in filtered if p.get("year") in sel_years]
    if sel_topics:
        filtered = [p for p in filtered if any(t in (p.get("topics") or []) for t in sel_topics)]
    if q:
        filtered = [p for p in filtered
                    if q in p.get("title","").lower()
                    or any(q in a.lower() for a in (p.get("authors") or []))]

    st.caption(f"Mostrando {len(filtered)} de {len(papers)} papers")
    for p in filtered:
        auths = "; ".join((p.get("authors") or [])[:3])
        if len(p.get("authors") or []) > 3:
            auths += " et al."
        with st.expander(f"**{p.get('title','?')}** — {auths} ({p.get('year','?')})"):
            ca, cb = st.columns([2, 1])
            with ca:
                st.markdown(f"**Autores:** {auths}")
                st.markdown(f"**Venue:** {p.get('venue') or '—'}")
                if p.get("doi"):
                    st.markdown(f"**DOI:** [{p['doi']}](https://doi.org/{p['doi']})")
                if p.get("abstract"):
                    st.caption(p["abstract"])
            with cb:
                for t in (p.get("topics") or []):
                    st.markdown(f"- {t}")


# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — COMPARE STRATEGIES
# ══════════════════════════════════════════════════════════════════════════

def render_compare_tab(chunk_strategy, top_k):
    st.header("🔬 Comparar Estrategias de Prompts")
    question = st.text_area(
        "Pregunta:",
        value="¿Cuál es la relación entre gobernanza criminal y el Estado en América Latina?",
        height=80, key="compare_q",
    )
    if st.button("▶ Ejecutar las 4 estrategias", type="primary"):
        from src.retrieval import search
        from src.generation import generate_answer, STRATEGY_LABELS
        chroma = get_chroma_client_cached()
        oa = get_openai_client()
        with st.spinner("Recuperando fragmentos…"):
            try:
                chunks = search(question, top_k=top_k, strategy=chunk_strategy,
                                chroma_client=chroma, openai_client=oa)
            except RuntimeError as e:
                st.error(str(e)); return
        st.success(f"{len(chunks)} fragmentos recuperados.")
        with st.expander("📎 Fragmentos", expanded=False):
            for c in chunks:
                st.markdown(f"**[{c.score:.3f}]** {c.title} ({c.year})")
                st.caption(c.text[:300] + "…")
        cols = st.columns(2)
        for i, strat in enumerate(["delimiters", "json", "fewshot", "cot"]):
            with cols[i % 2]:
                st.subheader(STRATEGY_LABELS[strat])
                with st.spinner(f"Generando…"):
                    res = generate_answer(question=question, chunks=chunks,
                                          strategy=strat, client=oa)
                st.markdown(res["answer"])
                st.caption(f"Tokens: {res['total_tokens']} | {res['elapsed_seconds']}s")
                st.divider()


# ══════════════════════════════════════════════════════════════════════════
# TAB 4 — ABOUT
# ══════════════════════════════════════════════════════════════════════════

def render_about_tab():
    st.header("ℹ️ Acerca del Research Copilot")
    papers = load_papers_metadata()
    st.markdown("""
## Arquitectura RAG

```
20 PDFs → PyMuPDF → Chunking (256 / 1024 tok) → text-embedding-3-small
                                                         ↓
                                                  ChromaDB (cosine)
                                                         ↓
Query → embed → Top-K Retrieval → Prompt Strategy → GPT-4o → Respuesta
```

## Las 4 Estrategias de Prompting

| # | Estrategia | Descripción |
|---|-----------|-------------|
| 1 | **Delimitadores** | Secciones `<<<CONTEXTO>>>` / `<<<PREGUNTA>>>` |
| 2 | **JSON Output** | Respuesta estructurada con campos predefinidos |
| 3 | **Few-Shot** | Dos ejemplos Q&A enseñan el estilo esperado |
| 4 | **Chain-of-Thought** | 5 pasos explícitos de razonamiento |

## Seguridad
La API key se solicita en el navegador y se guarda **solo en memoria de sesión**.
Nunca se escribe en disco ni en el código fuente.

## Papers indexados
""")
    for p in (papers or []):
        auths = "; ".join((p.get("authors") or [])[:2])
        st.markdown(f"- **{p.get('title','?')}** — {auths} ({p.get('year','?')})")

    st.markdown("""
## Uso local
```bash
git clone https://github.com/krlos20203563/research-copilot
cd research-copilot
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```
La app pedirá la API key y construirá el índice automáticamente en la primera ejecución.
""")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    # Step 1 — try .env for local dev
    _try_load_env_key()

    # Step 2 — API key gate
    if not render_api_key_gate():
        st.stop()

    # Step 3 — Index gate (auto-builds on first run)
    if not render_index_gate():
        st.stop()

    # Step 4 — Main app
    strategy, chunk_strategy, top_k = render_sidebar()
    tab_chat, tab_papers, tab_compare, tab_about = st.tabs([
        "💬 Chat", "📄 Papers", "🔬 Comparar", "ℹ️ Acerca de",
    ])
    with tab_chat:    render_chat_tab(strategy, chunk_strategy, top_k)
    with tab_papers:  render_papers_tab()
    with tab_compare: render_compare_tab(chunk_strategy, top_k)
    with tab_about:   render_about_tab()


if __name__ == "__main__":
    main()
