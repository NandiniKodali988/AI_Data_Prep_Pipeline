import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="DocPipe",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

ACCEPTED_EXTENSIONS = [
    "pdf",
    "docx",
    "pptx",
    "xlsx",
    "txt",
    "md",
    "json",
    "yaml",
    "png",
    "jpg",
    "jpeg",
]


@st.cache_resource(show_spinner="Loading pipeline...")
def get_pipeline():
    from src.pipeline import Pipeline

    return Pipeline(output_dir=Path("./output"), chroma_db_path="./chroma_db")


@st.cache_resource(show_spinner=False)
def get_rag():
    from src.agents.rag_agent import RAGAgent

    return RAGAgent()


@st.cache_resource(show_spinner=False)
def get_index():
    from src.agents.indexing_agent import IndexingAgent

    return IndexingAgent(chroma_db_path="./chroma_db")


# conversation history lives in session state so it survives reruns
if "messages" not in st.session_state:
    st.session_state.messages = []

# last upload summary persists across the rerun that updates the sidebar chunk count
if "last_summary" not in st.session_state:
    st.session_state.last_summary = None


with st.sidebar:
    st.header("DocPipe")
    st.divider()

    try:
        index_agent = get_index()
        total_chunks = index_agent.collection_size()
        st.metric("Chunks indexed", total_chunks)

        if total_chunks > 0:
            raw = index_agent.collection.get(include=["metadatas"])
            filenames = sorted(
                set(
                    m.get("filename", Path(m.get("source_file", "")).name) for m in raw["metadatas"]
                )
            )
            st.caption(f"{len(filenames)} document(s)")
            for fname in filenames:
                st.markdown(f"- {fname}")
        else:
            st.caption("Nothing indexed yet.")
    except Exception as e:
        st.error(f"Could not connect to index: {e}")

    st.divider()
    top_k = st.slider("Results to pull", min_value=2, max_value=10, value=5)

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()


tab_upload, tab_chat = st.tabs(["Upload", "Chat"])

with tab_upload:
    st.subheader("Add a file")

    uploaded = st.file_uploader(
        "Select a file",
        type=ACCEPTED_EXTENSIONS,
        label_visibility="collapsed",
    )

    if st.session_state.last_summary:
        s = st.session_state.last_summary
        st.success(f"{s['filename']} indexed")
        st.markdown("**Summary**")
        st.markdown(s["text"])
        st.divider()

    if uploaded:
        st.caption(f"{uploaded.name} — {len(uploaded.getvalue()) / 1024:.1f} KB")

        if st.button("Index", type="primary"):
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir) / uploaded.name
                tmp_path.write_bytes(uploaded.getvalue())

                with st.spinner(f"Processing {uploaded.name}..."):
                    try:
                        pipeline = get_pipeline()
                        result = pipeline.process_single(tmp_path)
                        if result is None:
                            st.warning("Couldn't process this file — format may not be supported.")
                        else:
                            st.success(f"{uploaded.name} indexed ({result['chunks']} chunks)")

                            # summarize and stash in session state before rerun so
                            # it survives the page refresh that updates the sidebar
                            out_md = Path("./output") / (Path(uploaded.name).stem + ".md")
                            if out_md.exists():
                                with st.spinner("Summarizing..."):
                                    try:
                                        rag = get_rag()
                                        st.session_state.last_summary = {
                                            "filename": uploaded.name,
                                            "text": rag.summarize(
                                                out_md.read_text(encoding="utf-8"), uploaded.name
                                            ),
                                        }
                                    except Exception:
                                        pass  # summary is best-effort; don't block on API errors

                            st.rerun()
                    except Exception as e:
                        st.error(str(e))
                        import traceback

                        st.code(traceback.format_exc())

with tab_chat:
    try:
        index_agent = get_index()

        if index_agent.collection_size() == 0:
            st.caption("Index some documents first.")
        else:
            # replay the conversation from session state
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if msg["role"] == "assistant" and msg.get("sources"):
                        st.caption(f"Sources: {', '.join(msg['sources'])}")
                        if msg.get("chunks"):
                            with st.expander("Show sources"):
                                for i, chunk in enumerate(msg["chunks"], 1):
                                    meta = chunk["metadata"]
                                    fname = meta.get(
                                        "filename", Path(meta.get("source_file", "")).name
                                    )
                                    section = meta.get("section_heading", "")
                                    st.markdown(
                                        f"**[{i}] {fname}**" + (f" - {section}" if section else "")
                                    )
                                    st.caption(
                                        f"score: {chunk.get('score', 1 - chunk.get('distance', 0)):.3f}"
                                    )
                                    st.text(
                                        chunk["text"][:500]
                                        + ("..." if len(chunk["text"]) > 500 else "")
                                    )
                                    if i < len(msg["chunks"]):
                                        st.divider()

            # capture history before the new message is appended so we don't
            # include the current question in the context we pass to Claude
            if prompt := st.chat_input("Ask a question about your documents"):
                history = [
                    {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
                ]
                st.session_state.messages.append({"role": "user", "content": prompt})

                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    rag = get_rag()
                    with st.spinner("Rewriting query..."):
                        search_query = rag.rewrite_query(prompt, history)

                    with st.spinner("Searching..."):
                        chunks = index_agent.search(search_query, top_k=top_k)

                    if not chunks:
                        response_text = (
                            "I couldn't find anything relevant in the indexed documents."
                        )
                        st.markdown(response_text)
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": response_text,
                                "sources": [],
                                "chunks": [],
                            }
                        )
                    else:
                        result = rag.answer(prompt, chunks, history=history)
                        st.markdown(result["answer"])
                        st.caption(f"Sources: {', '.join(result['sources'])}")

                        with st.expander("Show sources"):
                            for i, chunk in enumerate(chunks, 1):
                                meta = chunk["metadata"]
                                fname = meta.get("filename", Path(meta.get("source_file", "")).name)
                                section = meta.get("section_heading", "")
                                st.markdown(
                                    f"**[{i}] {fname}**" + (f" - {section}" if section else "")
                                )
                                st.caption(f"score: {chunk.get('score', 1 - chunk.get('distance', 0)):.3f}")
                                st.text(
                                    chunk["text"][:500]
                                    + ("..." if len(chunk["text"]) > 500 else "")
                                )
                                if i < len(chunks):
                                    st.divider()

                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": result["answer"],
                                "sources": result["sources"],
                                "chunks": chunks,
                            }
                        )

    except Exception as e:
        st.error(f"Error: {e}")
        import traceback

        st.code(traceback.format_exc())
