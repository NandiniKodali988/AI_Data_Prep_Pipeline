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

ACCEPTED_EXTENSIONS = ["pdf", "docx", "pptx", "xlsx", "txt", "md", "json", "yaml", "png", "jpg", "jpeg"]


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
                set(m.get("filename", Path(m.get("source_file", "")).name) for m in raw["metadatas"])
            )
            st.caption(f"{len(filenames)} document(s)")
            for fname in filenames:
                st.markdown(f"- {fname}")
        else:
            st.caption("Nothing indexed yet.")
    except Exception as e:
        st.error(f"Could not connect to index: {e}")


tab_upload, tab_qa = st.tabs(["Upload", "Ask"])

with tab_upload:
    st.subheader("Add a file")

    uploaded = st.file_uploader(
        "Select a file",
        type=ACCEPTED_EXTENSIONS,
        label_visibility="collapsed",
    )

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
                            st.rerun()
                    except Exception as e:
                        st.error(str(e))
                        import traceback
                        st.code(traceback.format_exc())

with tab_qa:
    st.subheader("Ask")

    try:
        index_agent = get_index()
        if index_agent.collection_size() == 0:
            st.caption("Index some documents first.")
        else:
            question = st.text_input("Question", placeholder="What does the paper say about RAG?", label_visibility="collapsed")
            top_k = st.slider("Results to pull", min_value=2, max_value=10, value=5)

            if st.button("Ask", type="primary") and question:
                with st.spinner("Thinking..."):
                    chunks = index_agent.search(question, top_k=top_k)

                if not chunks:
                    st.caption("Nothing relevant found.")
                else:
                    rag = get_rag()
                    result = rag.answer(question, chunks)

                    st.markdown(result["answer"])
                    st.caption(f"Sources: {', '.join(result['sources'])}")

                    with st.expander("Show sources"):
                        for i, chunk in enumerate(chunks, 1):
                            meta = chunk["metadata"]
                            fname = meta.get("filename", Path(meta.get("source_file", "")).name)
                            section = meta.get("section_heading", "")
                            st.markdown(f"**[{i}] {fname}**" + (f" - {section}" if section else ""))
                            st.caption(f"score: {1 - chunk['distance']:.3f}")
                            st.text(chunk["text"][:500] + ("..." if len(chunk["text"]) > 500 else ""))
                            if i < len(chunks):
                                st.divider()
    except Exception as e:
        st.error(f"Error: {e}")
        import traceback
        st.code(traceback.format_exc())
