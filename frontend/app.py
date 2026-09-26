"""Streamlit frontend for the Hybrid RAG application."""

import os

import streamlit as st

from components.answer_view import render_answer
from services.api_client import ApiError, BackendClient

DEFAULT_BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
SUPPORTED_TYPES = ["pdf", "docx", "txt", "md"]
GRAPH_STATUS_ICON = {"done": "🕸️ graph ready", "partial": "🕸️ graph partial",
                     "processing": "⏳ building graph", "failed": "⚠️ graph failed",
                     "not_built": "graph not built", "unavailable": ""}

st.set_page_config(page_title="Hybrid RAG", page_icon="📚", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role": "user"|"assistant", "content": str|dict}]


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Settings")
    backend_url = st.text_input("Backend URL", value=DEFAULT_BACKEND_URL).rstrip("/")
    api = BackendClient(backend_url)

    # ----- Status -----
    with st.expander("Backend status", expanded=False):
        if st.button("Check backend"):
            try:
                data = api.health()
                st.success("Backend is running")
                for name, enabled in data["retrieval"].items():
                    st.write(f"{'✅' if enabled else '⛔'} {name}")
                st.write(f"{'✅' if data.get('neo4j_connected') else '⚠️'} Neo4j")
                st.write(f"{'✅' if data['llm_configured'] else '⚠️'} LLM API key")
            except ApiError as error:
                st.error(str(error))

    st.divider()

    # ----- Upload -----
    st.subheader("Upload a document")
    uploaded = st.file_uploader(
        "Choose a file", type=SUPPORTED_TYPES, accept_multiple_files=False
    )

    if uploaded is not None and st.button("Ingest document", type="primary"):
        with st.spinner(f"Ingesting {uploaded.name}..."):
            try:
                result = api.upload(uploaded.name, uploaded.getvalue())
                st.success(result["message"])
            except ApiError as error:
                (st.warning if error.status_code == 409 else st.error)(str(error))

    st.divider()

    # ----- Document list -----
    st.subheader("Documents")
    try:
        documents = api.list_documents()
    except ApiError:
        documents = []
        st.caption("Backend unavailable.")

    if not documents:
        st.caption("No documents ingested yet.")
    for document in documents:
        with st.container(border=True):
            st.write(f"**{document['filename']}**")
            try:
                graph = api.graph_status(document["document_id"]).get("kg_status", "")
            except ApiError:
                graph = ""
            graph_text = GRAPH_STATUS_ICON.get(graph, graph)
            st.caption(
                f"{document['page_count']} page(s) · {document['chunk_count']} chunks · "
                f"{document['file_type']}" + (f" · {graph_text}" if graph_text else "")
            )
            if st.button("Delete", key=f"del_{document['document_id']}"):
                try:
                    api.delete(document["document_id"])
                except ApiError as error:
                    st.error(str(error))
                st.rerun()

    if st.session_state.messages and st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

# ---------------- Main area ----------------
st.title("📚 Hybrid RAG")
st.caption("Ask questions about your uploaded documents. "
           "Every answer is grounded in sources you can inspect.")

if documents:
    with st.expander("Inspect chunks"):
        options = {d["filename"]: d["document_id"] for d in documents}
        chosen = st.selectbox("Document", list(options))
        try:
            detail = api.get_document(options[chosen])
            for chunk in detail.get("chunks", []):
                page = chunk.get("page_number")
                st.markdown(f"**{chunk['chunk_id']}**" + (f" · page {page}" if page else ""))
                st.text(chunk["text"][:600])
                st.divider()
        except ApiError as error:
            st.error(str(error))

# ----- conversation history -----
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.write(message["content"])
        else:
            render_answer(message["content"], key=f"m{i}")

# ----- new question -----
question = st.chat_input("Ask a question about your documents")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and verifying the answer..."):
            try:
                response = api.query(question)
            except ApiError as error:
                response = {"answer": f"⚠️ {error}", "sources": [], "grounded": False}
        render_answer(response, key=f"m{len(st.session_state.messages)}")
    st.session_state.messages.append({"role": "assistant", "content": response})
