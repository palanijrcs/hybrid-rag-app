"""Streamlit frontend for the Hybrid RAG application."""

import os

import requests
import streamlit as st

DEFAULT_BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
SUPPORTED_TYPES = ["pdf", "docx", "txt", "md"]

st.set_page_config(page_title="Hybrid RAG", page_icon="📚", layout="wide")


def error_detail(response: requests.Response) -> str:
    try:
        return response.json().get("detail", response.text)
    except ValueError:
        return response.text


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Settings")
    backend_url = st.text_input("Backend URL", value=DEFAULT_BACKEND_URL).rstrip("/")

    # ----- Status -----
    with st.expander("Backend status", expanded=False):
        if st.button("Check backend"):
            try:
                data = requests.get(f"{backend_url}/health", timeout=5).json()
                st.success("Backend is running")
                for name, enabled in data["retrieval"].items():
                    st.write(f"{'✅' if enabled else '⛔'} {name}")
                st.write(f"{'✅' if data['neo4j_configured'] else '⚠️'} Neo4j")
                st.write(f"{'✅' if data['llm_configured'] else '⚠️'} LLM API key")
            except requests.RequestException as error:
                st.error(f"Cannot reach backend: {error}")

    st.divider()

    # ----- Upload -----
    st.subheader("Upload a document")
    uploaded = st.file_uploader(
        "Choose a file", type=SUPPORTED_TYPES, accept_multiple_files=False
    )

    if uploaded is not None and st.button("Ingest document", type="primary"):
        with st.spinner(f"Ingesting {uploaded.name}..."):
            try:
                response = requests.post(
                    f"{backend_url}/documents/upload",
                    files={"file": (uploaded.name, uploaded.getvalue())},
                    timeout=120,
                )
            except requests.RequestException as error:
                st.error(f"Upload failed: {error}")
            else:
                if response.status_code == 201:
                    st.success(response.json()["message"])
                elif response.status_code == 409:
                    st.warning(error_detail(response))
                else:
                    st.error(error_detail(response))

    st.divider()

    # ----- Document list -----
    st.subheader("Documents")
    try:
        documents = requests.get(f"{backend_url}/documents", timeout=10).json()
    except requests.RequestException:
        documents = []
        st.caption("Backend unavailable.")

    if not documents:
        st.caption("No documents ingested yet.")
    for document in documents:
        with st.container(border=True):
            st.write(f"**{document['filename']}**")
            st.caption(
                f"{document['page_count']} page(s) · "
                f"{document['chunk_count']} chunks · {document['file_type']}"
            )
            if st.button("Delete", key=f"del_{document['document_id']}"):
                requests.delete(
                    f"{backend_url}/documents/{document['document_id']}", timeout=10
                )
                st.rerun()

# ---------------- Main area ----------------
st.title("📚 Hybrid RAG")
st.caption("Ask questions about your uploaded documents. Answers are grounded in sources.")

if documents:
    with st.expander("Inspect chunks"):
        options = {d["filename"]: d["document_id"] for d in documents}
        chosen = st.selectbox("Document", list(options))
        detail = requests.get(f"{backend_url}/documents/{options[chosen]}", timeout=10).json()
        for chunk in detail["chunks"]:
            page = chunk["page_number"]
            st.markdown(
                f"**{chunk['chunk_id']}**"
                + (f" · page {page}" if page is not None else "")
            )
            st.text(chunk["text"][:600])
            st.divider()

question = st.chat_input("Ask a question about your documents")

if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        st.info("Answering is not built yet. It arrives in a later phase.")