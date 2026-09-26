"""Streamlit frontend for the Hybrid RAG application."""

import hmac
import os

import streamlit as st

from components.answer_view import render_answer
from services.api_client import ApiError, BackendClient

DEFAULT_BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
# Public website: visitors can only ask questions (no upload, delete or backend settings)
PUBLIC_MODE = os.getenv("PUBLIC_MODE", "false").strip().lower() in ("1", "true", "yes")
# In public mode, the owner unlocks Upload/Delete with this password (empty = no admin login)
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")
MAX_ADMIN_ATTEMPTS = 5
SUPPORTED_TYPES = ["pdf", "docx", "txt", "md"]
GRAPH_STATUS_ICON = {"done": "🕸️ graph ready", "partial": "🕸️ graph partial",
                     "processing": "⏳ building graph", "failed": "⚠️ graph failed",
                     "not_built": "graph not built", "unavailable": ""}

st.set_page_config(page_title="Hybrid RAG", page_icon="📚", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role": "user"|"assistant", "content": str|dict}]


# ---------------- Sidebar ----------------
def render_backend_status(api: BackendClient) -> None:
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


def render_upload(api: BackendClient) -> None:
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


def render_documents(api: BackendClient, can_delete: bool) -> list[dict]:
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
            if can_delete and st.button("Delete", key=f"del_{document['document_id']}"):
                try:
                    api.delete(document["document_id"])
                except ApiError as error:
                    st.error(str(error))
                st.rerun()
    return documents


def render_admin_login() -> bool:
    """Small password box for the site owner. Returns True once unlocked."""
    if st.session_state.get("is_admin"):
        if st.button("Log out of admin"):
            st.session_state.is_admin = False
            st.rerun()
        return True
    if not ADMIN_PASSWORD:
        return False
    attempts = st.session_state.get("admin_attempts", 0)
    with st.expander("Admin", expanded=False):
        if attempts >= MAX_ADMIN_ATTEMPTS:
            st.caption("Too many attempts. Reload the page to try again later.")
            return False
        password = st.text_input("Admin password", type="password", key="admin_password")
        if st.button("Unlock"):
            if hmac.compare_digest(password.encode(), ADMIN_PASSWORD.encode()):
                st.session_state.is_admin = True
                st.session_state.admin_attempts = 0
                st.rerun()
            else:
                st.session_state.admin_attempts = attempts + 1
                st.error("Wrong password.")
    return False


with st.sidebar:
    if PUBLIC_MODE:
        api = BackendClient(DEFAULT_BACKEND_URL)
        is_admin = render_admin_login()
        if is_admin:
            render_upload(api)
            st.divider()
    else:
        is_admin = True
        st.header("Settings")
        api = BackendClient(st.text_input("Backend URL", value=DEFAULT_BACKEND_URL).rstrip("/"))
        render_backend_status(api)
        st.divider()
        render_upload(api)
        st.divider()

    documents = render_documents(api, can_delete=not PUBLIC_MODE or is_admin)

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
