"""Streamlit frontend for the Hybrid RAG application."""

import os

import requests
import streamlit as st

DEFAULT_BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Hybrid RAG", page_icon="📚", layout="wide")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Settings")
    backend_url = st.text_input("Backend URL", value=DEFAULT_BACKEND_URL)

    if st.button("Check backend"):
        try:
            response = requests.get(f"{backend_url}/health", timeout=5)
            response.raise_for_status()
            data = response.json()
            st.success("Backend is running")

            st.subheader("Retrieval methods")
            for name, enabled in data["retrieval"].items():
                st.write(f"{'✅' if enabled else '⛔'} {name}")

            st.subheader("Connections")
            st.write(f"{'✅' if data['neo4j_configured'] else '⚠️'} Neo4j")
            st.write(f"{'✅' if data['llm_configured'] else '⚠️'} LLM API key")
        except requests.RequestException as error:
            st.error(f"Cannot reach backend: {error}")

    st.divider()
    st.caption("Document upload will appear here in a later step.")

# ---------------- Main area ----------------
st.title("📚 Hybrid RAG")
st.caption("Ask questions about your uploaded documents. Answers are grounded in sources.")

question = st.chat_input("Ask a question about your documents")

if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        st.info("Answering is not built yet. It arrives in a later phase.")