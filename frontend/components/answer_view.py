"""Renders one /query response: answer, status, sources and retrieval details."""
from __future__ import annotations

import streamlit as st

from components.formatting import (
    link_citations,
    relevance_label,
    retrieval_method,
    retriever_badges,
    source_heading,
    status_banners,
)

_BANNER = {"success": st.success, "info": st.info, "warning": st.warning, "error": st.error}


def render_answer(response: dict, key: str) -> None:
    sources = response.get("sources") or []

    st.markdown(link_citations(response.get("answer", ""), sources))

    for banner in status_banners(response):
        _BANNER[banner.kind](banner.message)

    if sources:
        render_sources(sources, key)
    render_details(response)


def render_sources(sources: list[dict], key: str) -> None:
    st.markdown("**Sources**")
    for source in sources:
        # anchor target for the [n] links in the answer
        st.markdown(f'<div id="source-{source["ref"]}"></div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown(f"**{source_heading(source)}**")
            st.markdown(
                f"{retriever_badges(source.get('retrievers', []))} &nbsp; "
                f"Relevance: {relevance_label(source.get('rerank_score'))}"
            )
            with st.expander("Show evidence text"):
                st.text(source.get("text", ""))
                st.caption(f"Chunk: {source.get('chunk_id', '')}")


def render_details(response: dict) -> None:
    retrieval = response.get("retrieval") or {}
    if not retrieval and not response.get("input_guardrail"):
        return
    with st.expander("How this answer was produced"):
        if retrieval:
            st.markdown(f"**Retrieval method:** {retrieval_method(retrieval)}")
            cols = st.columns(4)
            cols[0].metric("Vector", retrieval.get("vector", 0))
            cols[1].metric("BM25", retrieval.get("bm25", 0))
            cols[2].metric("Graph", retrieval.get("knowledge_graph", 0))
            cols[3].metric("After re-rank", retrieval.get("reranked", 0))

        timings = response.get("timings_ms") or {}
        if timings:
            st.caption(" · ".join(f"{k}: {v / 1000:.1f}s" for k, v in timings.items()))
        if response.get("model"):
            st.caption(f"Model: {response['model']}")

        guard_out = response.get("output_guardrail") or {}
        if guard_out.get("issues"):
            st.markdown("**Verification notes**")
            for issue in guard_out["issues"]:
                st.caption(f"• {issue}")
        for note in response.get("notes") or []:
            st.caption(f"ℹ️ {note}")
