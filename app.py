"""Streamlit UI for IBM Knowledge RAG Assistant.
Run: streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

from query_engine import IBMKnowledgeRAG

# Lazy init (cache across reruns)
@st.cache_resource(show_spinner=False)
def load_engine():
    return IBMKnowledgeRAG()

st.set_page_config(page_title="IBM Knowledge RAG Assistant", page_icon="🤖", layout="wide")

st.title("IBM Knowledge RAG Assistant")

st.markdown(
    "Ask questions about IBM AI / Research topics and the synthetic HR dataset."
)

engine = load_engine()

with st.form("rag_form"):
    question = st.text_input("Enter your question", value="What are IBM's current AI research goals?")
    submitted = st.form_submit_button("Ask")

if submitted and question.strip():
    with st.spinner("Retrieving and generating answer..."):
        answer, sources, latency = engine.answer_question(question)
    st.subheader("Answer")
    st.write(answer)

    st.subheader("Sources")
    if sources:
        st.write(", ".join(sources))
    else:
        st.write("No sources found.")

    st.caption(f"Latency: {latency:.2f} seconds")

st.markdown("---")
st.markdown(
    "*This is a demo RAG system combining FAISS vector search, BM25 keyword search, and LangChain with an optional OpenAI backend.*"
)
