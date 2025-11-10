"""
Hybrid query engine for IBM Knowledge RAG Assistant.

Responsibilities:
- Load persisted FAISS vector store + BM25 corpus
- Perform hybrid retrieval (vector similarity + BM25 keyword)
- Fuse and re-rank results by a simple scoring heuristic
- Generate an answer using LangChain LLM (OpenAI if key present, else fallback pseudo model)
- Return answer + cited sources

Usage:
    from query_engine import IBMKnowledgeRAG
    engine = IBMKnowledgeRAG()
    answer, sources = engine.answer_question("What are IBM's AI research goals?")

Design Notes:
- Simplified scoring: normalized vector similarity + BM25 score (log form)
- Citations: top unique sources from final ranked chunks
- Fallback model: deterministic template-based summarizer if no OpenAI API key
"""
from __future__ import annotations

import os
import json
import math
import time
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from dotenv import load_dotenv

from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# LLM (OpenAI or Gemini) or fallback
try:
    from langchain_openai import ChatOpenAI
except ImportError:  # older langchain may use openai directly
    from langchain.chat_models import ChatOpenAI  # type: ignore

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:
    ChatGoogleGenerativeAI = None  # type: ignore
try:
    from langchain_groq import ChatGroq
except Exception:
    ChatGroq = None  # type: ignore

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

PROJECT_ROOT = Path(__file__).parent
STORAGE_DIR = PROJECT_ROOT / "storage"
VSTORE_DIR = STORAGE_DIR / "faiss_lc"
BM25_PATH = STORAGE_DIR / "bm25_corpus.txt"
CORPUS_JSONL = STORAGE_DIR / "corpus.jsonl"

DEFAULT_PROMPT = """You are the IBM Knowledge RAG Assistant.
Answer the user's question clearly. Use bullet points only if helpful.
Cite sources by their file or record name inside parentheses, e.g. (source.pdf, ibm_hr_row_2).
If unsure, say you are unsure rather than hallucinating.

Question: {question}
Context Snippets:
{context}

Answer:"""

class IBMKnowledgeRAG:
    """Hybrid retrieval + answer generation engine."""

    def __init__(self, top_k_vector: int = 5, top_k_keyword: int = 5):
        load_dotenv()  # load .env if present
        self.top_k_vector = top_k_vector
        self.top_k_keyword = top_k_keyword
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
        self.vector_store = self._load_vector_store()
        self.bm25 = self._load_bm25()
        self.corpus_records = self._load_corpus_records()
        self.llm_chain = self._build_llm_chain()

    # ------------------ Loading ------------------
    def _load_vector_store(self) -> FAISS:
        if not VSTORE_DIR.exists():
            raise FileNotFoundError("FAISS vector store not found. Run build_index.py first.")
        return FAISS.load_local(str(VSTORE_DIR), self.embeddings, allow_dangerous_deserialization=True)

    def _load_bm25(self) -> BM25Okapi:
        if not BM25_PATH.exists():
            raise FileNotFoundError("BM25 corpus not found. Run build_index.py first.")
        corpus_text = BM25_PATH.read_text(encoding="utf-8")
        docs = [d for d in corpus_text.split("\n\n") if d.strip()]
        tokenized = [d.lower().split() for d in docs]
        return BM25Okapi(tokenized)

    def _load_corpus_records(self) -> List[Dict[str, str]]:
        records: List[Dict[str, str]] = []
        if CORPUS_JSONL.exists():
            with open(CORPUS_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return records

    def _build_llm_chain(self):
        prompt = PromptTemplate(input_variables=["question", "context"], template=DEFAULT_PROMPT)

        # Prefer Groq if key present (user requested Groq for now)
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key and ChatGroq is not None:
            try:
                # Fast, cost-effective default
                llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)
                return LLMChain(llm=llm, prompt=prompt)
            except Exception:
                pass

        # Next prefer Gemini
        gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if gemini_key and ChatGoogleGenerativeAI is not None:
            try:
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
                return LLMChain(llm=llm, prompt=prompt)
            except Exception:
                pass

        # Fallback to OpenAI if available
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
            return LLMChain(llm=llm, prompt=prompt)

        # Fallback pseudo LLM
        return None

    # ------------------ Retrieval ------------------
    def _vector_search(self, query: str) -> List[Tuple[Document, float]]:
        # similarity search returns Documents; FAISS similarity scores accessible via similarity attribute? Provide manual embedding similarity.
        query_emb = self.embeddings.embed_query(query)
        docs = self.vector_store.similarity_search(query, k=self.top_k_vector)
        results: List[Tuple[Document, float]] = []
        # Recompute cosine similarity for transparency
        for d in docs:
            # FAISS doesn't expose embedding directly; re-embed chunk content
            chunk_emb = self.embeddings.embed_query(d.page_content)
            sim = self._cosine(query_emb, chunk_emb)
            results.append((d, sim))
        return results

    def _keyword_search(self, query: str) -> List[Tuple[str, float]]:
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        # Pair each score with raw text (index matches corpus_records)
        pairs = []
        for i, score in enumerate(scores):
            if i < len(self.corpus_records):
                pairs.append((self.corpus_records[i]["text"], score))
        # Sort descending BM25 score
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[: self.top_k_keyword]

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        a_np = np.array(a)
        b_np = np.array(b)
        denom = (np.linalg.norm(a_np) * np.linalg.norm(b_np)) or 1e-9
        return float(np.dot(a_np, b_np) / denom)

    def _fuse_results(self, vector_results: List[Tuple[Document, float]], keyword_results: List[Tuple[str, float]]):
        # Normalize scores
        vec_scores = [v for _, v in vector_results] or [0.0]
        kw_scores = [k for _, k in keyword_results] or [0.0]
        max_vec = max(vec_scores)
        max_kw = max(kw_scores)
        fused: List[Tuple[str, str, float]] = []  # (source, text, fused_score)

        for doc, sim in vector_results:
            norm_sim = sim / max_vec if max_vec else 0.0
            fused.append((doc.metadata.get("source", "unknown"), doc.page_content, norm_sim * 0.6))
        for text, score in keyword_results:
            norm_kw = score / max_kw if max_kw else 0.0
            # Attempt to find source via corpus_records match
            source = self._find_source_by_text(text)
            fused.append((source, text, norm_kw * 0.4))

        # Combine duplicate sources by taking max score per chunk text
        fused.sort(key=lambda x: x[2], reverse=True)
        return fused[: max(self.top_k_vector, self.top_k_keyword)]

    def _find_source_by_text(self, text: str) -> str:
        for rec in self.corpus_records:
            if rec["text"] == text:
                return rec.get("source", "unknown")
        return "unknown"

    # ------------------ Answer Generation ------------------
    def _generate_answer(self, question: str, contexts: List[Tuple[str, str, float]]) -> Tuple[str, List[str]]:
        # Build context string with source markers
        context_lines = []
        cited_sources = []
        for source, chunk, _score in contexts:
            context_lines.append(f"Source: {source}\n{chunk}")
            cited_sources.append(source)
        cited_sources = list(dict.fromkeys(cited_sources))  # preserve order unique
        joined_context = "\n\n".join(context_lines[:8])

        if self.llm_chain:
            answer = self.llm_chain.run(question=question, context=joined_context).strip()
        else:
            # Fallback deterministic summarizer
            answer = (
                "(Fallback) Based on available internal documents, key points include: "
                + "; ".join([chunk[:120] + ("..." if len(chunk) > 120 else "") for _, chunk, _ in contexts[:3]])
                + f"\nSources: {', '.join(cited_sources)}"
            )
        return answer, cited_sources

    # ------------------ Public API ------------------
    def answer_question(self, question: str) -> Tuple[str, List[str], float]:
        start = time.time()
        vector_results = self._vector_search(question)
        keyword_results = self._keyword_search(question)
        fused = self._fuse_results(vector_results, keyword_results)
        answer, sources = self._generate_answer(question, fused)
        latency = time.time() - start
        return answer, sources, latency


if __name__ == "__main__":
    engine = IBMKnowledgeRAG()
    q = "What are IBM's current AI research goals?"
    ans, srcs, lat = engine.answer_question(q)
    print(f"Q: {q}\n\nA: {ans}\nSources: {srcs}\nLatency: {lat:.2f}s")
