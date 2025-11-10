"""
Build indexes for IBM Knowledge RAG Assistant.

- Reads data from data/pdfs, data/website_text.txt, data/ibm_hr.csv
- Chunks and embeds with sentence-transformers
- Stores vectors in FAISS (LangChain VectorStore) under storage/
- Builds and persists a simple BM25 keyword corpus for hybrid retrieval

Run:
    python build_index.py
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple, Dict

import json
import pandas as pd
from PyPDF2 import PdfReader

# LangChain components
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document


# Keyword/BM25
from rank_bm25 import BM25Okapi

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
WEBSITE_TEXT_PATH = DATA_DIR / "website_text.txt"
CSV_PATH = DATA_DIR / "ibm_hr.csv"
STORAGE_DIR = PROJECT_ROOT / "storage"
VSTORE_DIR = STORAGE_DIR / "faiss_lc"
BM25_PATH = STORAGE_DIR / "bm25_corpus.txt"
CORPUS_JSONL = STORAGE_DIR / "corpus.jsonl"  # chunk texts + metadata for BM25 and citations

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_pdfs(pdf_dir: Path) -> List[Tuple[str, str]]:
    """Return list of (source, text) from PDFs."""
    out: List[Tuple[str, str]] = []
    for p in sorted(pdf_dir.glob("*.pdf")):
        try:
            reader = PdfReader(str(p))
            pages = []
            for page in reader.pages:
                txt = page.extract_text() or ""
                if txt.strip():
                    pages.append(txt)
            if pages:
                out.append((p.name, "\n".join(pages)))
        except Exception:
            continue
    return out


def load_website_text(path: Path) -> List[Tuple[str, str]]:
    docs: List[Tuple[str, str]] = []
    if path.exists():
        text = path.read_text(encoding="utf-8")
        # Split by double newline paragraphs for BM25 granularity
        for i, chunk in enumerate(text.split("\n\n")):
            chunk = chunk.strip()
            if chunk:
                docs.append((f"website_text_{i}", chunk))
    return docs


def load_csv(path: Path) -> List[Tuple[str, str]]:
    docs: List[Tuple[str, str]] = []
    if path.exists():
        df = pd.read_csv(path)
        # Turn each row into a paragraph-like text for retrieval
        for idx, row in df.iterrows():
            meta_pairs = [f"{col}: {row[col]}" for col in df.columns]
            docs.append((f"ibm_hr_row_{idx}", "; ".join(meta_pairs)))
    return docs


def chunk_documents(raw_docs: List[Tuple[str, str]]) -> List[Document]:
    """Chunk raw (source, text) into LangChain Documents with metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", ".", " "]
    )
    docs: List[Document] = []
    for source, text in raw_docs:
        for chunk in splitter.split_text(text):
            if chunk.strip():
                docs.append(Document(page_content=chunk, metadata={"source": source}))
    return docs


def build_faiss_vectorstore(chunks: List[Document]) -> FAISS:
    """Build and persist a FAISS vector store using sentence-transformers embeddings."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    vstore = FAISS.from_documents(chunks, embedding=embeddings)
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    VSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vstore.save_local(str(VSTORE_DIR))
    return vstore


def persist_corpus_jsonl(chunks: List[Document]) -> None:
    """Save chunk texts + metadata for BM25 and citations."""
    with open(CORPUS_JSONL, "w", encoding="utf-8") as f:
        for d in chunks:
            rec: Dict[str, str] = {
                "text": d.page_content,
                "source": d.metadata.get("source", "unknown"),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def build_bm25_corpus(chunks: List[Document]) -> BM25Okapi:
    corpus_texts = [d.page_content for d in chunks]
    tokenized = [t.lower().split() for t in corpus_texts]
    bm25 = BM25Okapi(tokenized)
    # Save raw corpus for quick reload
    BM25_PATH.write_text("\n\n".join(corpus_texts), encoding="utf-8")
    return bm25


def main():
    all_docs: List[Tuple[str, str]] = []
    all_docs.extend(load_pdfs(PDF_DIR))
    all_docs.extend(load_website_text(WEBSITE_TEXT_PATH))
    all_docs.extend(load_csv(CSV_PATH))

    if not all_docs:
        print("No documents found. Run ingest_data.py first.")
        return

    print(f"Loaded {len(all_docs)} source documents.")

    print("Chunking documents...")
    chunks = chunk_documents(all_docs)
    print(f"Created {len(chunks)} chunks.")

    print("Building FAISS index (LangChain) with sentence-transformers embeddings...")
    _ = build_faiss_vectorstore(chunks)
    print(f"FAISS vector store saved to: {VSTORE_DIR}")

    print("Persisting corpus metadata for citations...")
    persist_corpus_jsonl(chunks)

    print("Building BM25 keyword index (and saving corpus)...")
    _ = build_bm25_corpus(chunks)
    print(f"BM25 corpus saved to: {BM25_PATH}")

    print("Done. Indexes are ready in 'storage/'.")


if __name__ == "__main__":
    main()
