# IBM Knowledge RAG Assistant

A minimal, end-to-end Retrieval-Augmented Generation (RAG) demo that ingests IBM PDFs, IBM AI/Research blog content, and a structured CSV dataset. It builds a FAISS vector database, performs hybrid retrieval (vector + keyword + metadata), re-ranks results, and answers questions with citations via a Streamlit UI.

## Features
- Automated data collection:
  - PDF downloader (2–3 IBM research/AI PDFs, IBM domains only)
  - Web scraper for IBM AI/Research blogs
  - Sample IBM HR-style CSV generator
- Hybrid retrieval: FAISS vector search + BM25 keyword matching + optional metadata filtering
- Indexing with LlamaIndex + LangChain
- LLM answer generation with citations (OpenAI or local fallback)
- Streamlit frontend with question input, answer, sources, and latency

## Folder Structure
```
ibm_rag_project/
├── data/
│   ├── pdfs/
│   ├── ibm_hr.csv
│   ├── website_text.txt
├── storage/               # Persisted FAISS and indexes
├── app.py                 # Streamlit UI
├── ingest_data.py         # Download/scrape/build dataset
├── build_index.py         # Build embeddings + FAISS + BM25
├── query_engine.py        # Hybrid retrieval + answer generation
├── requirements.txt
└── README.md
```

## Quickstart

1) Create and activate a Python 3.10+ environment (named `ibm_rag`).

2) Install dependencies:
```
python -m venv ibm_rag
./ibm_rag/Scripts/Activate.ps1  # Windows PowerShell
pip install -r requirements.txt
```

3) (Optional) Set your Groq, Gemini, or OpenAI API Key:
- Windows PowerShell:
```
# For Groq (preferred if set)
$env:GROQ_API_KEY="your_groq_key"
# For Gemini
$env:GEMINI_API_KEY="your_gemini_key"
# Or for OpenAI
$env:OPENAI_API_KEY="sk-..."
```
- Or create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_key
GEMINI_API_KEY=your_gemini_key
# OPENAI_API_KEY=sk-...
```

4) Ingest data (downloads PDFs, scrapes blogs, generates CSV):
```
python ingest_data.py
```

IBM-only PDFs: add your exact IBM PDF URLs to `data/ibm_pdf_urls.txt` (one per line). Only IBM domains are used.

5) Build indexes (FAISS + BM25):
```
python build_index.py
```

6) Launch the app:
```
streamlit run app.py
```

The engine will choose Groq first (if `GROQ_API_KEY` set), then Gemini, then OpenAI, else a local fallback template answer.

## Example Questions
- What are IBM’s current AI research goals?
- How does IBM apply AI in enterprises?
- What insights are available from the IBM HR dataset?

## Notes
- If no OpenAI key is set, the app will fall back to a small local model-like response for demo purposes (no external calls).
- All indexes are saved in `storage/` and can be rebuilt anytime.
- This is a demo; scraping and PDF URLs are limited for reliability.
- For exact IBM-only PDFs, populate `data/ibm_pdf_urls.txt`. Non-IBM links are ignored.

## License
This project is for educational/demo purposes. Respect website robots and terms when scraping.