# IBM Knowledge RAG Assistant

## 🎯 Project Overview

**Enterprise-grade AI-powered knowledge retrieval system** combining hybrid search technology with professional Gradio interface for intelligent document retrieval and question answering.

**Live Demo:** `http://localhost:7861`

---

## ✨ Key Features

- **Hybrid Retrieval Engine**: Vector similarity + BM25 keyword search
- **LLM Integration**: Groq LLaMA API for intelligent answer generation
- **Enterprise UI**: Professional dark-themed Gradio interface with IBM branding
- **Performance Metrics**: Real-time latency and processing analytics
- **Debug Mode**: Technical insights into retrieval process
- **Responsive Design**: Mobile-friendly adaptive layout

---

## 🏗️ Architecture

### Tech Stack
- **Frontend**: Gradio (Python web framework)
- **LLM**: Groq LLaMA (Fast inference API)
- **Search**: 
  - Vector: Sentence Transformers + FAISS
  - Keyword: BM25 (Okapi ranking)
- **Language**: Python 3.8+

### System Components

```
┌─────────────────────────────────────────┐
│   User Query (Gradio Interface)         │
└──────────────┬──────────────────────────┘
               │
       ┌───────▼───────┐
       │  Query Parser │
       └───────┬───────┘
               │
    ┌──────────┴──────────┐
    │                     │
┌───▼────┐          ┌────▼────┐
│ Vector │          │ Keyword │
│ Search │          │ Search  │
│(FAISS) │          │ (BM25)  │
└───┬────┘          └────┬────┘
    │                     │
    └──────────┬──────────┘
               │
        ┌──────▼──────┐
        │   Reranker  │
        └──────┬──────┘
               │
        ┌──────▼──────────┐
        │ Groq LLaMA API  │
        └──────┬──────────┘
               │
        ┌──────▼──────┐
        │   Response  │  │
        └─────────────┘
```

---

## 📊 Performance Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| Vector Search | ~50ms | FAISS optimized |
| Keyword Search | ~30ms | BM25 ranking |
| LLM Response | ~1-2s | Groq API latency |
| Total E2E | ~2-3s | Including formatting |

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <repo-url>
cd ibm_rag_project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create `.env` file:
```
GROQ_API_KEY=your_groq_api_key_here
VECTOR_DB_PATH=./data/vector_db
KNOWLEDGE_BASE_PATH=./data/documents
```

### Run Application

```bash
python gradio_app.py
```

Access at: `http://127.0.0.1:7861`

---

## 📁 Project Structure

```
ibm_rag_project/
├── gradio_app.py              # Main Gradio interface
├── query_engine.py            # RAG engine logic
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables
├── README.md                  # This file
│
├── data/
│   ├── documents/             # Knowledge base documents
│   ├── vector_db/             # FAISS vector store
│   ├── logo/
│   │   └── ibm_logo.png       # IBM branding
│   └── configs/
│       └── retrieval_config.json
│
├── models/
│   ├── embeddings/            # Sentence Transformer models
│   └── ranking/               # Reranker models
│
```

---

## 💻 Code Highlights

### Hybrid Search Implementation

```python
# Simultaneous vector + keyword search
vector_results = search_vector_db(query, top_k=5)      # FAISS
keyword_results = search_bm25(query, top_k=5)          # BM25

# Merge and rerank results
merged_results = rerank_results(
    vector_results, 
    keyword_results
)

# Generate answer using LLM
answer = llm.generate(query, merged_results)
```

### Performance Optimization

- **FAISS Indexing**: O(log n) search complexity
- **BM25 Ranking**: Efficient term matching
- **Batch Processing**: Parallel search execution
- **Caching**: Query result memoization

---

## 🎨 UI/UX Features

### Dark Enterprise Theme
- Professional IBM color scheme (#0f62fe primary)
- High contrast for accessibility
- Responsive grid layout
- Smooth animations & transitions

### User Experience
- Real-time processing feedback
- Tabbed results view (Answer | Sources | Metrics | Debug)
- Example questions for quick start
- Advanced configuration panel
- Technical debug information

---

## 🔒 Security & Compliance

- ✅ Environment variable protection (API keys in `.env`)
- ✅ Input validation & sanitization
- ✅ Error handling without data leakage
- ✅ CORS configuration for API security
- ✅ Rate limiting ready

---

## 📈 Deployment Options

### Local Development
```bash
python gradio_app.py
```

### Docker Deployment
```bash
docker build -t ibm-rag .
docker run -p 7861:7861 ibm-rag
```

### Hugging Face Spaces
1. Push to GitHub
2. Connect Hugging Face Spaces
3. Auto-deploy with CI/CD

### Cloud Platforms
- **AWS**: EC2 + Lambda for serverless
- **Azure**: App Service + Cognitive Services
- **GCP**: Cloud Run + Vertex AI

---

## 📚 Learning Resources

- [Gradio Documentation](https://gradio.app/)
- [Groq API Docs](https://console.groq.com/)
- [FAISS Tutorial](https://github.com/facebookresearch/faiss)
- [BM25 Algorithm](https://en.wikipedia.org/wiki/Okapi_BM25)
- [RAG Paper](https://arxiv.org/abs/2005.11401)

---

## 🤝 Contributing

1. Fork repository
2. Create feature branch (`git checkout -b feature/enhancement`)
3. Commit changes (`git commit -m 'Add feature'`)
4. Push to branch (`git push origin feature/enhancement`)
5. Open Pull Request

---

## 📝 License

MIT License - See LICENSE file for details

---

## 👤 Author

**Dandu Yeshwanth** | AI/ML Engineer
- LinkedIn: [https://www.linkedin.com/in/yeshwanthdandu/]
- GitHub: [https://github.com/YeshwanthDandu180903]
- Email: yeshwanthdandu2003@gmail.com

---

## 🎓 Learning Outcomes

This project demonstrates:
- ✅ LLM Integration & API management
- ✅ Hybrid search architecture design
- ✅ Vector databases (FAISS)
- ✅ Full-stack web development (Python)
- ✅ UI/UX with professional design
- ✅ Performance optimization
- ✅ Enterprise software best practices
- ✅ DevOps & deployment pipelines
