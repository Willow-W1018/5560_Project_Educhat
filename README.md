# EduChat - Educational AI Assistant

A comprehensive RAG-based (Retrieval-Augmented Generation) educational question answering system with advanced features.

## Features Overview

| Feature | Description | Status |
|---------|-------------|--------|
| **Web Interface** | Beautiful, responsive chat UI |
| **Multi-turn Conversation** | Context-aware follow-up questions |
| **Document Management** | Upload, list, delete documents |
| **Hybrid Search** | Vector + BM25 keyword search |
| **Evaluation Metrics** | Quality assessment tools |
| **User Authentication** | JWT & API key auth |
| **Multiple File Formats** | PDF, TXT, DOCX, MD, HTML |
## Architecture

┌─────────────────────────────────────────────────────────────────┐
│                      Web Interface (frontend/)                   │
│                    Chat UI • Document Manager                    │
└─────────────────────────────┬───────────────────────────────────┘
                              │ HTTP/SSE
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI Server (app/main.py)                 │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐  │
│  │ /ask    │ │ /chat   │ │/upload  │ │/search  │ │ /auth   │  │
│  │         │ │         │ │         │ │         │ │         │  │
│  │ Q&A     │ │ Multi-  │ │ Doc     │ │ Direct  │ │ JWT/Key │  │
│  │ Stream  │ │ turn    │ │ Index   │ │ Query   │ │ Auth    │  │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘  │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      RAG Pipeline (app/rag.py)                   │
│                                                                  │
│   Question ──► Embedding ──► Hybrid Search ──► Context ──► LLM  │
│                  │              │                          │     │
│                  ▼              ▼                          ▼     │
│             MiniLM-L6    FAISS + BM25               Qwen3-32B   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Vector Database (app/db.py)                     │
│  ┌──────────────────┐    ┌──────────────────┐                   │
│  │   FAISS Index    │    │   BM25 Index     │                   │
│  │  (Semantic)      │    │  (Keyword)       │                   │
│  └──────────────────┘    └──────────────────┘                   │
│  ┌──────────────────────────────────────────┐                   │
│  │              Metadata Store               │                   │
│  │   (text, source, timestamps, etc.)       │                   │
│  └──────────────────────────────────────────┘                   │
└─────────────────────────────────────────────────────────────────┘


## Project Structure

EduChat/
├── app/
│   ├── __init__.py           # Package initialization
│   ├── main.py               # FastAPI application with all endpoints
│   ├── schemas.py            # Pydantic request/response models
│   ├── db.py                 # Vector DB with Hybrid Search
│   ├── embed.py              # Text extraction & embedding
│   ├── rag.py                # RAG pipeline with streaming
│   ├── auth.py               # User authentication (JWT/API key)
│   └── train-v2.0.json       # SQuAD 2.0 dataset
├── scripts/
│   ├── init_squad.py         # Database initialization
│   └── evaluate.py           # Evaluation metrics
├── frontend/
│   └── index.html            # Web chat interface
├── data/
│   ├── vector_db.index       # FAISS index
│   ├── vector_db.meta        # Metadata + BM25 index
│   └── users.json            # User data (if auth enabled)
├── Models/
│   └── paraphrase-MiniLM-L6-v2/  # Local embedding model
├── docker/
│   └── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md



## Setup (First Time)

### 1. Download SQuAD Dataset
# Download train-v2.0.json to app/ folder
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json -O app/train-v2.0.json

### 2. Download Embedding Model

# The model will auto-download on first run
# Or manually download to Models/ folder
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-MiniLM-L6-v2').save('Models/paraphrase-MiniLM-L6-v2')"

### 3. Initialize Database

python scripts/init_squad.py



## Quick Start

### Prerequisites

- Python 3.10+
- 4GB+ RAM (for embedding model)
- SQuAD 2.0 dataset file (`train-v2.0.json`)

### Option 1: Local Development

# 1. Clone the repository
git clone https://github.com/Willow-W1018/5560_Project_Educhat
cd educhat

# 2. Create virtual environment
python3.10 -m venv venv  # Replace the version for your own but recommend using 3.10
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Initialize vector database
python scripts/init_squad.py --max-chunks 1000

# 5. Start the API server
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 6. Open frontend (in another terminal or browser)
# Simply open frontend/index.html in your browser
# Or serve it: python -m http.server 3000 --directory frontend

### Option 2: Docker

# Build and start all services
docker-compose up --build

# API: http://localhost:8000
# Frontend: http://localhost:3000

## API Documentation

### Interactive Docs
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
# Simply open frontend/index.html in your browser

### Endpoint Summary

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/health` | Health check |
| `GET` | `/stats` | System statistics |
| `GET` | `/config` | System configuration |
| `GET` | `/documents` | List indexed documents |
| `POST` | `/upload` | Upload document |
| `POST` | `/index-text` | Index raw text |
| `DELETE` | `/documents/{source}` | Delete document |
| `POST` | `/ask` | Ask question |
| `POST` | `/ask/stream` | Ask with streaming |
| `POST` | `/search` | Direct search |
| `POST` | `/chat` | Multi-turn chat |
| `GET` | `/chat/{session_id}/history` | Get chat history |
| `DELETE` | `/chat/{session_id}` | Clear chat |
| `POST` | `/auth/register` | Register user |
| `POST` | `/auth/login` | Login |
| `GET` | `/auth/me` | Get user info |

## Usage Examples

### Basic Question Answering

curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is machine learning?", "use_hybrid": true}'

### Streaming Response

curl -X POST "http://localhost:8000/ask/stream" \
  -H "Content-Type: application/json" \
  -d '{"question": "Explain neural networks"}'

### Multi-turn Conversation

# First question
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is NLP?", "session_id": "user123"}'

# Follow-up (uses context)
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are its applications?", "session_id": "user123"}'

### Upload Document

curl -X POST "http://localhost:8000/upload" \
  -F "file=@lecture_notes.pdf"

### Python Client

import requests

# Ask question
response = requests.post(
    "http://localhost:8000/ask",
    json={"question": "What is deep learning?", "use_hybrid": True}
)
print(response.json())

# Streaming response
response = requests.post(
    "http://localhost:8000/ask/stream",
    json={"question": "Explain transformers"},
    stream=True
)
for line in response.iter_lines():
    if line:
        print(line.decode())

## Evaluation

Run the evaluation script to measure system quality:

python scripts/evaluate.py

### Metrics Measured

| Metric | Description |
|--------|-------------|
| **Retrieval Relevance** | Semantic similarity between query and retrieved docs |
| **Keyword Coverage** | Presence of expected keywords in answers |
| **Response Coherence** | Semantic similarity between question and answer |
| **Latency** | Response time in seconds |
| **Hybrid Search Improvement** | Quality gain from hybrid vs vector-only search |

### Sample Output

EVALUATION SUMMARY
===========================================================

Success Rate: 8/8 (100.0%)

Average Metrics:
   Retrieval Relevance:  0.542
   Keyword Coverage:     0.438
   Response Coherence:   0.621
   Average Latency:      2.341s

Hybrid Search Improvement: +0.087

Performance by Difficulty:
   Easy: 2 tests, coherence=0.712, latency=1.823s
   Medium: 4 tests, coherence=0.598, latency=2.456s
   Hard: 2 tests, coherence=0.553, latency=2.744s

OVERALL QUALITY SCORE: 0.555
   Assessment: GOOD

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `JWT_SECRET` | Secret key for JWT tokens | `change-this-in-production` |
| `LLM_API_KEY` | API key for LLM service | (set in code) |
| `LLM_BASE_URL` | LLM API endpoint | (set in code) |

### Customization

- **Chunk Size**: Adjust in `/upload` endpoint query params
- **Search Results**: Adjust `k` parameter in requests
- **Hybrid Weight**: Adjust `alpha` in search requests (0=keyword only, 1=vector only)

## Authentication (Optional)

The system supports optional JWT and API key authentication:

# Register
curl -X POST "http://localhost:8000/auth/register" \
  -H "Content-Type: application/json" \
  -d '{"username": "student1", "password": "secure123"}'

# Login
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "student1", "password": "secure123"}'

# Use token
curl -X GET "http://localhost:8000/auth/me" \
  -H "Authorization: Bearer <token>"

# Or use API key
curl -X GET "http://localhost:8000/auth/me" \
  -H "X-API-Key: edu_xxxxx"

## Value-Added Features

Beyond basic RAG implementation:

1. **Hybrid Search** - Combines semantic similarity with keyword matching for better retrieval
2. **Streaming Responses** - Real-time token generation for better UX
3. **Multi-turn Conversations** - Context-aware follow-up questions
4. **Document Management** - Full CRUD operations for knowledge base
5. **Multiple File Formats** - PDF, DOCX, TXT, Markdown, HTML support
6. **Evaluation Suite** - Quantitative quality metrics
7. **User Authentication** - JWT and API key support
8. **Production-Ready Docker** - Multi-stage build, health checks

## Team Responsibilities

| Role | Responsibilities |
|------|------------------|
| **Coder A** | Backend API, Docker, Frontend, Documentation |
| **Coder B** | RAG Pipeline, Embeddings, Vector DB, Evaluation |

## License

MIT License

## Acknowledgments

- [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/) for the dataset
- [Sentence Transformers](https://www.sbert.net/) for embeddings
- [FAISS](https://github.com/facebookresearch/faiss) for vector search
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
