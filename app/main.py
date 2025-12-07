"""
EduChat API - Enhanced FastAPI Application

Features:
- File upload (PDF, TXT, DOCX, MD, HTML)
- Question answering with RAG
- Streaming responses
- Multi-turn conversation
- Hybrid search (vector + keyword)
- Document management
- User authentication (optional)
- System statistics
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import os
import sys
import json
from typing import List, Optional, Dict
from datetime import datetime

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import schemas
from app.schemas import (
    AskRequest, AskResponse, UploadResponse, HealthResponse,
    ChatRequest, ChatResponse, ConversationHistory, ConversationTurn,
    DocumentInfo, DocumentListResponse, DocumentDeleteResponse, StatsResponse,
    SearchRequest, SearchResponse, HybridSearchResult,
    TextIndexRequest, TextIndexResponse,
    UserRegisterRequest, UserLoginRequest, TokenResponse, UserInfoResponse,
    ErrorResponse
)

# Import core modules
from app.embed import index_upload, index_text, get_supported_formats, model as embed_model
from app.rag import answer_question, answer_question_stream, retrieve_context
from app.db import vector_db

# Import auth (optional)
try:
    from app.auth import (
        register_user, login_user, get_current_user, require_auth,
        get_user_info, regenerate_api_key, UserRegister, UserLogin
    )
    AUTH_ENABLED = True
except ImportError:
    AUTH_ENABLED = False

# ============== Application Setup ==============

app = FastAPI(
    title="EduChat API",
    description="""
    RAG-based Educational Question Answering System
    
    ## Features
    - Question answering with retrieval-augmented generation
    - Support for multiple file formats (PDF, TXT, DOCX, MD, HTML)
    - Hybrid search combining semantic and keyword matching
    - Multi-turn conversation with context memory
    - Streaming responses for real-time interaction
    - Optional user authentication
    
    ## Authentication
    Most endpoints work without authentication. Protected endpoints require either:
    - Bearer token in Authorization header
    - API key in X-API-Key header
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============== In-Memory Storage ==============

# Conversation history storage
conversation_store: Dict[str, Dict] = {}

# ============== Startup/Shutdown Events ==============

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    # Load vector database
    db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'vector_db')
    if os.path.exists(f"{db_path}.index") and os.path.exists(f"{db_path}.meta"):
        vector_db.load(db_path)
        stats = vector_db.get_document_stats()
        print(f"Vector database loaded: {stats['total_chunks']} chunks from {stats['unique_sources']} sources")
    else:
        print("Vector database not found. Run 'python scripts/init_squad.py' to initialize.")
    
    print(f"Authentication: {'Enabled' if AUTH_ENABLED else 'Disabled'}")
    print("EduChat API is ready!")


# ============== Health & Info Endpoints ==============

@app.get("/", response_model=HealthResponse, tags=["Health"])
async def root():
    """Root endpoint - API information"""
    return HealthResponse(
        status="ok",
        message="EduChat API is running. Visit /docs for documentation.",
        version="2.0.0"
    )


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="ok",
        message="Service is healthy",
        version="2.0.0"
    )


@app.get("/stats", response_model=StatsResponse, tags=["System"])
async def get_system_stats():
    """Get system statistics"""
    stats = vector_db.get_document_stats()
    return StatsResponse(
        total_chunks=stats['total_chunks'],
        total_vectors=stats['total_vectors'],
        unique_sources=stats['unique_sources'],
        sources=stats['sources']
    )


@app.get("/config", tags=["System"])
async def get_config():
    """Get system configuration"""
    return {
        "embedding_model": "paraphrase-MiniLM-L6-v2",
        "llm_model": "Qwen3-32B",
        "default_k": 3,
        "default_chunk_size": 250,
        "supported_formats": get_supported_formats(),
        "hybrid_search_enabled": True,
        "auth_enabled": AUTH_ENABLED
    }


# ============== Document Management Endpoints ==============

@app.get("/documents", response_model=DocumentListResponse, tags=["Documents"])
async def list_documents():
    """List all indexed documents"""
    docs = vector_db.list_documents()
    stats = vector_db.get_document_stats()
    
    return DocumentListResponse(
        documents=[DocumentInfo(**d) for d in docs],
        total_documents=len(docs),
        total_chunks=stats['total_chunks']
    )


@app.post("/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_file(
    file: UploadFile = File(...),
    chunk_size: int = Query(default=250, ge=50, le=1000),
    overlap: int = Query(default=50, ge=0, le=200)
):
    """
    Upload and index a document
    
    Supported formats: PDF, TXT, DOCX, MD, HTML
    """
    # Check file type
    supported = get_supported_formats()
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in supported:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Supported: {supported}"
        )
    
    try:
        chunks_count = index_upload(file, chunk_size=chunk_size, overlap=overlap)
        
        # Save updated database
        db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'vector_db')
        vector_db.save(db_path)
        
        return UploadResponse(
            message="File processed successfully",
            filename=file.filename,
            added_chunks=chunks_count
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.post("/index-text", response_model=TextIndexResponse, tags=["Documents"])
async def index_raw_text(request: TextIndexRequest):
    """Index raw text directly without file upload"""
    try:
        chunks_count = index_text(
            request.text, 
            request.source,
            chunk_size=request.chunk_size,
            overlap=request.overlap
        )
        
        # Save updated database
        db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'vector_db')
        vector_db.save(db_path)
        
        return TextIndexResponse(
            message="Text indexed successfully",
            source=request.source,
            added_chunks=chunks_count
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error indexing text: {str(e)}")


@app.delete("/documents/{source}", response_model=DocumentDeleteResponse, tags=["Documents"])
async def delete_document(source: str):
    """Delete a document and all its chunks"""
    deleted_count = vector_db.delete_document(source)
    
    if deleted_count == 0:
        raise HTTPException(
            status_code=404,
            detail=f"Document not found: {source}"
        )
    
    # Save updated database
    db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'vector_db')
    vector_db.save(db_path)
    
    return DocumentDeleteResponse(
        message="Document deleted successfully",
        source=source,
        deleted_chunks=deleted_count
    )


# ============== Question Answering Endpoints ==============

@app.post("/ask", response_model=AskResponse, tags=["Q&A"])
async def ask_question_endpoint(request: AskRequest):
    """
    Ask a question and get an answer
    
    Uses RAG to retrieve relevant context and generate an answer.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        answer, sources = answer_question(
            request.question,
            k=request.k,
            use_hybrid=request.use_hybrid
        )
        
        return AskResponse(answer=answer, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")


@app.post("/ask/stream", tags=["Q&A"])
async def ask_question_stream_endpoint(request: AskRequest):
    """
    Ask a question with streaming response
    
    Returns Server-Sent Events (SSE) stream with tokens as they're generated.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    async def generate():
        try:
            for chunk in answer_question_stream(
                request.question,
                k=request.k,
                use_hybrid=request.use_hybrid
            ):
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )


@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search_documents(request: SearchRequest):
    """
    Search documents directly without generating an answer
    
    Useful for exploring what's in the knowledge base.
    """
    try:
        question_embedding = embed_model.encode(request.query)
        
        if request.use_hybrid:
            results = vector_db.hybrid_search(
                request.query,
                question_embedding,
                k=request.k,
                alpha=request.alpha
            )
        else:
            results = vector_db.search_vector_db(question_embedding, k=request.k)
        
        return SearchResponse(
            results=results,
            query=request.query,
            total_results=len(results)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


# ============== Multi-turn Conversation Endpoints ==============

@app.post("/chat", response_model=ChatResponse, tags=["Conversation"])
async def chat_endpoint(request: ChatRequest):
    """
    Multi-turn conversation endpoint
    
    Maintains conversation history for context-aware responses.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    session_id = request.session_id
    
    # Initialize session if needed
    if session_id not in conversation_store:
        conversation_store[session_id] = {
            "turns": [],
            "created_at": datetime.utcnow().isoformat(),
            "last_activity": datetime.utcnow().isoformat()
        }
    
    # Get history for context
    history = None
    if request.include_history and conversation_store[session_id]["turns"]:
        history = conversation_store[session_id]["turns"]
    
    try:
        answer, sources = answer_question(
            request.question,
            k=request.k,
            use_hybrid=request.use_hybrid,
            history=history
        )
        
        # Store turn
        conversation_store[session_id]["turns"].append({
            "question": request.question,
            "answer": answer,
            "timestamp": datetime.utcnow().isoformat()
        })
        conversation_store[session_id]["last_activity"] = datetime.utcnow().isoformat()
        
        # Limit history size
        if len(conversation_store[session_id]["turns"]) > 20:
            conversation_store[session_id]["turns"] = conversation_store[session_id]["turns"][-20:]
        
        return ChatResponse(
            answer=answer,
            sources=sources,
            session_id=session_id,
            history_length=len(conversation_store[session_id]["turns"])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/chat/{session_id}/history", response_model=ConversationHistory, tags=["Conversation"])
async def get_conversation_history(session_id: str):
    """Get conversation history for a session"""
    if session_id not in conversation_store:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = conversation_store[session_id]
    
    return ConversationHistory(
        session_id=session_id,
        turns=[ConversationTurn(**t) for t in session["turns"]],
        created_at=session["created_at"],
        last_activity=session["last_activity"]
    )


@app.delete("/chat/{session_id}", tags=["Conversation"])
async def clear_conversation(session_id: str):
    """Clear conversation history for a session"""
    if session_id in conversation_store:
        del conversation_store[session_id]
        return {"message": f"Session {session_id} cleared"}
    return {"message": "Session not found"}


@app.get("/chat/sessions", tags=["Conversation"])
async def list_sessions():
    """List all active conversation sessions"""
    sessions = []
    for session_id, data in conversation_store.items():
        sessions.append({
            "session_id": session_id,
            "turns_count": len(data["turns"]),
            "created_at": data["created_at"],
            "last_activity": data["last_activity"]
        })
    return {"sessions": sessions, "total": len(sessions)}


# ============== Authentication Endpoints (Optional) ==============

if AUTH_ENABLED:
    @app.post("/auth/register", response_model=TokenResponse, tags=["Authentication"])
    async def register(request: UserRegisterRequest):
        """Register a new user"""
        return register_user(UserRegister(
            username=request.username,
            password=request.password
        ))
    
    @app.post("/auth/login", response_model=TokenResponse, tags=["Authentication"])
    async def login(request: UserLoginRequest):
        """Login and get access token"""
        return login_user(UserLogin(
            username=request.username,
            password=request.password
        ))
    
    @app.get("/auth/me", response_model=UserInfoResponse, tags=["Authentication"])
    async def get_current_user_info(username: str = Depends(require_auth)):
        """Get current user information (requires authentication)"""
        return get_user_info(username)
    
    @app.post("/auth/regenerate-key", tags=["Authentication"])
    async def regenerate_user_api_key(username: str = Depends(require_auth)):
        """Regenerate API key (requires authentication)"""
        new_key = regenerate_api_key(username)
        return {"api_key": new_key, "message": "API key regenerated successfully"}


# ============== Main Entry Point ==============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
