"""
Pydantic schemas for request and response validation

All API request/response models are defined here for type safety and documentation
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


# ============== Basic Request/Response Models ==============

class AskRequest(BaseModel):
    """Request model for asking questions"""
    question: str = Field(..., min_length=1, description="The question to ask")
    k: int = Field(default=3, ge=1, le=10, description="Number of documents to retrieve")
    use_hybrid: bool = Field(default=True, description="Use hybrid search (vector + keyword)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is natural language processing?",
                "k": 3,
                "use_hybrid": True
            }
        }


class AskResponse(BaseModel):
    """Response model for question answers"""
    answer: str
    sources: List[str]
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Natural language processing (NLP) is a field of AI...",
                "sources": ["SQuAD 2.0", "lecture_notes.pdf"]
            }
        }


class UploadResponse(BaseModel):
    """Response model for file upload"""
    message: str
    filename: str
    added_chunks: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "File processed successfully",
                "filename": "lecture_notes.pdf",
                "added_chunks": 15
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    message: str
    version: str = "2.0.0"


# ============== Multi-turn Conversation Models ==============

class ChatRequest(BaseModel):
    """Request model for multi-turn conversation"""
    question: str = Field(..., min_length=1)
    session_id: str = Field(default="default", description="Session ID for conversation tracking")
    include_history: bool = Field(default=True, description="Include conversation history in context")
    k: int = Field(default=3, ge=1, le=10)
    use_hybrid: bool = Field(default=True)
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What are its main applications?",
                "session_id": "user123",
                "include_history": True
            }
        }


class ChatResponse(BaseModel):
    """Response model for chat"""
    answer: str
    sources: List[str]
    session_id: str
    history_length: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "NLP has many applications including...",
                "sources": ["SQuAD 2.0"],
                "session_id": "user123",
                "history_length": 2
            }
        }


class ConversationTurn(BaseModel):
    """Single turn in a conversation"""
    question: str
    answer: str
    timestamp: Optional[str] = None


class ConversationHistory(BaseModel):
    """Full conversation history"""
    session_id: str
    turns: List[ConversationTurn]
    created_at: str
    last_activity: str


# ============== Document Management Models ==============

class DocumentInfo(BaseModel):
    """Information about an indexed document"""
    source: str
    chunk_count: int


class DocumentListResponse(BaseModel):
    """Response model for document listing"""
    documents: List[DocumentInfo]
    total_documents: int
    total_chunks: int


class DocumentDeleteResponse(BaseModel):
    """Response model for document deletion"""
    message: str
    source: str
    deleted_chunks: int


class StatsResponse(BaseModel):
    """System statistics response"""
    total_chunks: int
    total_vectors: int
    unique_sources: int
    sources: Dict[str, int]


# ============== Search Models ==============

class SearchResult(BaseModel):
    """Single search result"""
    text: str
    source: str
    score: float
    index: int


class HybridSearchResult(BaseModel):
    """Hybrid search result with detailed scores"""
    text: str
    source: str
    combined_score: float
    vector_score: float
    bm25_score: float
    index: int


class SearchRequest(BaseModel):
    """Request model for direct search"""
    query: str = Field(..., min_length=1)
    k: int = Field(default=5, ge=1, le=20)
    use_hybrid: bool = Field(default=True)
    alpha: float = Field(default=0.7, ge=0, le=1, description="Weight for vector search in hybrid mode")


class SearchResponse(BaseModel):
    """Response model for search"""
    results: List[Dict[str, Any]]
    query: str
    total_results: int


# ============== Streaming Models ==============

class StreamChunk(BaseModel):
    """Chunk in streaming response"""
    token: Optional[str] = None
    sources: Optional[List[str]] = None
    context: Optional[List[Dict]] = None
    error: Optional[str] = None
    done: bool = False


# ============== User Authentication Models ==============

class UserRegisterRequest(BaseModel):
    """Request model for user registration"""
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6, max_length=100)
    
    class Config:
        json_schema_extra = {
            "example": {
                "username": "student1",
                "password": "securepassword123"
            }
        }


class UserLoginRequest(BaseModel):
    """Request model for user login"""
    username: str
    password: str


class TokenResponse(BaseModel):
    """Response model for authentication token"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 86400
    api_key: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
                "token_type": "bearer",
                "expires_in": 86400,
                "api_key": "edu_abc123..."
            }
        }


class UserInfoResponse(BaseModel):
    """Response model for user information"""
    username: str
    created_at: str
    usage_count: int
    has_api_key: bool


# ============== Evaluation Models ==============

class EvaluationMetrics(BaseModel):
    """Metrics for a single evaluation"""
    retrieval_relevance: float
    keyword_coverage: float
    response_coherence: float
    latency_seconds: float


class EvaluationResult(BaseModel):
    """Single evaluation test result"""
    question: str
    topic: str
    answer: str
    sources: List[str]
    metrics: EvaluationMetrics


class EvaluationSummary(BaseModel):
    """Summary of evaluation results"""
    total_tests: int
    successful_tests: int
    avg_retrieval_relevance: float
    avg_keyword_coverage: float
    avg_response_coherence: float
    avg_latency_seconds: float
    overall_quality_score: float


# ============== Error Models ==============

class ErrorResponse(BaseModel):
    """Standard error response"""
    detail: str
    error_code: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "detail": "An error occurred",
                "error_code": "ERR_001"
            }
        }


# ============== Text Input Models ==============

class TextIndexRequest(BaseModel):
    """Request model for indexing raw text"""
    text: str = Field(..., min_length=10)
    source: str = Field(..., min_length=1, description="Source identifier for the text")
    chunk_size: int = Field(default=250, ge=50, le=1000)
    overlap: int = Field(default=50, ge=0, le=200)
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Machine learning is a subset of artificial intelligence...",
                "source": "ml_overview",
                "chunk_size": 250,
                "overlap": 50
            }
        }


class TextIndexResponse(BaseModel):
    """Response model for text indexing"""
    message: str
    source: str
    added_chunks: int


# ============== Configuration Models ==============

class SystemConfig(BaseModel):
    """System configuration"""
    embedding_model: str
    llm_model: str
    default_k: int
    default_chunk_size: int
    supported_formats: List[str]
    hybrid_search_enabled: bool
    auth_enabled: bool
