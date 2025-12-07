"""
Enhanced RAG (Retrieval-Augmented Generation) Module

Features:
- Streaming response generation
- Hybrid search (vector + keyword)
- Configurable retrieval parameters
- Multi-turn conversation support
"""

from sentence_transformers import SentenceTransformer
from app.db import vector_db
from openai import OpenAI
import sys
import os
from typing import Generator, Tuple, List, Dict, Optional

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Load the sentence transformer model
MODEL_PATH = 'Models/paraphrase-MiniLM-L6-v2'
if os.path.exists(MODEL_PATH):
    model = SentenceTransformer(MODEL_PATH)
else:
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Initialize OpenAI client
client = OpenAI(
    api_key="sk-cZVYlu6yblnu6CP-CxVsCw",
    base_url="https://llmapi.paratera.com",
)


def build_prompt(question: str, context: str, history: Optional[List[Dict]] = None) -> str:
    """
    Build the prompt for the LLM
    
    Args:
        question: User's question
        context: Retrieved context from documents
        history: Optional conversation history
        
    Returns:
        Formatted prompt string
    """
    prompt_parts = []
    
    # System instruction
    prompt_parts.append(
        "You are an educational assistant. Answer questions based on the provided context. "
        "If the context doesn't contain relevant information, say so honestly. "
        "Keep your answers concise and accurate."
    )
    
    # Add conversation history if provided
    if history:
        prompt_parts.append("\nPrevious conversation:")
        for turn in history[-3:]:
            prompt_parts.append(f"User: {turn.get('question', '')}")
            prompt_parts.append(f"Assistant: {turn.get('answer', '')}")
    
    # Add context
    prompt_parts.append(f"\nContext:\n{context}")
    
    # Add current question
    prompt_parts.append(f"\nQuestion: {question}")
    prompt_parts.append("\nAnswer:")
    
    return "\n".join(prompt_parts)


def retrieve_context(question: str, k: int = 3, use_hybrid: bool = True, 
                    alpha: float = 0.7) -> Tuple[str, List[Dict]]:
    """
    Retrieve relevant context for a question
    
    Args:
        question: User's question
        k: Number of documents to retrieve
        use_hybrid: Whether to use hybrid search
        alpha: Weight for vector search in hybrid mode
        
    Returns:
        Tuple of (context_string, results_list)
    """
    # Generate question embedding
    question_embedding = model.encode(question)
    
    # Search for relevant documents
    if use_hybrid:
        results = vector_db.hybrid_search(question, question_embedding, k=k, alpha=alpha)
    else:
        results = vector_db.search_vector_db(question_embedding, k=k)
    
    # Build context string
    context = "\n\n".join([result['text'] for result in results])
    
    return context, results


def answer_question(question: str, k: int = 3, use_hybrid: bool = True,
                   history: Optional[List[Dict]] = None) -> Tuple[str, List[str]]:
    """
    Answer a question using RAG approach (non-streaming)
    
    Args:
        question: User's question
        k: Number of documents to retrieve
        use_hybrid: Whether to use hybrid search
        history: Optional conversation history
        
    Returns:
        Tuple of (answer, sources)
    """
    # Retrieve context
    context, results = retrieve_context(question, k=k, use_hybrid=use_hybrid)
    
    if not results:
        return "I couldn't find any relevant information to answer your question.", []
    
    # Build prompt
    prompt = build_prompt(question, context, history)
    
    # Call LLM API
    try:
        messages = [{"role": "user", "content": prompt}]
        
        completion = client.chat.completions.create(
            model="Qwen3-32B",
            messages=messages,
            extra_body={"enable_thinking": False},
            stream=True,
        )
        
        answer_content = ""
        
        for chunk in completion:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                answer_content += delta.content
                
        answer = answer_content.strip() if answer_content else "I couldn't generate an answer."
        
    except Exception as e:
        answer = f"Error generating answer: {str(e)}"
    
    # Extract unique sources
    sources = list(set(result['source'] for result in results))
    
    return answer, sources


def answer_question_stream(question: str, k: int = 3, use_hybrid: bool = True,
                          history: Optional[List[Dict]] = None) -> Generator[Dict, None, None]:
    """
    Answer a question using RAG approach with streaming
    
    Args:
        question: User's question
        k: Number of documents to retrieve
        use_hybrid: Whether to use hybrid search
        history: Optional conversation history
        
    Yields:
        Dict with either:
        - {'token': str} for partial answer
        - {'sources': List[str]} for sources (at the end)
        - {'context': List[Dict]} for retrieved context (at the start)
        - {'error': str} for errors
    """
    # Retrieve context
    context, results = retrieve_context(question, k=k, use_hybrid=use_hybrid)
    
    if not results:
        yield {"token": "I couldn't find any relevant information to answer your question."}
        yield {"sources": []}
        return
    
    # Yield context info first
    yield {"context": [{"text": r['text'][:200] + "...", "source": r['source']} for r in results]}
    
    # Build prompt
    prompt = build_prompt(question, context, history)
    
    # Call LLM API with streaming
    try:
        messages = [{"role": "user", "content": prompt}]
        
        completion = client.chat.completions.create(
            model="Qwen3-32B",
            messages=messages,
            extra_body={"enable_thinking": False},
            stream=True,
        )
        
        for chunk in completion:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                yield {"token": delta.content}
        
        # Send sources at the end
        sources = list(set(result['source'] for result in results))
        yield {"sources": sources}
        
    except Exception as e:
        yield {"error": str(e)}
        sources = list(set(result['source'] for result in results))
        yield {"sources": sources}


def get_similar_questions(question: str, k: int = 5) -> List[Dict]:
    """
    Find similar questions/chunks in the database
    
    Useful for "Did you mean?" suggestions
    
    Args:
        question: User's question
        k: Number of similar items to return
        
    Returns:
        List of similar chunks with scores
    """
    question_embedding = model.encode(question)
    results = vector_db.search_vector_db(question_embedding, k=k)
    return results


def test_rag():
    """Test function to verify RAG system functionality"""
    # Load the vector database if it exists
    db_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'vector_db')
    if os.path.exists(f"{db_file}.index") and os.path.exists(f"{db_file}.meta"):
        vector_db.load(db_file)
        print("Vector database loaded successfully.")
        print(f"Total chunks: {len(vector_db.metadata)}")
    else:
        print("Warning: Vector database not found. Run init_squad.py first.")
        return
    
    # Test question
    test_question = "What is natural language processing?"
    print(f"\nTesting with question: '{test_question}'")
    
    # Test non-streaming with hybrid search
    print("\n--- Non-streaming test (Hybrid Search) ---")
    answer, sources = answer_question(test_question, use_hybrid=True)
    print(f"Answer: {answer[:500]}...")
    print(f"Sources: {sources}")
    
    # Test streaming
    print("\n--- Streaming test ---")
    print("Answer: ", end="", flush=True)
    for chunk in answer_question_stream(test_question):
        if "token" in chunk:
            print(chunk["token"], end="", flush=True)
        elif "sources" in chunk:
            print(f"\nSources: {chunk['sources']}")
        elif "context" in chunk:
            print(f"\n[Retrieved {len(chunk['context'])} documents]")


if __name__ == "__main__":
    test_rag()
