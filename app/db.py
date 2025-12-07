"""
Enhanced Vector Database Module

Features:
- FAISS vector similarity search
- BM25 keyword search (Hybrid Search)
- Document management (list, delete, stats)
- Persistent storage
"""

import faiss
import numpy as np
import pickle
import os
import re
from typing import List, Dict, Optional, Tuple
from collections import Counter
import math


class BM25:
    """
    BM25 keyword search implementation for Hybrid Search
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = []
        self.doc_lengths = []
        self.avg_doc_length = 0
        self.doc_freqs = {}
        self.idf = {}
        self.doc_term_freqs = []
        
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase and split on non-alphanumeric"""
        text = text.lower()
        tokens = re.findall(r'\b[a-z0-9]+\b', text)
        return tokens
    
    def fit(self, corpus: List[str]):
        """Build BM25 index from corpus"""
        self.corpus = corpus
        self.doc_lengths = []
        self.doc_term_freqs = []
        term_doc_count = Counter()
        
        # Calculate term frequencies for each document
        for doc in corpus:
            tokens = self._tokenize(doc)
            self.doc_lengths.append(len(tokens))
            
            term_freq = Counter(tokens)
            self.doc_term_freqs.append(term_freq)
            
            # Count documents containing each term
            for term in set(tokens):
                term_doc_count[term] += 1
        
        # Calculate average document length
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0
        
        # Calculate IDF for each term
        n_docs = len(corpus)
        self.idf = {}
        for term, df in term_doc_count.items():
            self.idf[term] = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
    
    def search(self, query: str, k: int = 5) -> List[Tuple[int, float]]:
        """
        Search for top-k documents matching query
        
        Returns: List of (doc_index, score) tuples
        """
        query_tokens = self._tokenize(query)
        scores = []
        
        for idx, (doc_tf, doc_len) in enumerate(zip(self.doc_term_freqs, self.doc_lengths)):
            score = 0
            for term in query_tokens:
                if term in doc_tf:
                    tf = doc_tf[term]
                    idf = self.idf.get(term, 0)
                    
                    # BM25 formula
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)
                    score += idf * numerator / denominator
            
            scores.append((idx, score))
        
        # Sort by score descending and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]
    
    def add_document(self, doc: str):
        """Add a single document to the index"""
        tokens = self._tokenize(doc)
        self.corpus.append(doc)
        self.doc_lengths.append(len(tokens))
        
        term_freq = Counter(tokens)
        self.doc_term_freqs.append(term_freq)
        
        # Update IDF (simplified - full rebuild for accuracy)
        n_docs = len(self.corpus)
        for term in set(tokens):
            df = sum(1 for tf in self.doc_term_freqs if term in tf)
            self.idf[term] = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
        
        # Update average document length
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths)


class VectorDB:
    """
    Enhanced Vector Database with Hybrid Search
    
    Combines:
    - FAISS for semantic similarity search
    - BM25 for keyword matching
    """
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata = []
        self.bm25 = BM25()
        
    def add_to_vector_db(self, chunks: List[str], vectors: np.ndarray, source: str):
        """Add vectors and corresponding metadata to the database"""
        # Add vectors to FAISS index
        vectors_array = np.array(vectors).astype('float32')
        self.index.add(vectors_array)
        
        # Store metadata
        for chunk in chunks:
            self.metadata.append({
                'text': chunk,
                'source': source
            })
        
        # Update BM25 index
        self.bm25.fit([m['text'] for m in self.metadata])
    
    def search_vector_db(self, query_vec: np.ndarray, k: int = 5) -> List[Dict]:
        """
        Search for k nearest neighbors using vector similarity only
        
        Args:
            query_vec: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of result dicts with text, source, and distance
        """
        if self.index.ntotal == 0:
            return []
        
        # Search in FAISS index
        query_array = np.array([query_vec]).astype('float32')
        distances, indices = self.index.search(query_array, min(k, self.index.ntotal))
        
        # Retrieve metadata for results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata) and idx >= 0:
                results.append({
                    'text': self.metadata[idx]['text'],
                    'source': self.metadata[idx]['source'],
                    'distance': float(distances[0][i]),
                    'index': int(idx)
                })
                
        return results
    
    def hybrid_search(self, query: str, query_vec: np.ndarray, k: int = 5, 
                      alpha: float = 0.7) -> List[Dict]:
        """
        Hybrid search combining vector similarity and BM25 keyword matching
        
        Args:
            query: Query text for BM25
            query_vec: Query embedding vector for FAISS
            k: Number of results to return
            alpha: Weight for vector search (1-alpha for BM25)
                   0.7 means 70% vector, 30% keyword
            
        Returns:
            List of result dicts with combined scores
        """
        if self.index.ntotal == 0:
            return []
        
        # Get more candidates than needed for re-ranking
        n_candidates = min(k * 3, self.index.ntotal)
        
        # Vector search
        query_array = np.array([query_vec]).astype('float32')
        distances, indices = self.index.search(query_array, n_candidates)
        
        # Normalize vector scores
        max_dist = max(distances[0]) if max(distances[0]) > 0 else 1
        vector_scores = {int(idx): 1 - (dist / max_dist) 
                        for idx, dist in zip(indices[0], distances[0]) if idx >= 0}
        
        # BM25 search
        bm25_results = self.bm25.search(query, n_candidates)
        max_bm25 = max(score for _, score in bm25_results) if bm25_results and bm25_results[0][1] > 0 else 1
        bm25_scores = {idx: score / max_bm25 for idx, score in bm25_results if score > 0}
        
        # Combine scores
        all_indices = set(vector_scores.keys()) | set(bm25_scores.keys())
        combined_scores = []
        
        for idx in all_indices:
            v_score = vector_scores.get(idx, 0)
            b_score = bm25_scores.get(idx, 0)
            combined = alpha * v_score + (1 - alpha) * b_score
            combined_scores.append((idx, combined, v_score, b_score))
        
        # Sort by combined score
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Build results
        results = []
        for idx, combined, v_score, b_score in combined_scores[:k]:
            if idx < len(self.metadata):
                results.append({
                    'text': self.metadata[idx]['text'],
                    'source': self.metadata[idx]['source'],
                    'combined_score': float(combined),
                    'vector_score': float(v_score),
                    'bm25_score': float(b_score),
                    'index': int(idx)
                })
        
        return results
    
    def get_document_stats(self) -> Dict:
        """Get statistics about indexed documents"""
        source_counts = {}
        for meta in self.metadata:
            source = meta['source']
            source_counts[source] = source_counts.get(source, 0) + 1
        
        return {
            'total_chunks': len(self.metadata),
            'total_vectors': self.index.ntotal,
            'sources': source_counts,
            'unique_sources': len(source_counts)
        }
    
    def list_documents(self) -> List[Dict]:
        """List all indexed documents with chunk counts"""
        source_counts = {}
        for meta in self.metadata:
            source = meta['source']
            source_counts[source] = source_counts.get(source, 0) + 1
        
        return [
            {'source': source, 'chunk_count': count}
            for source, count in source_counts.items()
        ]
    
    def delete_document(self, source: str) -> int:
        """
        Delete all chunks from a specific source
        
        Note: FAISS doesn't support efficient deletion, so we rebuild the index
        
        Args:
            source: Source name to delete
            
        Returns:
            Number of chunks deleted
        """
        # Find indices to keep
        indices_to_keep = [i for i, m in enumerate(self.metadata) if m['source'] != source]
        deleted_count = len(self.metadata) - len(indices_to_keep)
        
        if deleted_count == 0:
            return 0
        
        # Extract vectors to keep
        if indices_to_keep:
            # Get all vectors
            all_vectors = faiss.rev_swig_ptr(
                self.index.get_xb(), self.index.ntotal * self.dimension
            ).reshape(self.index.ntotal, self.dimension).copy()
            
            vectors_to_keep = all_vectors[indices_to_keep]
            metadata_to_keep = [self.metadata[i] for i in indices_to_keep]
            
            # Rebuild index
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(vectors_to_keep.astype('float32'))
            self.metadata = metadata_to_keep
        else:
            # Delete everything
            self.index = faiss.IndexFlatL2(self.dimension)
            self.metadata = []
        
        # Rebuild BM25 index
        self.bm25.fit([m['text'] for m in self.metadata])
        
        return deleted_count
    
    def save(self, filepath: str):
        """Save the FAISS index, metadata, and BM25 index to disk"""
        # Ensure directory exists
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        # Save FAISS index
        faiss.write_index(self.index, f"{filepath}.index")
        
        # Save metadata and BM25
        with open(f"{filepath}.meta", 'wb') as f:
            pickle.dump({
                'metadata': self.metadata,
                'bm25_corpus': self.bm25.corpus,
                'bm25_doc_lengths': self.bm25.doc_lengths,
                'bm25_doc_term_freqs': self.bm25.doc_term_freqs,
                'bm25_idf': self.bm25.idf,
                'bm25_avg_doc_length': self.bm25.avg_doc_length
            }, f)
            
    def load(self, filepath: str):
        """Load the FAISS index, metadata, and BM25 index from disk"""
        # Load FAISS index
        self.index = faiss.read_index(f"{filepath}.index")
        
        # Load metadata and BM25
        with open(f"{filepath}.meta", 'rb') as f:
            data = pickle.load(f)
            
            # Handle both old and new format
            if isinstance(data, dict) and 'metadata' in data:
                self.metadata = data['metadata']
                self.bm25.corpus = data.get('bm25_corpus', [])
                self.bm25.doc_lengths = data.get('bm25_doc_lengths', [])
                self.bm25.doc_term_freqs = data.get('bm25_doc_term_freqs', [])
                self.bm25.idf = data.get('bm25_idf', {})
                self.bm25.avg_doc_length = data.get('bm25_avg_doc_length', 0)
            else:
    
                self.metadata = data
                # Rebuild BM25 from metadata
                self.bm25.fit([m['text'] for m in self.metadata])


# Global vector database instance
vector_db = VectorDB()
