"""
EduChat RAG System Evaluation Script

Measures:
1. Retrieval Quality - Relevance of retrieved documents
2. Response Quality - Semantic coherence and accuracy
3. Hybrid Search Effectiveness - Comparison of search methods
4. Latency Metrics - Response time measurements

Usage:
    python scripts/evaluate.py
"""

import sys
import os

# Fix tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
import json
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from sentence_transformers import SentenceTransformer
from app.db import vector_db
from app.rag import answer_question, retrieve_context


# Test cases with expected information
# These questions are designed to work with general content
TEST_CASES = [
    {
        "question": "What is the capital of France?",
        "expected_keywords": ["paris", "france", "capital", "city"],
        "topic": "Geography",
        "difficulty": "easy"
    },
    {
        "question": "Who was the first president of the United States?",
        "expected_keywords": ["washington", "george", "president", "first", "united"],
        "topic": "History",
        "difficulty": "easy"
    },
    {
        "question": "What causes earthquakes?",
        "expected_keywords": ["plate", "tectonic", "fault", "earth", "movement", "seismic"],
        "topic": "Science",
        "difficulty": "medium"
    },
    {
        "question": "How does photosynthesis work?",
        "expected_keywords": ["plant", "light", "sun", "energy", "carbon", "oxygen", "chlorophyll"],
        "topic": "Biology",
        "difficulty": "medium"
    },
    {
        "question": "What is the theory of evolution?",
        "expected_keywords": ["darwin", "species", "natural", "selection", "evolution", "adapt"],
        "topic": "Biology",
        "difficulty": "medium"
    },
    {
        "question": "When did World War II end?",
        "expected_keywords": ["1945", "war", "world", "end", "germany", "japan"],
        "topic": "History",
        "difficulty": "easy"
    },
    {
        "question": "What is democracy?",
        "expected_keywords": ["government", "people", "vote", "election", "citizen", "political"],
        "topic": "Politics",
        "difficulty": "medium"
    },
    {
        "question": "How do vaccines work?",
        "expected_keywords": ["immune", "antibody", "disease", "virus", "protection", "body"],
        "topic": "Medicine",
        "difficulty": "hard"
    }
]


class RAGEvaluator:
    """Evaluator class for RAG system"""
    
    def __init__(self, model_path: str = None):
        """Initialize evaluator with embedding model"""
        if model_path and os.path.exists(model_path):
            self.model = SentenceTransformer(model_path)
        else:
            self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        
        self.results = []
        self.summary = {}
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
    
    def evaluate_retrieval_relevance(self, question: str, retrieved_docs: List[Dict]) -> float:
        """
        Evaluate how relevant retrieved documents are to the question
        
        Returns: Average similarity score (0-1)
        """
        if not retrieved_docs:
            return 0.0
        
        question_emb = self.model.encode(question)
        
        similarities = []
        for doc in retrieved_docs:
            doc_emb = self.model.encode(doc['text'])
            sim = self.cosine_similarity(question_emb, doc_emb)
            similarities.append(sim)
        
        return float(np.mean(similarities))
    
    def evaluate_keyword_coverage(self, answer: str, expected_keywords: List[str]) -> float:
        """
        Evaluate presence of expected keywords in answer
        
        Returns: Coverage ratio (0-1)
        """
        if not expected_keywords:
            return 0.0
        
        answer_lower = answer.lower()
        found = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
        return found / len(expected_keywords)
    
    def evaluate_response_coherence(self, question: str, answer: str) -> float:
        """
        Evaluate semantic coherence between question and answer
        
        Returns: Coherence score (0-1)
        """
        q_emb = self.model.encode(question)
        a_emb = self.model.encode(answer)
        return self.cosine_similarity(q_emb, a_emb)
    
    def evaluate_answer_length(self, answer: str) -> Dict:
        """Evaluate answer length characteristics"""
        words = answer.split()
        return {
            "word_count": len(words),
            "char_count": len(answer),
            "sentence_count": answer.count('.') + answer.count('!') + answer.count('?'),
            "is_adequate": 20 <= len(words) <= 500
        }
    
    def compare_search_methods(self, question: str, k: int = 3) -> Dict:
        """
        Compare vector-only vs hybrid search
        
        Returns: Comparison metrics
        """
        question_emb = self.model.encode(question)
        
        # Vector-only search
        start = time.time()
        vector_results = vector_db.search_vector_db(question_emb, k=k)
        vector_time = time.time() - start
        
        # Hybrid search
        start = time.time()
        hybrid_results = vector_db.hybrid_search(question, question_emb, k=k)
        hybrid_time = time.time() - start
        
        # Calculate relevance for both
        vector_relevance = self.evaluate_retrieval_relevance(question, vector_results)
        hybrid_relevance = self.evaluate_retrieval_relevance(question, hybrid_results)
        
        return {
            "vector_relevance": vector_relevance,
            "hybrid_relevance": hybrid_relevance,
            "relevance_improvement": hybrid_relevance - vector_relevance,
            "vector_time": vector_time,
            "hybrid_time": hybrid_time,
            "time_overhead": hybrid_time - vector_time
        }
    
    def run_single_test(self, test_case: Dict, use_hybrid: bool = True) -> Dict:
        """Run evaluation for a single test case"""
        question = test_case["question"]
        expected_keywords = test_case["expected_keywords"]
        
        # Measure latency
        start_time = time.time()
        
        # Get answer
        answer, sources = answer_question(question, use_hybrid=use_hybrid)
        
        latency = time.time() - start_time
        
        # Get retrieved documents for analysis
        context, retrieved_docs = retrieve_context(question, use_hybrid=use_hybrid)
        
        # Calculate metrics
        retrieval_relevance = self.evaluate_retrieval_relevance(question, retrieved_docs)
        keyword_coverage = self.evaluate_keyword_coverage(answer, expected_keywords)
        response_coherence = self.evaluate_response_coherence(question, answer)
        length_analysis = self.evaluate_answer_length(answer)
        
        # Compare search methods
        search_comparison = self.compare_search_methods(question)
        
        return {
            "question": question,
            "topic": test_case["topic"],
            "difficulty": test_case["difficulty"],
            "answer": answer,
            "answer_preview": answer[:300] + "..." if len(answer) > 300 else answer,
            "sources": sources,
            "metrics": {
                "retrieval_relevance": round(retrieval_relevance, 4),
                "keyword_coverage": round(keyword_coverage, 4),
                "response_coherence": round(response_coherence, 4),
                "latency_seconds": round(latency, 4)
            },
            "length_analysis": length_analysis,
            "search_comparison": {k: round(v, 4) if isinstance(v, float) else v 
                                 for k, v in search_comparison.items()},
            "success": retrieval_relevance > 0.1 and response_coherence > 0.4
        }
    
    def run_full_evaluation(self, test_cases: List[Dict] = None) -> Dict:
        """Run full evaluation suite"""
        if test_cases is None:
            test_cases = TEST_CASES
        
        print("\n" + "=" * 70)
        print("EduChat RAG System Evaluation")
        print("=" * 70)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Test cases: {len(test_cases)}")
        print("=" * 70 + "\n")
        
        self.results = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📝 Test {i}/{len(test_cases)}: {test_case['topic']}")
            print(f"   Question: {test_case['question']}")
            print(f"   Difficulty: {test_case['difficulty']}")
            
            try:
                result = self.run_single_test(test_case)
                self.results.append(result)
                
                print(f"   Success: {result['success']}")
                print(f"   Retrieval Relevance: {result['metrics']['retrieval_relevance']:.3f}")
                print(f"   Keyword Coverage: {result['metrics']['keyword_coverage']:.3f}")
                print(f"   Response Coherence: {result['metrics']['response_coherence']:.3f}")
                print(f"   Latency: {result['metrics']['latency_seconds']:.3f}s")
                print(f"   Hybrid vs Vector improvement: {result['search_comparison']['relevance_improvement']:.3f}")
                
            except Exception as e:
                print(f"   Error: {str(e)}")
                self.results.append({
                    "question": test_case["question"],
                    "topic": test_case["topic"],
                    "error": str(e),
                    "success": False
                })
        
        # Calculate summary statistics
        self.summary = self.calculate_summary()
        
        return {
            "results": self.results,
            "summary": self.summary
        }
    
    def calculate_summary(self) -> Dict:
        """Calculate summary statistics from results"""
        # Use all results with metrics, not just "successful" ones
        with_metrics = [r for r in self.results if "metrics" in r]
        successful = [r for r in self.results if r.get("success", False) and "metrics" in r]
        
        if not with_metrics:
            return {
                "total_tests": len(self.results),
                "successful_tests": 0,
                "success_rate": 0,
                "error": "No test cases with metrics"
            }
        
        # Use with_metrics for calculations (all tests that ran)
        results_to_analyze = with_metrics
        
        # Aggregate metrics
        avg_retrieval = np.mean([r["metrics"]["retrieval_relevance"] for r in results_to_analyze])
        avg_keyword = np.mean([r["metrics"]["keyword_coverage"] for r in results_to_analyze])
        avg_coherence = np.mean([r["metrics"]["response_coherence"] for r in results_to_analyze])
        avg_latency = np.mean([r["metrics"]["latency_seconds"] for r in results_to_analyze])
        
        # Hybrid search improvement
        avg_hybrid_improvement = np.mean([
            r["search_comparison"]["relevance_improvement"] 
            for r in results_to_analyze if "search_comparison" in r
        ])
        
        # By difficulty
        difficulty_breakdown = {}
        for difficulty in ["easy", "medium", "hard"]:
            diff_results = [r for r in results_to_analyze if r.get("difficulty") == difficulty]
            if diff_results:
                difficulty_breakdown[difficulty] = {
                    "count": len(diff_results),
                    "avg_coherence": round(np.mean([r["metrics"]["response_coherence"] for r in diff_results]), 3),
                    "avg_latency": round(np.mean([r["metrics"]["latency_seconds"] for r in diff_results]), 3)
                }
        
        # Overall quality score (weighted average)
        overall_score = (avg_retrieval * 0.25 + avg_keyword * 0.25 + avg_coherence * 0.5)
        
        return {
            "total_tests": len(self.results),
            "successful_tests": len(successful),
            "tests_with_metrics": len(with_metrics),
            "success_rate": round(len(successful) / len(self.results), 3) if self.results else 0,
            "avg_retrieval_relevance": round(avg_retrieval, 3),
            "avg_keyword_coverage": round(avg_keyword, 3),
            "avg_response_coherence": round(avg_coherence, 3),
            "avg_latency_seconds": round(avg_latency, 3),
            "avg_hybrid_improvement": round(avg_hybrid_improvement, 3),
            "overall_quality_score": round(overall_score, 3),
            "difficulty_breakdown": difficulty_breakdown,
            "evaluation_timestamp": datetime.now().isoformat()
        }
    
    def print_summary(self):
        """Print evaluation summary"""
        if not self.summary:
            print("No summary available. Run evaluation first.")
            return
        
        if "error" in self.summary and "successful_tests" not in self.summary:
            print(f"\n Evaluation Error: {self.summary.get('error')}")
            return
        
        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)
        
        total = self.summary.get('total_tests', 0)
        successful = self.summary.get('successful_tests', 0)
        rate = self.summary.get('success_rate', 0)
        
        print(f"\n Tests Run: {total}")
        print(f"Successful: {successful}/{total} ({rate:.1%})")
        
        print(f"\n Average Metrics:")
        print(f"   Retrieval Relevance:  {self.summary.get('avg_retrieval_relevance', 0):.3f}")
        print(f"   Keyword Coverage:     {self.summary.get('avg_keyword_coverage', 0):.3f}")
        print(f"   Response Coherence:   {self.summary.get('avg_response_coherence', 0):.3f}")
        print(f"   Average Latency:      {self.summary.get('avg_latency_seconds', 0):.3f}s")
        
        hybrid_imp = self.summary.get('avg_hybrid_improvement', 0)
        print(f"\n Hybrid Search Improvement: {hybrid_imp:+.3f}")
        if hybrid_imp < 0:
            print("   (Negative value is normal for datasets without keyword-heavy content)")
        
        print(f"\n Performance by Difficulty:")
        for diff, stats in self.summary.get('difficulty_breakdown', {}).items():
            print(f"   {diff.capitalize()}: {stats['count']} tests, coherence={stats['avg_coherence']:.3f}, latency={stats['avg_latency']:.3f}s")
        
        score = self.summary.get('overall_quality_score', 0)
        print(f"\n OVERALL QUALITY SCORE: {score:.3f}")
        
        # Quality assessment
        if score >= 0.7:
            print("   Assessment: EXCELLENT")
        elif score >= 0.5:
            print("   Assessment: GOOD")
        elif score >= 0.3:
            print("   Assessment: FAIR")
        else:
            print("   Assessment: NEEDS IMPROVEMENT")
            print("   (Low scores may indicate dataset mismatch with test questions)")
        
        print("\n" + "=" * 70)
    
    def save_results(self, filepath: str = None):
        """Save evaluation results to JSON file"""
        if filepath is None:
            filepath = os.path.join(os.path.dirname(__file__), '..', 'data', 'evaluation_results.json')
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                "results": self.results,
                "summary": self.summary
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n Results saved to: {filepath}")


def main():
    """Main function to run evaluation"""
    # Load vector database
    db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'vector_db')
    
    if not os.path.exists(f"{db_path}.index"):
        print("Vector database not found!")
        print("   Run 'python scripts/init_squad.py' first.")
        return
    
    vector_db.load(db_path)
    stats = vector_db.get_document_stats()
    print(f"Loaded vector database: {stats['total_chunks']} chunks")
    
    # Initialize evaluator
    model_path = os.path.join(os.path.dirname(__file__), '..', 'Models', 'paraphrase-MiniLM-L6-v2')
    evaluator = RAGEvaluator(model_path)
    
    # Run evaluation
    evaluator.run_full_evaluation()
    
    # Print and save results
    evaluator.print_summary()
    evaluator.save_results()


if __name__ == "__main__":
    main()
