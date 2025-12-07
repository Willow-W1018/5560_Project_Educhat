import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import json
from sentence_transformers import SentenceTransformer
from app.db import vector_db

# Load the sentence transformer model
model = SentenceTransformer('Models/paraphrase-MiniLM-L6-v2')

def load_squad_data():
    """Load SQuAD 2.0 dataset from local file"""
    squad_file_path = os.path.join(os.path.dirname(__file__), '..', 'app', 'train-v2.0.json')
    
    with open(squad_file_path, 'r', encoding='utf-8') as f:
        squad_data = json.load(f)
    
    return squad_data

def init_squad():
    """Initialize vector database with SQuAD 2.0 dataset"""
    # Get SQuAD data
    squad_data = load_squad_data()
    
    chunks = []
    # Process the dataset
    for article in squad_data['data']:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            # Add context as a chunk
            chunks.append(context)
    
    # Limit chunks to avoid overloading memory (can be adjusted)
    chunks = chunks[:1000]  # Process only first 1000 chunks
    
    # Generate embeddings
    embeddings = model.encode(chunks)
    
    # Add to vector database
    vector_db.add_to_vector_db(chunks, embeddings, "SQuAD 2.0")
    
    # Save the vector database
    vector_db.save("data/vector_db")
    
    print(f"Added {len(chunks)} chunks from SQuAD 2.0 to vector database")

if __name__ == "__main__":
    init_squad()