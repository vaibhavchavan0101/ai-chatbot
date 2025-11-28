#!/usr/bin/env python3
"""
Simple data loader for cloud Milvus collection
"""
import os
import sys
import json
from dotenv import load_dotenv

# Load environment
load_dotenv()
sys.path.append('src')

def load_data_to_cloud():
    """Load data from local file to cloud collection"""
    try:
        from pymilvus import Collection, connections
        from sentence_transformers import SentenceTransformer
        
        print("🔗 Connecting to cloud Milvus...")
        connections.connect(
            'default', 
            uri=os.getenv('MILVUS_URI'),
            token=os.getenv('MILVUS_TOKEN')
        )
        
        collection = Collection('ecom')
        print(f"📊 Current collection size: {collection.num_entities}")
        
        # Load local data
        print("📄 Loading local data...")
        with open('data/vector_database.json', 'r') as f:
            docs = json.load(f)
        print(f"Found {len(docs)} documents")
        
        # Clear existing data if it looks corrupted
        if collection.num_entities > 0:
            print("🗑️ Clearing existing data...")
            collection.delete(expr="id >= 0")
            collection.flush()
            print("✅ Collection cleared")
        
        # Prepare data
        print("🔄 Generating embeddings...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        texts = [doc['text'] for doc in docs]
        embeddings = model.encode(texts).tolist()
        
        data = {
            'embedding': embeddings,
            'text': texts,
            'metadata': [doc['metadata'] for doc in docs]
        }
        
        print("💾 Inserting data...")
        collection.insert(data)
        collection.flush()
        
        print(f"✅ Inserted {len(texts)} documents")
        print(f"📊 Collection now has {collection.num_entities} documents")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = load_data_to_cloud()
    if success:
        print("🎉 Data loading complete!")
    else:
        print("❌ Data loading failed")