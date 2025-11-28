#!/usr/bin/env python3
"""
Load data into cloud Milvus database
This script connects to your cloud Milvus and loads the e-commerce documents
"""
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.append('src')

from typing import List, Dict, Any
import json

# Import required modules
try:
    from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, connections, utility
    from sentence_transformers import SentenceTransformer
    MILVUS_AVAILABLE = True
    print("✅ PyMilvus available")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Install with: pip install pymilvus")
    MILVUS_AVAILABLE = False
    sys.exit(1)


def connect_to_cloud_milvus():
    """Connect to cloud Milvus using environment credentials"""
    try:
        # Get credentials from environment
        uri = os.getenv("MILVUS_URI")
        token = os.getenv("MILVUS_TOKEN")
        
        if uri and token:
            print(f"🔗 Connecting to cloud Milvus: {uri}")
            connections.connect(
                alias="default",
                uri=uri,
                token=token
            )
            print("✅ Connected to cloud Milvus successfully!")
            return True
        else:
            # Fallback to host/port credentials
            host = os.getenv("MILVUS_HOST", "localhost")
            port = int(os.getenv("MILVUS_PORT", "19530"))
            user = os.getenv("MILVUS_USER")
            password = os.getenv("MILVUS_PASSWORD")
            secure = os.getenv("MILVUS_SECURE", "false").lower() == "true"
            
            connection_params = {
                "alias": "default",
                "host": host,
                "port": str(port)
            }
            
            if user and password:
                connection_params["user"] = user
                connection_params["password"] = password
                
            if secure:
                connection_params["secure"] = True
                
            print(f"🔗 Connecting to Milvus: {host}:{port}")
            connections.connect(**connection_params)
            print("✅ Connected to Milvus successfully!")
            return True
            
    except Exception as e:
        print(f"❌ Failed to connect to Milvus: {e}")
        return False


def create_collection():
    """Create or recreate the collection"""
    collection_name = os.getenv("MILVUS_COLLECTION_NAME", "ecom123")
    
    # Drop existing collection if it exists
    if utility.has_collection(collection_name):
        print(f"🗑️ Collection '{collection_name}' exists, dropping it...")
        collection = Collection(collection_name)
        collection.drop()
        print(f"✅ Dropped existing collection")
    
    # Define schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=10000),
        FieldSchema(name="metadata", dtype=DataType.JSON)
    ]
    
    schema = CollectionSchema(fields, "E-commerce documents collection")
    collection = Collection(collection_name, schema)
    print(f"✅ Created collection: {collection_name}")
    
    # Create index for vector search
    index_params = {
        "metric_type": "COSINE",
        "index_type": "AUTOINDEX",  # Use AUTOINDEX for cloud Milvus
        "params": {}
    }
    
    collection.create_index("embedding", index_params)
    print("✅ Created vector index")
    
    return collection


def load_documents_from_file():
    """Load documents from the vector database file"""
    try:
        with open('data/vector_database.json', 'r') as f:
            data = json.load(f)
        print(f"📄 Loaded {len(data)} documents from file")
        return data
    except Exception as e:
        print(f"❌ Error loading documents: {e}")
        return []


def generate_embeddings(texts: List[str]):
    """Generate embeddings for text chunks"""
    try:
        print("🔄 Loading embedding model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("🔄 Generating embeddings...")
        embeddings = model.encode(texts, show_progress_bar=True)
        print(f"✅ Generated {len(embeddings)} embeddings")
        return embeddings.tolist()
    except Exception as e:
        print(f"❌ Error generating embeddings: {e}")
        return None


def insert_data_to_collection(collection, documents):
    """Insert documents into the collection"""
    if not documents:
        print("❌ No documents to insert")
        return False
    
    # Extract texts and metadata
    texts = [doc.get("text", "") for doc in documents]
    metadata = [doc.get("metadata", {}) for doc in documents]
    
    # Generate embeddings
    embeddings = generate_embeddings(texts)
    if embeddings is None:
        return False
    
    # Prepare data for insertion
    data = {
        "embedding": embeddings,
        "text": texts,
        "metadata": metadata
    }
    
    # Insert data in batches
    try:
        print("🔄 Inserting documents into collection...")
        insert_result = collection.insert(data)
        print(f"✅ Inserted {len(texts)} documents")
        print(f"   Primary keys: {insert_result.primary_keys[:3]}...")
        
        # Flush to ensure data is written
        print("🔄 Flushing data...")
        collection.flush()
        print("✅ Data flushed to storage")
        
        return True
    except Exception as e:
        print(f"❌ Error inserting data: {e}")
        return False


def test_search(collection):
    """Test vector search functionality"""
    try:
        print("🔄 Loading collection for search...")
        collection.load()
        print("✅ Collection loaded")
        
        # Test search
        print("🔍 Testing search functionality...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode(["What is your return policy?"]).tolist()
        
        search_params = {
            "metric_type": "COSINE",
            "params": {"ef": 64}  # For AUTOINDEX
        }
        
        results = collection.search(
            data=query_embedding,
            anns_field="embedding",
            param=search_params,
            limit=3,
            output_fields=["text", "metadata"]
        )
        
        if results and len(results[0]) > 0:
            print("✅ Search test successful!")
            for i, hit in enumerate(results[0][:2]):
                text = hit.entity.get('text', '')[:100]
                score = hit.score
                print(f"   {i+1}. Score: {score:.3f} - {text}...")
        else:
            print("⚠️ Search test returned no results")
        
        return True
    except Exception as e:
        print(f"❌ Search test failed: {e}")
        return False


def main():
    """Main function to load data into cloud Milvus"""
    print("🚀 Loading E-commerce Data into Cloud Milvus")
    print("=" * 60)
    
    # Step 1: Connect to cloud Milvus
    if not connect_to_cloud_milvus():
        print("❌ Cannot proceed without Milvus connection")
        return False
    
    # Step 2: Create collection
    try:
        collection = create_collection()
    except Exception as e:
        print(f"❌ Error creating collection: {e}")
        return False
    
    # Step 3: Load documents from file
    documents = load_documents_from_file()
    if not documents:
        print("❌ No documents to load")
        return False
    
    # Step 4: Insert documents
    if not insert_data_to_collection(collection, documents):
        return False
    
    # Step 5: Test search
    if not test_search(collection):
        return False
    
    print("\\n" + "=" * 60)
    print("🎉 Data loading complete!")
    print(f"✅ Collection: {collection.name}")
    print(f"✅ Documents: {len(documents)} documents loaded")
    print("✅ Vector search: Ready")
    print("\\n💡 Your chatbot will now use the cloud database!")
    print("   The system will automatically connect to your cloud Milvus.")
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        print("\\n❌ Data loading failed. Check errors above.")
        sys.exit(1)