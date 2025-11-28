#!/usr/bin/env python3
"""
Quick Milvus Setup with Sample E-commerce Data
Creates a collection and inserts sample documents for testing
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('src')

from typing import List, Dict, Any
import json
from datetime import datetime

# Import required modules
try:
    from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, connections, utility
    from sentence_transformers import SentenceTransformer
    MILVUS_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    MILVUS_AVAILABLE = False

# Try Milvus Lite
try:
    from milvus import default_server
    MILVUS_LITE_AVAILABLE = True
except ImportError:
    MILVUS_LITE_AVAILABLE = False


def start_milvus_lite():
    """Start Milvus Lite server"""
    if MILVUS_LITE_AVAILABLE:
        try:
            from milvus import default_server
            default_server.start()
            print("✅ Milvus Lite server started")
            return True
        except Exception as e:
            print(f"❌ Failed to start Milvus Lite: {e}")
            return False
    return False


def connect_to_milvus():
    """Connect to Milvus"""
    try:
        connections.connect(
            alias="default",
            host="127.0.0.1",
            port="19530"
        )
        print("✅ Connected to Milvus")
        return True
    except Exception as e:
        print(f"❌ Failed to connect to Milvus: {e}")
        return False


def create_collection():
    """Create the ecommerce_docs collection"""
    collection_name = "ecommerce_docs"
    
    # Drop existing collection if it exists
    if utility.has_collection(collection_name):
        collection = Collection(collection_name)
        collection.drop()
        print(f"🗑️ Dropped existing collection: {collection_name}")
    
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
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
    
    collection.create_index("embedding", index_params)
    print("✅ Created vector index")
    
    return collection


def get_sample_documents():
    """Get sample e-commerce documents"""
    documents = [
        {
            "text": "Our return policy allows customers to return items within 30 days of purchase. All items must be in original condition with tags attached. Refunds will be processed within 5-7 business days after we receive the returned item. For defective items, we offer free return shipping.",
            "metadata": {
                "filename": "return_policy.pdf",
                "topic": "returns",
                "source": "policy_documents",
                "created_at": "2024-11-28"
            }
        },
        {
            "text": "We offer several shipping options: Standard shipping (5-7 business days) for $5.99, Express shipping (2-3 business days) for $12.99, and Overnight shipping (1 business day) for $24.99. Free standard shipping is available on orders over $50. International shipping is available to select countries.",
            "metadata": {
                "filename": "shipping_guide.pdf",
                "topic": "shipping",
                "source": "customer_service",
                "created_at": "2024-11-28"
            }
        },
        {
            "text": "Customer service is available Monday through Friday from 9 AM to 6 PM EST. You can contact us via email at support@ecommerce.com, phone at 1-800-SHOP-NOW, or live chat on our website. We respond to emails within 24 hours and phone calls are typically answered within 5 minutes.",
            "metadata": {
                "filename": "contact_info.pdf",
                "topic": "support",
                "source": "customer_service",
                "created_at": "2024-11-28"
            }
        },
        {
            "text": "Our size guide helps you find the perfect fit. For clothing: XS (0-2), S (4-6), M (8-10), L (12-14), XL (16-18). For shoes: we offer sizes 5-12 in both standard and wide widths. Measurements should be taken without clothing for the most accurate sizing.",
            "metadata": {
                "filename": "size_guide.pdf",
                "topic": "sizing",
                "source": "product_info",
                "created_at": "2024-11-28"
            }
        },
        {
            "text": "We offer a 1-year warranty on all electronic items and a 90-day warranty on clothing and accessories. Warranty covers manufacturing defects but does not cover damage from normal wear and tear or misuse. To make a warranty claim, contact customer service with your order number and photos of the defect.",
            "metadata": {
                "filename": "warranty_info.pdf",
                "topic": "warranty",
                "source": "policy_documents",
                "created_at": "2024-11-28"
            }
        },
        {
            "text": "Payment methods accepted include all major credit cards (Visa, MasterCard, American Express, Discover), PayPal, Apple Pay, Google Pay, and buy-now-pay-later options through Klarna and Afterpay. All transactions are secured with 256-bit SSL encryption for your protection.",
            "metadata": {
                "filename": "payment_methods.pdf",
                "topic": "payment",
                "source": "billing",
                "created_at": "2024-11-28"
            }
        },
        {
            "text": "Frequently Asked Questions: Q: Can I track my order? A: Yes, you'll receive a tracking number via email once your order ships. Q: Do you offer gift wrapping? A: Yes, gift wrapping is available for $3.99. Q: Can I change my shipping address? A: Yes, but only before the order ships.",
            "metadata": {
                "filename": "faq.pdf",
                "topic": "faq",
                "source": "customer_service",
                "created_at": "2024-11-28"
            }
        }
    ]
    return documents


def generate_embeddings(texts: List[str]):
    """Generate embeddings for text chunks"""
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(texts)
        print(f"✅ Generated {len(embeddings)} embeddings")
        return embeddings.tolist()
    except Exception as e:
        print(f"❌ Error generating embeddings: {e}")
        return None


def insert_sample_data(collection):
    """Insert sample documents into collection"""
    documents = get_sample_documents()
    
    # Extract texts for embedding generation
    texts = [doc["text"] for doc in documents]
    
    # Generate embeddings
    embeddings = generate_embeddings(texts)
    if embeddings is None:
        print("❌ Failed to generate embeddings")
        return False
    
    # Prepare data for insertion
    data = {
        "embedding": embeddings,
        "text": texts,
        "metadata": [doc["metadata"] for doc in documents]
    }
    
    # Insert data
    try:
        insert_result = collection.insert(data)
        collection.flush()
        print(f"✅ Inserted {len(texts)} documents")
        print(f"   Insert IDs: {insert_result.primary_keys[:3]}...")
        return True
    except Exception as e:
        print(f"❌ Error inserting data: {e}")
        return False


def setup_milvus_database():
    """Complete Milvus setup with sample data"""
    print("🚀 Setting up Milvus database for E-commerce Orchestrator")
    print("=" * 60)
    
    # Step 1: Start Milvus Lite
    if not start_milvus_lite():
        print("❌ Cannot proceed without Milvus")
        return False
    
    # Step 2: Connect to Milvus
    if not connect_to_milvus():
        print("❌ Cannot connect to Milvus")
        return False
    
    # Step 3: Create collection
    try:
        collection = create_collection()
    except Exception as e:
        print(f"❌ Error creating collection: {e}")
        return False
    
    # Step 4: Insert sample data
    if not insert_sample_data(collection):
        print("❌ Failed to insert sample data")
        return False
    
    # Step 5: Load collection for search
    try:
        collection.load()
        print("✅ Collection loaded and ready for search")
    except Exception as e:
        print(f"❌ Error loading collection: {e}")
        return False
    
    # Step 6: Test search
    try:
        # Test search
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode(["What is your return policy?"]).tolist()
        
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }
        
        results = collection.search(
            data=query_embedding,
            anns_field="embedding",
            param=search_params,
            limit=3,
            output_fields=["text", "metadata"]
        )
        
        if results and len(results[0]) > 0:
            print("✅ Search test successful")
            print(f"   Found: {results[0][0].entity.get('text', '')[:50]}...")
        else:
            print("⚠️ Search test returned no results")
        
    except Exception as e:
        print(f"⚠️ Search test failed: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 Milvus setup complete!")
    print("✅ Collection: ecommerce_docs")
    print("✅ Documents: 7 sample documents inserted")
    print("✅ Vector search: Ready")
    print("\n💡 Your chatbot can now use real Milvus vector database!")
    print("   Test with: 'What is your return policy?'")
    
    return True


if __name__ == "__main__":
    if not MILVUS_AVAILABLE:
        print("❌ PyMilvus not available. Install with: pip install pymilvus")
        exit(1)
    
    success = setup_milvus_database()
    if success:
        print("\n🚀 Ready to run the chatbot with Milvus!")
        print("   Command: streamlit run streamlit_app.py")
    else:
        print("\n❌ Setup failed. Check the errors above.")