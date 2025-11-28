#!/usr/bin/env python3
"""
Insert data into Milvus cloud database using REST API
"""
import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def insert_data_via_api():
    """Insert data using Milvus REST API"""
    
    # API endpoint
    # url = "https://in03-a39eed178c34f1b.serverless.aws-eu-central-1.cloud.zilliz.com/v2/vectordb/entities/insert"
    url = os.getenv("MILVUS_API_URL", "https://in03-a39eed178c34f1b.serverless.aws-eu-central-1.cloud.zilliz.com/v2/vectordb/entities/insert")
    # Headers
    headers = {
        "Authorization": f"Bearer {os.getenv('MILVUS_TOKEN')}",
        "Content-Type": "application/json"
    }
    
    # Sample data to insert
    sample_data = {
        "collectionName": "ecom",
        "data": [
            {
                "embedding": [0.1] * 384,  # 384-dimensional embedding
                "text": "Our return policy allows customers to return items within 30 days of purchase. All items must be in original condition with tags attached. Refunds will be processed within 5-7 business days after we receive the returned item.",
                "metadata": {
                    "filename": "return_policy.pdf",
                    "topic": "returns",
                    "source": "policy_documents",
                    "created_at": "2024-11-28"
                }
            },
            {
                "embedding": [0.2] * 384,  # 384-dimensional embedding
                "text": "We offer several shipping options: Standard shipping (5-7 business days) for $5.99, Express shipping (2-3 business days) for $12.99, and Overnight shipping (1 business day) for $24.99. Free standard shipping is available on orders over $50.",
                "metadata": {
                    "filename": "shipping_guide.pdf",
                    "topic": "shipping",
                    "source": "customer_service",
                    "created_at": "2024-11-28"
                }
            },
            {
                "embedding": [0.3] * 384,  # 384-dimensional embedding
                "text": "Return Shipping Costs: For defective, damaged, or incorrect items shipped to you, we provide a free return shipping label. For returns due to personal preference, wrong size selection, or change of mind, customers are responsible for return shipping costs, which are $5.99 via ground service.",
                "metadata": {
                    "filename": "return_shipping_policy.pdf",
                    "topic": "return_shipping",
                    "source": "policy_documents",
                    "created_at": "2024-11-28"
                }
            }
        ]
    }
    
    print(f"🔗 Making POST request to: {url}")
    print(f"📊 Inserting {len(sample_data['data'])} documents")
    
    try:
        # Make the POST request
        response = requests.post(url, headers=headers, json=sample_data, timeout=30)
        
        print(f"📡 Response Status: {response.status_code}")
        print(f"📄 Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success! Response: {json.dumps(result, indent=2)}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("⏱️ Request timed out")
    except requests.exceptions.RequestException as e:
        print(f"❌ Request error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def insert_from_local_data():
    """Insert data from local vector database file"""
    
    url = "https://in03-a39eed178c34f1b.serverless.aws-eu-central-1.cloud.zilliz.com/v2/vectordb/entities/insert"
    
    headers = {
        "Authorization": f"Bearer {os.getenv('MILVUS_TOKEN')}",
        "Content-Type": "application/json"
    }
    
    try:
        # Load local data
        print("📄 Loading local vector database...")
        with open('data/vector_database.json', 'r') as f:
            docs = json.load(f)
        
        # Prepare data for API
        api_data = {
            "collectionName": "ecom",
            "data": []
        }
        
        for doc in docs:
            if 'embedding' in doc and doc['embedding']:
                api_data['data'].append({
                    "embedding": doc['embedding'],
                    "text": doc['text'],
                    "metadata": doc['metadata']
                })
        
        if not api_data['data']:
            print("❌ No valid data found in local file")
            return
            
        print(f"🔗 Making POST request with {len(api_data['data'])} documents")
        
        # Make the request
        response = requests.post(url, headers=headers, json=api_data, timeout=60)
        
        print(f"📡 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success! Inserted {len(api_data['data'])} documents")
            print(f"Response: {json.dumps(result, indent=2)}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🚀 Milvus Cloud API Data Insertion")
    print("=" * 50)
    
    # Check if token is available
    token = os.getenv('MILVUS_TOKEN')
    if not token:
        print("❌ MILVUS_TOKEN not found in environment")
        exit(1)
    
    print("Choose insertion method:")
    print("1. Insert sample data")
    print("2. Insert from local vector database")
    
    try:
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "1":
            insert_data_via_api()
        elif choice == "2":
            insert_from_local_data()
        else:
            print("Invalid choice")
    except KeyboardInterrupt:
        print("\n🛑 Operation cancelled")