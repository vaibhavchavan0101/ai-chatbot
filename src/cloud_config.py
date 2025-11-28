"""
Cloud database configuration utilities
"""
import os
from typing import Optional, Dict, Any

def load_cloud_config() -> Dict[str, Any]:
    """Load cloud database configuration from environment variables"""
    return {
        'milvus_host': os.getenv('MILVUS_HOST', 'localhost'),
        'milvus_port': int(os.getenv('MILVUS_PORT', '19530')),
        'milvus_user': os.getenv('MILVUS_USER'),
        'milvus_password': os.getenv('MILVUS_PASSWORD'),
        'milvus_token': os.getenv('MILVUS_TOKEN'),
        'milvus_secure': os.getenv('MILVUS_SECURE', 'false').lower() == 'true',
        'collection_name': os.getenv('MILVUS_COLLECTION_NAME', 'ecommerce_docs')
    }

def create_rag_agent_with_cloud_config():
    """Create RAG agent with cloud configuration from environment"""
    from .rag_agent import RAGRetrieverAgent
    config = load_cloud_config()
    
    return RAGRetrieverAgent(
        collection_name=config['collection_name'],
        milvus_host=config['milvus_host'],
        milvus_port=config['milvus_port'],
        milvus_user=config['milvus_user'],
        milvus_password=config['milvus_password'],
        milvus_token=config['milvus_token'],
        milvus_secure=config['milvus_secure']
    )