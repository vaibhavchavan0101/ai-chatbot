"""
E-commerce RAG Tool - Interface for the RAG Retriever Agent
"""
from typing import Dict, Any
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.rag_agent import get_rag_agent

# Optional: Import cloud config for automatic cloud database usage
try:
    from cloud_config import create_rag_agent_with_cloud_config
    CLOUD_CONFIG_AVAILABLE = True
except ImportError:
    CLOUD_CONFIG_AVAILABLE = False


def ecom_rag_tool(query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    E-commerce RAG tool that handles static knowledge queries
    
    Args:
        query: User's question/query
        context: Additional context (optional)
    
    Returns:
        Dictionary with status, answer, and sources
    """
    try:
        # Get RAG agent instance
        rag_agent = get_rag_agent()
        
        # Process the query
        result = rag_agent.process_query(query, context or {})
        
        return result
        
    except Exception as e:
        return {
            "status": "error",
            "error": f"RAG tool error: {str(e)}",
            "query": query
        }


# Tool metadata for registration
TOOL_METADATA = {
    "name": "ecom_rag_tool",
    "description": "Handles static knowledge queries using RAG with Milvus vector search",
    "parameters": {
        "query": {
            "type": "string",
            "description": "The user's question or query",
            "required": True
        },
        "context": {
            "type": "object", 
            "description": "Additional context for the query",
            "required": False
        }
    },
    "returns": {
        "type": "object",
        "description": "Response with status, answer, and source documents"
    }
}