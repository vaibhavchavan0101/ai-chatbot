"""
Inventory Tool - Interface for ProductAgent
"""
from typing import Dict, Any
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.transactional_agents import get_product_agent


def inventory_tool(query: str, user_context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Inventory tool for handling product availability, search, and details
    
    Args:
        query: User's query about products/inventory
        user_context: Additional context
        
    Returns:
        Dictionary with product information or error
    """
    try:
        # Get product agent instance
        product_agent = get_product_agent()
        
        # Process the query
        result = product_agent.process_query(query, user_context or {})
        
        return result
        
    except Exception as e:
        return {
            "status": "error", 
            "error": f"Inventory tool error: {str(e)}",
            "query": query
        }


# Tool metadata
TOOL_METADATA = {
    "name": "inventory_tool",
    "description": "Handles product availability, search, and inventory queries",
    "parameters": {
        "query": {
            "type": "string",
            "description": "Product or inventory-related query from user", 
            "required": True
        },
        "user_context": {
            "type": "object",
            "description": "Additional context about the user",
            "required": False
        }
    },
    "returns": {
        "type": "object",
        "description": "Product information, availability, or search results"
    }
}