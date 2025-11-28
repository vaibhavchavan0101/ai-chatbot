"""
Order Tool - Interface for OrderStatusAgent
"""
from typing import Dict, Any
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.transactional_agents import get_order_agent


def order_tool(query: str, user_context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Order tool for handling order status and tracking queries
    
    Args:
        query: User's query about orders
        user_context: Additional context
        
    Returns:
        Dictionary with order information or error
    """
    try:
        # Get order agent instance
        order_agent = get_order_agent()
        
        # Process the query
        result = order_agent.process_query(query, user_context or {})
        
        return result
        
    except Exception as e:
        return {
            "status": "error",
            "error": f"Order tool error: {str(e)}",
            "query": query
        }


# Tool metadata
TOOL_METADATA = {
    "name": "order_tool",
    "description": "Handles order status, tracking, and order-related queries", 
    "parameters": {
        "query": {
            "type": "string",
            "description": "Order-related query from user",
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
        "description": "Order information or status"
    }
}