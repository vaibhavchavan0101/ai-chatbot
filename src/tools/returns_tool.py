"""
Returns Tool - Interface for ReturnAgent
"""
from typing import Dict, Any
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.transactional_agents import get_return_agent


def returns_tool(query: str, user_context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Returns tool for handling return policy, status, and return requests
    
    Args:
        query: User's query about returns
        user_context: Additional context
        
    Returns:
        Dictionary with return information or error
    """
    try:
        # Get return agent instance
        return_agent = get_return_agent()
        
        # Process the query
        result = return_agent.process_query(query, user_context or {})
        
        return result
        
    except Exception as e:
        return {
            "status": "error",
            "error": f"Returns tool error: {str(e)}",
            "query": query
        }


# Tool metadata
TOOL_METADATA = {
    "name": "returns_tool",
    "description": "Handles return policy, return status, and return initiation",
    "parameters": {
        "query": {
            "type": "string", 
            "description": "Return-related query from user",
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
        "description": "Return information, policy, or status"
    }
}