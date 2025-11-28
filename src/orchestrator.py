"""
Main Orchestrator that manages multiple agents and tools using Chat-Completion style prompting
Implements strict intent-based routing as per specifications
"""
import json
import re
import os
import sys
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llm_factory import LLMProcessorFactory, get_default_processor


class IntentType(Enum):
    """Intent types for routing"""
    RAG_QUERY = "rag_query"
    TRANSACTIONAL = "transactional"
    UNCLEAR = "unclear"


@dataclass
class RoutingResult:
    """Result of intent routing"""
    intent: IntentType
    confidence: float
    tool_name: Optional[str] = None
    reasoning: Optional[str] = None


class Orchestrator:
    """
    Main orchestrator that routes queries to appropriate agents/tools
    Follows strict intent-based routing rules
    """
    
    def __init__(self):
        self.llm_processor = get_default_processor()
        
        # Define routing keywords as per specifications
        self.rag_keywords = {
            'policy', 'rules', 'faq', 'how to', 'guide', 'manual', 
            'terms', 'details', 'brochure', 'information', 'help',
            'what is', 'explain', 'describe', 'documentation', 'who pays',
            'what happens', 'how do i', 'can i cancel', 'return shipping'
        }
        
        # Keep only very specific operational keywords - let RAG handle most queries
        self.transactional_keywords = {
            'track order', 'order status', 'my order #', 'order number',
            'tracking number', 'order #', 'return status for order',
            'refund status for order', 'cancel order #'
        }
        
        # Tool mappings
        self.tool_mappings = {
            'order': 'order_tool',
            'return': 'returns_tool', 
            'product': 'inventory_tool',
            'inventory': 'inventory_tool',
            'rag': 'ecom_rag_tool'
        }
    
    def route_query(self, query: str) -> RoutingResult:
        """
        Route query using strict intent-based heuristic
        
        Rules:
        1. IF query contains RAG keywords → RAG tool
        2. ELSE IF query contains transactional keywords → transactional tool
        3. ELSE → ask for clarification
        """
        query_lower = query.lower()
        
        # Check for RAG keywords first (higher priority)
        rag_matches = sum(1 for keyword in self.rag_keywords if keyword in query_lower)
        transactional_matches = sum(1 for keyword in self.transactional_keywords if keyword in query_lower)
        
        if rag_matches > 0:
            return RoutingResult(
                intent=IntentType.RAG_QUERY,
                confidence=min(rag_matches / len(self.rag_keywords) * 5, 1.0),
                tool_name='ecom_rag_tool',
                reasoning=f"Detected {rag_matches} RAG-related keywords"
            )
        
        elif transactional_matches > 0:
            # Determine specific transactional tool
            tool_name = self._determine_transactional_tool(query_lower)
            return RoutingResult(
                intent=IntentType.TRANSACTIONAL,
                confidence=min(transactional_matches / len(self.transactional_keywords) * 5, 1.0),
                tool_name=tool_name,
                reasoning=f"Detected {transactional_matches} transactional keywords"
            )
        
        else:
            # Default to RAG for any query - let LLM handle it intelligently
            return RoutingResult(
                intent=IntentType.RAG_QUERY,
                confidence=0.5,  # Medium confidence for general queries
                tool_name='ecom_rag_tool',
                reasoning="Defaulting to RAG for open-ended query - LLM will handle contextually"
            )
    
    def _determine_transactional_tool(self, query: str) -> str:
        """Determine which transactional tool to use"""
        query_lower = query.lower()
        
        # Check for product/inventory related queries first
        if any(word in query_lower for word in ['product', 'inventory', 'stock', 'availability', 'available', 'prod-', 'search', 'find', 'item']):
            return 'inventory_tool'
        elif any(word in query_lower for word in ['order', 'track', 'status', 'check']):
            return 'order_tool'
        elif any(word in query_lower for word in ['return', 'refund', 'exchange']):
            return 'returns_tool'
        else:
            return 'inventory_tool'  # Default to inventory for unclear product queries
    
    def process_query(self, query: str, user_context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Main entry point for processing user queries
        Returns either tool call or natural language response
        """
        routing_result = self.route_query(query)
        
        if routing_result.intent == IntentType.RAG_QUERY:
            return {
                "tool": routing_result.tool_name,
                "arguments": {
                    "query": query,
                    "context": user_context or {}
                }
            }
        
        elif routing_result.intent == IntentType.TRANSACTIONAL:
            return {
                "tool": routing_result.tool_name, 
                "arguments": {
                    "query": query,
                    "user_context": user_context or {}
                }
            }
        
        else:
            # Route any unclear queries to RAG - let LLM handle contextually
            return {
                "tool": "ecom_rag_tool",
                "arguments": {
                    "query": query,
                    "context": user_context or {}
                }
            }
    
    def _generate_clarification(self, query: str) -> str:
        """Generate clarification request when intent is unclear"""
        clarification_prompts = [
            "I need more information to help you properly. Are you looking for:",
            "• Information about our policies, guides, or FAQ?",
            "• Help with an order, return, or product availability?",
            "",
            "Please be more specific about what you'd like to do."
        ]
        return "\n".join(clarification_prompts)
    
    def validate_tool_response(self, tool_name: str, response: Any) -> bool:
        """Validate tool response format"""
        try:
            if isinstance(response, dict) and "status" in response:
                return True
            return False
        except:
            return False
    
    def format_response(self, tool_response: Dict[str, Any], original_query: str) -> str:
        """Format tool response for user consumption"""
        if not tool_response:
            return "I apologize, but I couldn't process your request. Please try again."
        
        # Extract relevant information from tool response
        if "error" in tool_response:
            return f"I encountered an error: {tool_response['error']}"
        
        if "data" in tool_response:
            return self._format_data_response(tool_response["data"], original_query)
        
        return str(tool_response.get("message", "Request processed successfully."))
    
    def _format_data_response(self, data: Any, query: str) -> str:
        """Format data response based on query type"""
        if isinstance(data, list):
            if len(data) == 0:
                return "No results found for your query."
            elif len(data) == 1:
                return f"Found: {data[0]}"
            else:
                return f"Found {len(data)} results:\n" + "\n".join([f"• {item}" for item in data[:5]])
        
        elif isinstance(data, dict):
            formatted_items = []
            for key, value in data.items():
                if key.lower() in ['title', 'name', 'status', 'description']:
                    formatted_items.append(f"{key.title()}: {value}")
            
            return "\n".join(formatted_items) if formatted_items else str(data)
        
        else:
            return str(data)


# Global orchestrator instance
_orchestrator_instance = None

def get_orchestrator() -> Orchestrator:
    """Get singleton orchestrator instance"""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = Orchestrator()
    return _orchestrator_instance