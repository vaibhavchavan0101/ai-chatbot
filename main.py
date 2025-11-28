"""
Main Application - Orchestrator Entry Point
Demonstrates the complete system with routing and tool execution
"""
import json
import sys
import os
from typing import Dict, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from orchestrator import get_orchestrator
from tools.ecom_rag_tool import ecom_rag_tool
from tools.order_tool import order_tool
from tools.returns_tool import returns_tool
from tools.inventory_tool import inventory_tool


class ECommerceOrchestrator:
    """Main application orchestrator"""
    
    def __init__(self):
        self.orchestrator = get_orchestrator()
        
        # Tool registry
        self.tools = {
            "ecom_rag_tool": ecom_rag_tool,
            "order_tool": order_tool,
            "returns_tool": returns_tool,
            "inventory_tool": inventory_tool
        }
    
    def process_user_query(self, query: str, user_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main entry point for processing user queries
        Follows the strict output format requirements
        """
        # Route the query using orchestrator
        routing_result = self.orchestrator.process_query(query, user_context)
        
        # Check if it's a tool call or clarification
        if isinstance(routing_result, dict) and "tool" in routing_result:
            # Execute the tool
            tool_name = routing_result["tool"]
            tool_args = routing_result["arguments"]
            
            if tool_name in self.tools:
                tool_result = self.tools[tool_name](**tool_args)
                return tool_result
            else:
                return {
                    "status": "error",
                    "error": f"Tool {tool_name} not found"
                }
        else:
            # Return clarification or direct response
            return routing_result
    
    def demonstrate_routing(self):
        """Demonstrate the routing functionality"""
        test_queries = [
            # RAG queries (should route to ecom_rag_tool)
            "What is your return policy?",
            "Can you explain the shipping guide?",
            "Tell me about the FAQ",
            "What are the terms and conditions?",
            
            # Transactional queries (should route to specific tools)
            "Track my order ORD-001",
            "Check order status for ORD-002", 
            "I want to return an item RET-001",
            "What's the return policy status?",
            "Is product PROD-001 available?",
            "Search for gaming laptop",
            
            # Unclear queries (should ask for clarification)
            "Hello",
            "Help me",
            "What can you do?"
        ]
        
        print("=== E-Commerce Orchestrator Demo ===\n")
        
        for i, query in enumerate(test_queries, 1):
            print(f"Query {i}: {query}")
            print("-" * 50)
            
            try:
                result = self.process_user_query(query)
                
                # Format output according to specifications
                if isinstance(result, dict) and "tool" in result:
                    # Tool call format
                    print("Response (Tool Call):")
                    print(json.dumps(result, indent=2))
                elif isinstance(result, dict):
                    # Tool result or error
                    print("Response (Tool Result):")
                    if result.get("status") == "success":
                        if "answer" in result:
                            # RAG tool result
                            print(f"Answer: {result['answer']}")
                            if "sources" in result:
                                print(f"Sources: {len(result['sources'])} documents")
                        elif "data" in result:
                            # Transactional tool result
                            print(f"Data: {json.dumps(result['data'], indent=2)}")
                        else:
                            print(f"Message: {result.get('message', 'Success')}")
                    else:
                        print(f"Error: {result.get('error', 'Unknown error')}")
                else:
                    # Clarification text
                    print("Response (Clarification):")
                    print(result)
                
            except Exception as e:
                print(f"Error processing query: {e}")
            
            print("\n" + "="*60 + "\n")


def run_demo():
    """Run the demonstration"""
    app = ECommerceOrchestrator()
    app.demonstrate_routing()


def interactive_mode():
    """Interactive mode for testing"""
    app = ECommerceOrchestrator()
    
    print("=== E-Commerce Orchestrator - Interactive Mode ===")
    print("Type your queries to test the system. Type 'quit' to exit.\n")
    
    while True:
        try:
            query = input("You: ").strip()
            
            if query.lower() in ['quit', 'exit', 'bye']:
                print("Goodbye!")
                break
            
            if not query:
                continue
            
            # Process query
            result = app.process_user_query(query)
            
            # Display result
            print("\nOrchestrator:", end=" ")
            
            if isinstance(result, dict):
                if "tool" in result:
                    print("(Calling tool...)")
                    print(json.dumps(result, indent=2))
                elif result.get("status") == "success":
                    if "answer" in result:
                        print(result["answer"])
                    elif "data" in result:
                        print(json.dumps(result["data"], indent=2))
                    else:
                        print(result.get("message", "Request processed successfully"))
                else:
                    print(f"Error: {result.get('error', 'Unknown error')}")
            else:
                print(result)
            
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="E-Commerce Orchestrator")
    parser.add_argument("--demo", action="store_true", help="Run demonstration")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    if args.demo:
        run_demo()
    elif args.interactive:
        interactive_mode()
    else:
        print("Use --demo for demonstration or --interactive for interactive mode")
        print("Example: python main.py --demo")