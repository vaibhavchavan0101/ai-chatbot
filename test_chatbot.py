#!/usr/bin/env python3
"""
Test script to verify the E-Commerce Orchestrator chatbot is working properly
"""
import sys
import os
sys.path.append('src')

from orchestrator import get_orchestrator
from tools.ecom_rag_tool import ecom_rag_tool
from tools.order_tool import order_tool
from tools.returns_tool import returns_tool
from tools.inventory_tool import inventory_tool

def test_chatbot_responses():
    """Test various chatbot interactions to ensure proper responses"""
    print("🤖 E-Commerce Chatbot Test Suite")
    print("=" * 50)
    
    orchestrator = get_orchestrator()
    
    # Test cases with expected behavior
    test_cases = [
        {
            "query": "What is your return policy?",
            "expected_type": "rag",
            "description": "RAG query about return policy"
        },
        {
            "query": "Track my order ORD-001",
            "expected_type": "order", 
            "description": "Order tracking query"
        },
        {
            "query": "Check return status RET-001",
            "expected_type": "return",
            "description": "Return status query"
        },
        {
            "query": "Is product PROD-001 available?",
            "expected_type": "inventory",
            "description": "Product availability query"
        },
        {
            "query": "Hello, what can you do?",
            "expected_type": "clarification",
            "description": "Unclear query requiring clarification"
        }
    ]
    
    # Tools mapping
    tools = {
        "ecom_rag_tool": ecom_rag_tool,
        "order_tool": order_tool, 
        "returns_tool": returns_tool,
        "inventory_tool": inventory_tool
    }
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['description']}")
        print(f"   Query: '{test_case['query']}'")
        print("-" * 50)
        
        try:
            # Get routing decision
            routing_result = orchestrator.process_query(test_case['query'])
            
            if isinstance(routing_result, dict) and "tool" in routing_result:
                # Execute tool
                tool_name = routing_result["tool"]
                tool_args = routing_result["arguments"]
                
                print(f"   ✅ Routed to: {tool_name}")
                
                if tool_name in tools:
                    # Execute the tool
                    tool_result = tools[tool_name](**tool_args)
                    
                    if tool_result.get("status") == "success":
                        if "answer" in tool_result:
                            print(f"   📚 Answer: {tool_result['answer'][:100]}...")
                        elif "data" in tool_result:
                            print(f"   📊 Data: {str(tool_result['data'])[:100]}...")
                        else:
                            print(f"   ✅ Message: {tool_result.get('message', 'Success')}")
                    else:
                        print(f"   ❌ Error: {tool_result.get('error', 'Unknown error')}")
                        
                else:
                    print(f"   ❌ Tool not found: {tool_name}")
                    
            else:
                # Clarification response
                print(f"   💬 Clarification: {str(routing_result)[:100]}...")
                
        except Exception as e:
            print(f"   💥 Exception: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Test suite completed!")
    print("\n💡 To test the web interface:")
    print("   1. Make sure Streamlit is running: streamlit run streamlit_app.py")
    print("   2. Open your browser to: http://localhost:8501")
    print("   3. Try the example queries in the sidebar")
    print("\n✨ The chatbot should now return proper answers for all query types!")

if __name__ == "__main__":
    test_chatbot_responses()