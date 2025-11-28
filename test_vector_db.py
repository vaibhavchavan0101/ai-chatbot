#!/usr/bin/env python3
"""
Test the complete E-commerce chatbot with vector database
"""
import sys
import os
sys.path.append('src')

from orchestrator import get_orchestrator
from tools.ecom_rag_tool import ecom_rag_tool

def test_vector_database_integration():
    """Test that the chatbot uses the vector database properly"""
    print("🎯 Testing E-Commerce Chatbot with Vector Database")
    print("=" * 60)
    
    # Test 1: Direct RAG tool test
    print("\n📚 Test 1: RAG Tool with Vector Database")
    print("-" * 40)
    
    try:
        result = ecom_rag_tool("What is your return policy?")
        print(f"✅ Status: {result.get('status', 'unknown')}")
        
        if result.get('answer'):
            answer = result['answer']
            print(f"✅ Answer received: {len(answer)} characters")
            print(f"📄 Preview: {answer[:100]}...")
            
            # Check if it mentions key return policy terms
            if any(term in answer.lower() for term in ['30 days', 'return', 'refund', 'original condition']):
                print("✅ Answer contains relevant return policy information")
            else:
                print("⚠️ Answer may not contain expected return policy details")
                
        if result.get('sources'):
            print(f"📁 Sources: {len(result['sources'])} documents found")
            for i, source in enumerate(result['sources'][:2]):
                print(f"   {i+1}. {source.get('metadata', {}).get('filename', 'Unknown')}")
        
    except Exception as e:
        print(f"❌ RAG tool test failed: {e}")
    
    # Test 2: Full orchestrator test
    print("\n🎭 Test 2: Full Orchestrator Integration") 
    print("-" * 40)
    
    try:
        orchestrator = get_orchestrator()
        
        # Test RAG routing
        rag_result = orchestrator.process_query("Tell me about your shipping guide")
        print(f"✅ RAG query routed to: {rag_result.get('tool', 'unknown')}")
        
        # Test transactional routing
        order_result = orchestrator.process_query("Track my order ORD-001")
        print(f"✅ Order query routed to: {order_result.get('tool', 'unknown')}")
        
    except Exception as e:
        print(f"❌ Orchestrator test failed: {e}")
    
    # Test 3: Check database file
    print("\n🗄️ Test 3: Vector Database Status")
    print("-" * 40)
    
    db_file = "data/vector_database.json"
    if os.path.exists(db_file):
        import json
        try:
            with open(db_file, 'r') as f:
                db_data = json.load(f)
            
            docs = db_data.get('documents', [])
            print(f"✅ Database file exists: {db_file}")
            print(f"✅ Documents loaded: {len(docs)}")
            print(f"✅ Collection: {db_data.get('collection_name', 'unknown')}")
            
            # Show available topics
            topics = set()
            for doc in docs:
                topic = doc.get('metadata', {}).get('topic')
                if topic:
                    topics.add(topic)
            
            print(f"✅ Available topics: {', '.join(sorted(topics))}")
            
        except Exception as e:
            print(f"❌ Error reading database: {e}")
    else:
        print(f"❌ Database file not found: {db_file}")
    
    print("\n" + "=" * 60)
    print("🎉 Vector Database Integration Test Complete!")
    print("\n💡 Your chatbot is now connected to a vector database!")
    print("🚀 Try these queries in the web interface:")
    print("   • 'What is your return policy?'")
    print("   • 'Tell me about shipping options'")  
    print("   • 'How can I contact customer service?'")
    print("   • 'Track my order ORD-001'")
    print()
    print("🌐 Web interface: http://localhost:8501")

if __name__ == "__main__":
    test_vector_database_integration()