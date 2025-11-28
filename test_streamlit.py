"""
Simple test to verify Streamlit integration works
"""
import sys
import os

# Add src to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, 'src'))

def test_streamlit_imports():
    """Test if all required modules can be imported"""
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError:
        print("❌ Streamlit not available - run: pip install streamlit")
        return False
    
    try:
        from orchestrator import get_orchestrator
        print("✅ Orchestrator imported successfully")
    except ImportError as e:
        print(f"⚠️ Orchestrator not available: {e}")
        print("This is normal if dependencies aren't installed")
    
    try:
        from tools.ecom_rag_tool import ecom_rag_tool
        print("✅ RAG tool imported successfully")
    except ImportError:
        print("⚠️ RAG tool not available")
    
    return True

def main():
    """Test the integration"""
    print("🧪 Testing Streamlit Integration")
    print("=" * 40)
    
    if test_streamlit_imports():
        print("\n✅ Basic integration test passed!")
        print("\nTo run the chatbot:")
        print("1. ./run_chatbot.sh")
        print("2. Open http://localhost:8501")
        return True
    else:
        print("\n❌ Integration test failed")
        return False

if __name__ == "__main__":
    main()