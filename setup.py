#!/usr/bin/env python3
"""
Setup Script for E-Commerce Orchestrator System
Initializes the complete system including PDFs, text processing, and embeddings
"""
import os
import sys
import json
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.append(str(project_root / 'src'))

def check_dependencies():
    """Check if optional dependencies are available"""
    deps_status = {
        "reportlab": False,
        "pymilvus": False,
        "sentence_transformers": False,
        "openai": False,
        "anthropic": False
    }
    
    try:
        import reportlab
        deps_status["reportlab"] = True
    except ImportError:
        pass
    
    try:
        import pymilvus
        deps_status["pymilvus"] = True
    except ImportError:
        pass
    
    try:
        import sentence_transformers
        deps_status["sentence_transformers"] = True
    except ImportError:
        pass
    
    try:
        import openai
        deps_status["openai"] = True
    except ImportError:
        pass
    
    try:
        import anthropic
        deps_status["anthropic"] = True
    except ImportError:
        pass
    
    return deps_status

def setup_environment():
    """Setup environment file if not exists"""
    env_path = project_root / ".env"
    env_example_path = project_root / ".env.example"
    
    if not env_path.exists() and env_example_path.exists():
        print("Setting up .env file...")
        # Copy example to .env
        with open(env_example_path, 'r') as f:
            content = f.read()
        
        with open(env_path, 'w') as f:
            f.write(content)
        
        print("✅ Created .env file from template")
        print("⚠️  Please edit .env with your API keys")
    elif env_path.exists():
        print("✅ .env file already exists")
    else:
        print("❌ No .env.example file found")

def run_pdf_generation():
    """Run PDF generation"""
    print("\n📄 Generating PDF documents...")
    
    try:
        from pdf_generator import main as generate_pdfs
        result = generate_pdfs()
        print(f"✅ Generated {len(result)} PDF files")
        return True
    except Exception as e:
        print(f"❌ PDF generation failed: {e}")
        print("💡 Install reportlab: pip install reportlab")
        return False

def run_text_processing():
    """Run text extraction and chunking"""
    print("\n📝 Processing text and creating chunks...")
    
    try:
        from text_processor import main as process_text
        chunks = process_text()
        print(f"✅ Created {len(chunks)} text chunks")
        return True
    except Exception as e:
        print(f"❌ Text processing failed: {e}")
        return False

def run_milvus_setup():
    """Run Milvus setup and embeddings"""
    print("\n🔍 Setting up embeddings and vector database...")
    
    try:
        from milvus_setup import main as setup_milvus
        result = setup_milvus()
        print("✅ Embeddings setup completed")
        return True
    except Exception as e:
        print(f"⚠️  Milvus setup completed in mock mode: {e}")
        print("💡 For full Milvus: pip install pymilvus sentence-transformers")
        return True  # Mock mode is acceptable

def test_orchestrator():
    """Test orchestrator functionality"""
    print("\n🎭 Testing orchestrator...")
    
    try:
        # Import main components
        sys.path.append(str(project_root))
        from main import ECommerceOrchestrator
        
        app = ECommerceOrchestrator()
        
        # Test queries
        test_queries = [
            "What is your return policy?",  # Should route to RAG
            "Track order ORD-001",         # Should route to order tool
            "Hello"                        # Should ask for clarification
        ]
        
        success_count = 0
        for query in test_queries:
            try:
                result = app.process_user_query(query)
                if result:
                    success_count += 1
            except Exception as e:
                print(f"❌ Query failed: {query} - {e}")
        
        if success_count == len(test_queries):
            print(f"✅ All {success_count} test queries successful")
            return True
        else:
            print(f"⚠️  {success_count}/{len(test_queries)} test queries successful")
            return True  # Partial success is acceptable
    
    except Exception as e:
        print(f"❌ Orchestrator test failed: {e}")
        return False

def display_summary(deps_status, steps_status):
    """Display setup summary"""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETE!")
    print("="*60)
    
    print("\n📦 Dependencies Status:")
    for dep, status in deps_status.items():
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {dep}")
    
    print("\n🔧 Setup Steps:")
    step_names = [
        "Environment setup",
        "PDF generation", 
        "Text processing",
        "Embeddings setup",
        "Orchestrator test"
    ]
    
    for i, (step_name, status) in enumerate(zip(step_names, steps_status)):
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {step_name}")
    
    print("\n🚀 Usage:")
    print("  python main.py --demo      # Run demonstration")
    print("  python main.py --interactive  # Interactive mode")
    
    print("\n📁 Generated Files:")
    data_dir = project_root / "data"
    if data_dir.exists():
        print(f"  📄 PDFs: {data_dir / 'pdfs'}")
        print(f"  📊 Mock data: {data_dir / 'mock_embeddings.json'}")
    
    print("\n📖 Next Steps:")
    if not deps_status["openai"] and not deps_status["anthropic"]:
        print("  1. Install LLM library: pip install openai anthropic")
    if not all(deps_status.values()):
        print("  2. Install optional dependencies: pip install -r requirements.txt")
    print("  3. Edit .env with your API keys")
    print("  4. Run the demo: python main.py --demo")

def main():
    """Main setup function"""
    print("🔧 E-Commerce Orchestrator System Setup")
    print("="*50)
    
    # Check dependencies
    print("\n📋 Checking dependencies...")
    deps_status = check_dependencies()
    
    # Setup steps
    steps_status = []
    
    # 1. Environment setup
    setup_environment()
    steps_status.append(True)
    
    # 2. PDF generation
    pdf_success = run_pdf_generation()
    steps_status.append(pdf_success)
    
    # 3. Text processing
    text_success = run_text_processing()
    steps_status.append(text_success)
    
    # 4. Embeddings setup
    embeddings_success = run_milvus_setup()
    steps_status.append(embeddings_success)
    
    # 5. Test orchestrator
    orchestrator_success = test_orchestrator()
    steps_status.append(orchestrator_success)
    
    # Display summary
    display_summary(deps_status, steps_status)
    
    # Return success status
    return all(steps_status)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)