#!/usr/bin/env python3
"""
Test script for cloud Milvus connection
Run this to test your cloud database credentials
"""
import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.append('src')

# Load environment variables
load_dotenv()

def test_cloud_connection():
    """Test connection to cloud Milvus database"""
    print("🧪 Testing Cloud Milvus Connection")
    print("=" * 50)
    
    # Load configuration
    from src.cloud_config import load_cloud_config
    config = load_cloud_config()
    
    print("📋 Current Configuration:")
    print(f"   Host: {config['milvus_host']}")
    print(f"   Port: {config['milvus_port']}")
    print(f"   User: {'***' if config['milvus_user'] else 'Not set'}")
    print(f"   Password: {'***' if config['milvus_password'] else 'Not set'}")
    print(f"   Token: {'***' if config['milvus_token'] else 'Not set'}")
    print(f"   Secure: {config['milvus_secure']}")
    print(f"   Collection: {config['collection_name']}")
    print()
    
    # Check if cloud credentials are provided
    if config['milvus_host'] == 'localhost':
        print("⚠️  Still using localhost configuration")
        print("   Update .env file with your cloud credentials:")
        print("   MILVUS_HOST=your-cloud-host.example.com")
        print("   MILVUS_USER=your-username")
        print("   MILVUS_PASSWORD=your-password")
        print("   MILVUS_SECURE=true")
        return
    
    # Test connection
    try:
        print("🔌 Attempting connection to cloud Milvus...")
        from src.cloud_config import create_rag_agent_with_cloud_config
        
        # Create agent with cloud config
        agent = create_rag_agent_with_cloud_config()
        
        if agent.milvus_connected:
            print("✅ Successfully connected to cloud Milvus!")
            
            # Test basic operations
            if agent.collection:
                print(f"📊 Collection '{agent.collection_name}' loaded")
                # You can add more tests here
            else:
                print("📋 Collection not found - you may need to create it")
        else:
            print("❌ Failed to connect to cloud Milvus")
            print("   Check your credentials and network connection")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("   Verify your credentials and try again")

def show_env_template():
    """Show environment template for cloud setup"""
    print("\\n" + "=" * 50)
    print("📝 Cloud Configuration Template")
    print("=" * 50)
    print("Add these lines to your .env file:")
    print()
    print("# Cloud Milvus Configuration")
    print("MILVUS_HOST=your-cluster-endpoint.milvus.io")
    print("MILVUS_PORT=19530")
    print("MILVUS_USER=your-username")
    print("MILVUS_PASSWORD=your-password")
    print("MILVUS_SECURE=true")
    print()
    print("# Optional: Use API token instead of username/password")
    print("# MILVUS_TOKEN=your-api-token")
    print()

if __name__ == "__main__":
    test_cloud_connection()
    show_env_template()