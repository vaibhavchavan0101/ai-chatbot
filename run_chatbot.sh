#!/bin/bash

# Launch script for E-Commerce AI Chatbot Streamlit App

echo "🚀 Starting E-Commerce AI Chatbot..."
echo "============================================"

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit not found. Installing..."
    pip install streamlit
fi

# Set environment variables for better Streamlit experience
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=localhost
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

echo "📱 Starting Streamlit app on http://localhost:8501"
echo "Press Ctrl+C to stop the server"
echo "============================================"

# Launch Streamlit app
streamlit run streamlit_app.py --server.port 8501 --server.address localhost