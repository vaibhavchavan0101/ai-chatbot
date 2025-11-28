# E-Commerce Orchestrator System

A sophisticated multi-agent orchestrator system that manages RAG (Retrieval-Augmented Generation) and transactional agents using Chat-Completion style prompting with strict intent-based routing.

## System Overview

This system implements an Orchestrator that routes user queries to appropriate agents and tools:

- **RAG Retriever Agent** → Handles static knowledge queries using Milvus vector search
- **Transactional Agents** → OrderStatusAgent, ReturnAgent, ProductAgent for dynamic operations
- **LLM Processor Factory** → Manages multiple LLM providers with singleton pattern
- **PDF Generation Pipeline** → Creates 6-15 PDFs with LLM-generated content
- **Text Processing** → Extracts and chunks text into 200-500 word segments
- **Milvus Integration** → Vector database for semantic search with embeddings

## Architecture

```
User Query → Orchestrator → Route Decision → Tool/Agent → Response
                ↓
        Intent Classification:
        - RAG Keywords → ecom_rag_tool
        - Transactional Keywords → order_tool/returns_tool/inventory_tool
        - Unclear → Clarification Request
```

## Routing Rules (Mandatory)

The system uses strict heuristic-based routing:

1. **IF** query contains: `policy`, `rules`, `FAQ`, `how to`, `guide`, `manual`, `terms`, `details`, `brochure`
   → **RAG tool** (ecom_rag_tool)

2. **ELSE IF** query contains: `book`, `cancel`, `return`, `check order`, `apply`, `track`, `order`, `availability`
   → **Transactional tool** (order_tool, returns_tool, inventory_tool)

3. **ELSE** → Ask user for clarification

## Output Format

When calling a tool:
```json
{
  "tool": "TOOL_NAME",
  "arguments": { ... }
}
```

When responding normally:
```
{text}
```

## Installation

### Prerequisites

- Python 3.8+
- Docker (optional, for Milvus)
- OpenAI API Key or Anthropic API Key

### Setup Steps

1. **Clone and install dependencies:**
```bash
cd project
pip install -r requirements.txt
```

2. **Configure environment:**
```bash
cp .env.example .env
# Edit .env with your API keys
```

3. **Install optional dependencies for full functionality:**
```bash
# For Milvus (vector database)
pip install pymilvus

# For PDF processing
pip install reportlab PyMuPDF pdfminer.six

# For embeddings
pip install sentence-transformers
```

4. **Start Milvus (optional - system works without it):**
```bash
# Using Docker
docker run -p 19530:19530 -p 9091:9091 milvusdb/milvus:latest
```

## Quick Start

### 1. Generate Sample Data
```bash
python src/pdf_generator.py
python src/text_processor.py  
python src/milvus_setup.py
```

### 2. Run Demo
```bash
python main.py --demo
```

### 3. Interactive Mode
```bash
python main.py --interactive
```

### 4. Streamlit Web Interface
```bash
# Install streamlit
pip install streamlit

# Launch the web chatbot
./run_chatbot.sh
# OR
streamlit run streamlit_app.py

# Open browser to http://localhost:8501
```

## Usage Examples

### RAG Queries (Routes to ecom_rag_tool)
```python
"What is your return policy?"
"Can you explain the shipping guide?"
"Tell me about the FAQ"
"What are the terms and conditions?"
```

### Transactional Queries (Routes to specific tools)
```python
"Track my order ORD-001"          # → order_tool
"Check order status for ORD-002"  # → order_tool
"I want to return item RET-001"   # → returns_tool
"Is product PROD-001 available?"  # → inventory_tool
"Search for gaming laptop"        # → inventory_tool
```

### Unclear Queries (Requests clarification)
```python
"Hello"
"Help me"
"What can you do?"
```

## System Components

### 1. LLM Processor Factory (`src/llm_factory.py`)
- Manages OpenAI and Anthropic processors
- Singleton pattern for (provider, model) pairs
- Fallback embedding generation

### 2. Orchestrator (`src/orchestrator.py`)
- Main routing logic with strict intent classification
- Chat-Completion style prompting
- No assumptions without user input

### 3. RAG Retriever Agent (`src/agents/rag_agent.py`)
- Handles static knowledge queries
- Milvus vector search integration
- LLM-powered answer synthesis

### 4. Transactional Agents (`src/agents/transactional_agents.py`)
- **OrderStatusAgent**: Order tracking and status
- **ReturnAgent**: Return policy and processing
- **ProductAgent**: Inventory and product search

### 5. Tools (`src/tools/`)
- **ecom_rag_tool**: Interface to RAG agent
- **order_tool**: Interface to OrderStatusAgent
- **returns_tool**: Interface to ReturnAgent  
- **inventory_tool**: Interface to ProductAgent

### 6. PDF Pipeline (`src/pdf_generator.py`)
- Generates 6-15 PDFs using LLM content
- Topics: returns, shipping, FAQ, policies, etc.
- ReportLab integration for professional documents

### 7. Text Processing (`src/text_processor.py`)
- Extracts text from PDFs (PyMuPDF/pdfminer.six)
- Chunks text into 200-500 word segments
- Maintains overlap for context preservation

### 8. Milvus Setup (`src/milvus_setup.py`)
- Configures vector database schema
- Generates embeddings (sentence-transformers)
- Inserts chunks with proper metadata

### 9. Streamlit Interface (`streamlit_app.py`)
- Live web chatbot interface on localhost
- Real-time chat with session management
- Visual tool call information and routing details
- Interactive examples and system status
- Responsive design with sidebar controls

## Database Schema (Milvus)

```sql
Collection: ecommerce_docs
Fields:
- id (INT64, PRIMARY KEY, AUTO_ID)
- embedding (FLOAT_VECTOR, dim=384) 
- text (VARCHAR, max_length=10000)
- metadata (JSON: filename, topic, source, created_at)
```

## Mock Mode

The system includes comprehensive mock functionality for development:
- **Mock PDFs**: Generated with realistic e-commerce content
- **Mock Milvus**: Local JSON file storage for testing
- **Mock Agents**: Realistic order/return/product data
- **Mock Embeddings**: Deterministic hash-based vectors

## Configuration

### Environment Variables
```bash
# API Keys
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# Milvus Configuration
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION_NAME=ecommerce_docs

# LLM Configuration  
DEFAULT_PROVIDER=openai
DEFAULT_MODEL=gpt-3.5-turbo
```

### Routing Keywords
```python
# RAG Keywords
rag_keywords = {
    'policy', 'rules', 'faq', 'how to', 'guide', 'manual',
    'terms', 'details', 'brochure', 'information', 'help'
}

# Transactional Keywords
transactional_keywords = {
    'book', 'cancel', 'return', 'check order', 'apply', 'track',
    'order', 'availability', 'status', 'refund', 'purchase'
}
```

## Student Deliverables Checklist

- ✅ **Agents implemented & routed correctly**
- ✅ **Mock transactional APIs** (orders, returns, products)
- ✅ **All PDFs generated with metadata** (10 documents)
- ✅ **Text extracted & chunked** (200-500 words)
- ✅ **Embeddings created & inserted** (Milvus/mock)
- ✅ **Retriever tool fully functional** (ecom_rag_tool)
- ✅ **RAG routing + transactional routing demonstrated**
- ✅ **README + demo examples**

## Testing

### Run All Tests
```bash
# Generate PDFs
python src/pdf_generator.py

# Process text and create chunks  
python src/text_processor.py

# Setup embeddings database
python src/milvus_setup.py

# Test orchestrator routing
python main.py --demo

# Interactive testing
python main.py --interactive
```

### Expected Outputs
- 10+ PDF files in `data/pdfs/`
- Text chunks with metadata
- Embeddings in Milvus or `data/mock_embeddings.json`
- Routing demonstration with proper tool calls
- Interactive query processing

## Troubleshooting

### Common Issues

1. **Import errors**: Install dependencies with `pip install -r requirements.txt`

2. **Milvus connection failed**: 
   - System works in mock mode without Milvus
   - Start Milvus with Docker or use mock storage

3. **API key errors**:
   - Set environment variables in `.env`
   - System includes mock LLM responses

4. **PDF generation fails**:
   - Install ReportLab: `pip install reportlab`
   - System falls back to mock content

### Mock vs Full Mode

| Component | Mock Mode | Full Mode |
|-----------|-----------|-----------|
| LLM | Fallback responses | OpenAI/Anthropic |
| Milvus | JSON file storage | Vector database |
| PDFs | Generated content | LLM-generated |
| Embeddings | Hash-based | sentence-transformers |

## Contributing

1. Follow the Chat-Completion style prompting format
2. Maintain strict intent-based routing rules
3. Never mix tool output with natural language
4. Use singleton patterns for processors and agents
5. Include comprehensive error handling and fallbacks

## License

Educational use only. Follow provider terms for LLM APIs.