"""
RAG Retriever Agent that handles static knowledge queries using Milvus vector search
"""
import os
import sys
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from pymilvus import Collection, connections, utility
    MILVUS_AVAILABLE = True
except ImportError:
    print("Warning: PyMilvus not available. Running in mock mode.")
    MILVUS_AVAILABLE = False

try:
    from milvus import default_server
    MILVUS_LITE_AVAILABLE = True
except ImportError:
    print("Warning: Milvus Lite not available. Will try regular Milvus.")
    MILVUS_LITE_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Warning: SentenceTransformers not available. Using mock embeddings.")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from llm_factory import get_default_processor, LLMProcessorFactory


@dataclass
class SearchResult:
    """Search result from Milvus"""
    id: int
    text: str
    metadata: Dict[str, Any]
    similarity_score: float


class RAGRetrieverAgent:
    """RAG agent for handling static knowledge queries"""
    
    def __init__(self, 
                 collection_name: str = "ecom",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 milvus_host: str = os.getenv("MILVUS_HOST", "localhost"),
                 milvus_port: int = int(os.getenv("MILVUS_PORT", 19530)),
                 milvus_user: str = os.getenv("MILVUS_USER", None),
                 milvus_password: str = os.getenv("MILVUS_PASSWORD", None),
                 milvus_token: str = os.getenv("MILVUS_TOKEN", None),
                 milvus_secure: bool = os.getenv("MILVUS_SECURE", "false").lower() == "true"):
        
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.milvus_user = milvus_user
        self.milvus_password = milvus_password
        self.milvus_token = milvus_token
        self.milvus_secure = milvus_secure
        
        # Initialize embedding model
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(embedding_model)
                print(f"Loaded embedding model: {embedding_model}")
            except Exception as e:
                print(f"Failed to load embedding model: {e}")
                self.embedding_model = None
        else:
            self.embedding_model = None
        
        # Initialize LLM processor
        self.llm_processor = get_default_processor()
        
        # Initialize milvus connection state
        self.milvus_connected = False
        self.collection = None
        
        # Try to connect to Milvus
        if MILVUS_AVAILABLE:
            # Try cloud Milvus first if credentials are provided
            uri = os.getenv("MILVUS_URI")
            token = os.getenv("MILVUS_TOKEN")
            cloud_host = self.milvus_host != "localhost"
            
            if uri and token:
                print("🔗 Attempting connection to cloud Milvus...")
                self._connect_to_cloud_milvus()
            elif cloud_host or self.milvus_user:
                print("🔗 Attempting connection to Milvus server...")
                self._connect_to_milvus()
            else:
                print("Using file-based vector database (no cloud credentials)")
                self.milvus_connected = False
        else:
            print("Milvus not available. Running in file-based mode.")
            self.milvus_connected = False
        
        # Load collection if connected
        if self.milvus_connected:
            self.collection = self._get_or_create_collection()
    
    def _start_milvus_lite(self):
        """Start Milvus Lite (embedded version)"""
        try:
            from milvus import default_server
            
            # Start Milvus Lite server
            default_server.start()
            print("Milvus Lite server started successfully")
            
            # Connect to the local Milvus Lite server
            connections.connect(
                alias="default",
                host="127.0.0.1",
                port="19530"
            )
            print("Connected to Milvus Lite")
            self.milvus_connected = True
            
        except Exception as e:
            print(f"Failed to start Milvus Lite: {e}")
            print("Falling back to regular Milvus connection")
            self._connect_to_milvus()

    def _connect_to_milvus(self):
        """Connect to Milvus database"""
        if not MILVUS_AVAILABLE:
            print("Milvus not available. Running in mock mode.")
            self.milvus_connected = False
            return
            
        try:
            # Quick check if port is open before attempting connection
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)  # 2 second timeout
            result = sock.connect_ex((self.milvus_host, self.milvus_port))
            sock.close()
            
            if result != 0:
                print(f"Milvus server not reachable at {self.milvus_host}:{self.milvus_port}")
                print("Running in mock mode without Milvus")
                self.milvus_connected = False
                return
            
            # Attempt Milvus connection with timeout
            connection_params = {
                "alias": "default",
                "host": self.milvus_host,
                "port": str(self.milvus_port)
            }
            
            # Add cloud credentials if provided
            if self.milvus_user and self.milvus_password:
                connection_params["user"] = self.milvus_user
                connection_params["password"] = self.milvus_password
                print(f"Connecting to cloud Milvus at {self.milvus_host}:{self.milvus_port} with credentials")
            
            if self.milvus_token:
                connection_params["token"] = self.milvus_token
                print(f"Connecting to cloud Milvus with API token")
                
            if self.milvus_secure:
                connection_params["secure"] = True
                print("Using secure connection (TLS/SSL)")
            
            connections.connect(**connection_params)
            print(f"Successfully connected to Milvus at {self.milvus_host}:{self.milvus_port}")
            self.milvus_connected = True
        except Exception as e:
            print(f"Failed to connect to Milvus: {e}")
            print("Running in mock mode without Milvus")
            self.milvus_connected = False
    
    def _connect_to_cloud_milvus(self):
        """Connect to cloud Milvus using URI and token"""
        try:
            uri = os.getenv("MILVUS_URI")
            token = os.getenv("MILVUS_TOKEN")
            
            if not (uri and token):
                print("Cloud Milvus credentials not found in environment")
                self.milvus_connected = False
                return
            
            print(f"Connecting to cloud Milvus: {uri}")
            connections.connect(
                alias="default",
                uri=uri,
                token=token
            )
            print("✅ Successfully connected to cloud Milvus!")
            self.milvus_connected = True
            
        except Exception as e:
            print(f"❌ Failed to connect to cloud Milvus: {e}")
            print("Falling back to file-based mode")
            self.milvus_connected = False
    
    def _get_or_create_collection(self) -> Optional[Collection]:
        """Get existing collection or create new one"""
        if not self.milvus_connected:
            return None
            
        try:
            if utility.has_collection(self.collection_name):
                collection = Collection(self.collection_name)
                collection.load()
                return collection
            else:
                print(f"Collection {self.collection_name} not found. Please run the setup script to create it.")
                return None
        except Exception as e:
            print(f"Error accessing collection: {e}")
            return None
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        if self.embedding_model:
            try:
                embedding = self.embedding_model.encode(text)
                return embedding.tolist()
            except Exception as e:
                print(f"Error generating embedding: {e}")
        
        # Fallback to hash-based mock embedding
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        hash_int = int(hash_obj.hexdigest()[:8], 16)
        # Generate a deterministic 384-dimensional embedding
        embedding = [(hash_int >> i) % 2 - 0.5 for i in range(384)]
        return embedding
    
    def search_documents(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Search for relevant documents in Milvus or file database"""
        # Use cloud Milvus if connected, otherwise fallback to file-based
        if self.milvus_connected and self.collection:
            print(f"🔍 Searching cloud database for: '{query}'")
            return self._search_milvus_collection(query, top_k)
        else:
            print(f"🔍 Searching file-based database for: '{query}'")
            return self._mock_search_results(query)
    
    def _search_milvus_collection(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Search the cloud Milvus collection"""
        try:
            # Generate query embedding
            query_embedding = self.generate_embedding(query)
            if not query_embedding:
                print("Failed to generate embedding, using file fallback")
                return self._mock_search_results(query)
            
            # Search in Milvus collection
            search_params = {
                "metric_type": "COSINE",
                "params": {"ef": 64}  # For AUTOINDEX
            }
            
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["text", "metadata"]
            )
            
            search_results = []
            if results and len(results[0]) > 0:
                for hit in results[0]:
                    search_results.append(SearchResult(
                        id=getattr(hit, 'id', 0),
                        text=hit.entity.get('text', ''),
                        metadata=hit.entity.get('metadata', {}),
                        similarity_score=float(hit.score)
                    ))
                print(f"📁 Found {len(search_results)} results from cloud database")
            else:
                print("No results from cloud database, using file fallback")
                return self._mock_search_results(query)
            
            return search_results
            
        except Exception as e:
            print(f"Error searching cloud database: {e}, using file fallback")
            return self._mock_search_results(query)
    
    def _mock_search_results(self, query: str) -> List[SearchResult]:
        """Generate mock search results from file-based database or defaults"""
        # Try to load from file-based database
        try:
            import json
            db_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'vector_database.json')
            
            if os.path.exists(db_file):
                with open(db_file, 'r') as f:
                    db_data = json.load(f)
                
                documents = db_data.get('documents', [])
                query_lower = query.lower()
                
                # Simple keyword matching for relevance
                scored_results = []
                for doc in documents:
                    text = doc['text'].lower()
                    metadata = doc['metadata']
                    
                    # Calculate relevance score based on keyword matches
                    score = 0.0
                    query_words = query_lower.split()
                    
                    for word in query_words:
                        if word in text:
                            score += 0.3
                        if word in metadata.get('topic', '').lower():
                            score += 0.5
                        if word in metadata.get('filename', '').lower():
                            score += 0.4
                    
                    if score > 0:
                        scored_results.append((score, doc))
                
                # Sort by relevance score
                scored_results.sort(key=lambda x: x[0], reverse=True)
                
                # Convert to SearchResult objects
                search_results = []
                for score, doc in scored_results[:3]:  # Top 3 results
                    search_results.append(SearchResult(
                        id=doc['id'],
                        text=doc['text'],
                        metadata=doc['metadata'],
                        similarity_score=min(score, 1.0)  # Cap at 1.0
                    ))
                
                if search_results:
                    print(f"📁 Found {len(search_results)} results from local database")
                    return search_results
                else:
                    print("📁 No matches found in local database, using defaults")
                
        except Exception as e:
            print(f"Error loading file database: {e}")
            print("Using hardcoded fallback data")
        
        # Fallback to hardcoded mock results
        mock_results = [
            SearchResult(
                id=1,
                text="Our return policy allows returns within 30 days of purchase. Items must be in original condition with tags attached. Refunds are processed within 5-7 business days.",
                metadata={"filename": "return_policy.pdf", "topic": "returns", "source": "policy_documents"},
                similarity_score=0.85
            ),
            SearchResult(
                id=2, 
                text="For order tracking, please use your order number and email address on our tracking page. Orders typically ship within 1-2 business days and arrive within 5-7 days.",
                metadata={"filename": "shipping_guide.pdf", "topic": "shipping", "source": "customer_service"},
                similarity_score=0.78
            ),
            SearchResult(
                id=3,
                text="Our customer service team is available Monday-Friday 9AM-6PM EST. You can reach us via email at support@ecommerce.com or phone at 1-800-SHOP-NOW.",
                metadata={"filename": "contact_info.pdf", "topic": "support", "source": "customer_service"}, 
                similarity_score=0.72
            )
        ]
        
        # Filter mock results based on query relevance
        query_lower = query.lower()
        relevant_results = []
        
        for result in mock_results:
            if any(word in result.text.lower() for word in query_lower.split()):
                relevant_results.append(result)
        
        return relevant_results[:3] if relevant_results else mock_results[:1]
    
    def synthesize_answer(self, query: str, search_results: List[SearchResult]) -> str:
        """Generate a synthesized answer using LLM"""
        if not search_results:
            return "I couldn't find any relevant information for your query. Please contact customer service for assistance."
        
        # Prepare context from search results
        context_parts = []
        for i, result in enumerate(search_results, 1):
            context_parts.append(f"Document {i}: {result.text}")
        
        context = "\n\n".join(context_parts)
        
        # Create messages for LLM
        messages = [
            {
                "role": "system",
                "content": "You are a helpful customer service assistant. Use the provided context to answer the user's question accurately and concisely. If the context doesn't contain relevant information, say so clearly."
            },
            {
                "role": "user", 
                "content": f"Question: {query}\n\nContext:\n{context}\n\nPlease provide a helpful answer based on the context above."
            }
        ]
        
        try:
            response = self.llm_processor.generate_completion(messages, max_tokens=300)
            return response
        except Exception as e:
            print(f"Error generating synthesis: {e}")
            # Fallback to simple concatenation
            return f"Based on our documentation: {search_results[0].text}"
    
    def process_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Main entry point for processing RAG queries"""
        try:
            # Search for relevant documents
            search_results = self.search_documents(query, top_k=5)
            
            # Generate synthesized answer
            answer = self.synthesize_answer(query, search_results)
            
            return {
                "status": "success",
                "answer": answer,
                "sources": [
                    {
                        "id": result.id,
                        "text": result.text[:200] + "..." if len(result.text) > 200 else result.text,
                        "metadata": result.metadata,
                        "similarity_score": result.similarity_score
                    }
                    for result in search_results
                ],
                "query": query
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "query": query
            }


# Global RAG agent instance
_rag_agent_instance = None

def get_rag_agent() -> RAGRetrieverAgent:
    """Get singleton RAG agent instance"""
    global _rag_agent_instance
    if _rag_agent_instance is None:
        _rag_agent_instance = RAGRetrieverAgent(
            collection_name=os.getenv("MILVUS_COLLECTION_NAME", "ecommerce_docs"),
            milvus_host=os.getenv("MILVUS_HOST", "localhost"),
            milvus_port=int(os.getenv("MILVUS_PORT", "19530"))
        )
    return _rag_agent_instance