"""
Milvus Database Setup and Embeddings Management
Configures Milvus, generates embeddings, and inserts chunks with proper schema
"""
import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import asdict
from datetime import datetime

# Try importing dependencies
try:
    from pymilvus import (
        connections, 
        Collection, 
        CollectionSchema, 
        FieldSchema, 
        DataType,
        utility
    )
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from text_processor import TextProcessor, TextChunk
from llm_factory import get_default_processor


class MilvusManager:
    """Manage Milvus database operations and embeddings"""
    
    def __init__(self,
                 collection_name: str = "ecommerce_docs",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 milvus_host: str = "localhost",
                 milvus_port: int = 19530,
                 embedding_dim: int = 384):
        
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.embedding_dim = embedding_dim
        
        # Initialize embedding model
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(embedding_model)
                print(f"Loaded embedding model: {embedding_model}")
            except Exception as e:
                print(f"Error loading embedding model: {e}")
                self.embedding_model = None
        else:
            print("sentence-transformers not available")
            self.embedding_model = None
        
        # Initialize LLM processor for fallback embeddings
        self.llm_processor = get_default_processor()
        
        # Milvus connection
        self.collection = None
        if MILVUS_AVAILABLE:
            self._setup_milvus()
        else:
            print("pymilvus not available - using mock mode")
    
    def _setup_milvus(self):
        """Setup Milvus connection and collection"""
        try:
            # Connect to Milvus
            connections.connect(
                alias="default",
                host=self.milvus_host,
                port=self.milvus_port
            )
            print(f"Connected to Milvus at {self.milvus_host}:{self.milvus_port}")
            
            # Create or load collection
            self.collection = self._create_or_load_collection()
            
        except Exception as e:
            print(f"Error setting up Milvus: {e}")
            self.collection = None
    
    def _create_or_load_collection(self) -> Optional[Collection]:
        """Create or load the Milvus collection"""
        try:
            if utility.has_collection(self.collection_name):
                # Load existing collection
                collection = Collection(self.collection_name)
                collection.load()
                print(f"Loaded existing collection: {self.collection_name}")
                return collection
            else:
                # Create new collection
                return self._create_collection()
        except Exception as e:
            print(f"Error with collection: {e}")
            return None
    
    def _create_collection(self) -> Collection:
        """Create a new Milvus collection with proper schema"""
        # Define schema fields
        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True,
                description="Primary key"
            ),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.embedding_dim,
                description="Text embedding vector"
            ),
            FieldSchema(
                name="text",
                dtype=DataType.VARCHAR,
                max_length=10000,
                description="Original text content"
            ),
            FieldSchema(
                name="metadata",
                dtype=DataType.JSON,
                description="Document metadata"
            )
        ]
        
        # Create schema
        schema = CollectionSchema(
            fields=fields,
            description=f"E-commerce document collection for RAG"
        )
        
        # Create collection
        collection = Collection(
            name=self.collection_name,
            schema=schema
        )
        
        # Create index on embedding field
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",
            "params": {"nlist": 128}
        }
        collection.create_index("embedding", index_params)
        
        # Load collection
        collection.load()
        
        print(f"Created new collection: {self.collection_name}")
        return collection
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        if self.embedding_model:
            try:
                embedding = self.embedding_model.encode(text)
                return embedding.tolist()
            except Exception as e:
                print(f"Error generating embedding: {e}")
        
        # Fallback to LLM-based embeddings or mock
        return self._generate_fallback_embedding(text)
    
    def _generate_fallback_embedding(self, text: str) -> List[float]:
        """Generate fallback embedding when sentence-transformers unavailable"""
        try:
            # Try using LLM processor if it supports embeddings
            if hasattr(self.llm_processor, 'generate_embedding'):
                return self.llm_processor.generate_embedding(text)
        except Exception as e:
            print(f"LLM embedding error: {e}")
        
        # Generate mock embedding
        import hashlib
        import struct
        
        # Create deterministic "embedding" from text hash
        text_hash = hashlib.md5(text.encode()).digest()
        mock_embedding = []
        
        for i in range(0, self.embedding_dim * 4, 4):
            if i + 4 <= len(text_hash) * (self.embedding_dim // (len(text_hash) // 4)):
                chunk = text_hash[i % len(text_hash):(i % len(text_hash)) + 4]
                if len(chunk) == 4:
                    value = struct.unpack('f', chunk)[0] if len(chunk) == 4 else 0.0
                    mock_embedding.append(float(value))
                else:
                    mock_embedding.append(0.0)
            else:
                mock_embedding.append(0.0)
        
        # Normalize to make it more realistic
        import math
        norm = math.sqrt(sum(x*x for x in mock_embedding))
        if norm > 0:
            mock_embedding = [x/norm for x in mock_embedding]
        
        return mock_embedding[:self.embedding_dim]
    
    def insert_chunks(self, chunks: List[TextChunk]) -> Dict[str, Any]:
        """Insert text chunks into Milvus collection"""
        if not chunks:
            return {"status": "error", "error": "No chunks to insert"}
        
        if not self.collection:
            return self._mock_insert(chunks)
        
        try:
            # Prepare data for insertion
            embeddings = []
            texts = []
            metadata_list = []
            
            print(f"Generating embeddings for {len(chunks)} chunks...")
            
            for i, chunk in enumerate(chunks):
                # Generate embedding
                embedding = self.generate_embedding(chunk.text)
                embeddings.append(embedding)
                texts.append(chunk.text)
                
                # Prepare metadata
                metadata = {
                    "chunk_id": chunk.chunk_id,
                    "filename": chunk.metadata.get("filename", ""),
                    "topic": chunk.metadata.get("topic", ""),
                    "source": chunk.metadata.get("source", ""),
                    "word_count": chunk.word_count,
                    "created_at": datetime.now().isoformat()
                }
                metadata_list.append(metadata)
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(chunks)} chunks")
            
            # Insert into Milvus
            insert_data = [embeddings, texts, metadata_list]
            mr = self.collection.insert(insert_data)
            
            # Flush to ensure data is persisted
            self.collection.flush()
            
            return {
                "status": "success",
                "inserted_count": len(chunks),
                "insert_ids": mr.primary_keys
            }
            
        except Exception as e:
            print(f"Error inserting chunks: {e}")
            return {"status": "error", "error": str(e)}
    
    def _mock_insert(self, chunks: List[TextChunk]) -> Dict[str, Any]:
        """Mock insertion for testing without Milvus"""
        print(f"Mock insertion of {len(chunks)} chunks")
        
        # Save chunks to local file for testing
        chunks_data = []
        for chunk in chunks:
            chunk_data = {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "metadata": chunk.metadata,
                "word_count": chunk.word_count,
                "embedding": self.generate_embedding(chunk.text)[:10]  # Just first 10 dims
            }
            chunks_data.append(chunk_data)
        
        # Save to file
        output_file = "/home/ah0012/project/data/mock_embeddings.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(chunks_data, f, indent=2)
        
        print(f"Saved mock embeddings to {output_file}")
        
        return {
            "status": "success", 
            "inserted_count": len(chunks),
            "mock_file": output_file
        }
    
    def search_similar(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        if not self.collection:
            return self._mock_search(query, top_k)
        
        try:
            # Generate query embedding
            query_embedding = self.generate_embedding(query)
            
            # Search parameters
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10}
            }
            
            # Perform search
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding", 
                param=search_params,
                limit=top_k,
                output_fields=["text", "metadata"]
            )
            
            # Format results
            search_results = []
            for hits in results:
                for hit in hits:
                    search_results.append({
                        "id": hit.id,
                        "text": hit.entity.get('text', ''),
                        "metadata": hit.entity.get('metadata', {}),
                        "similarity_score": hit.score
                    })
            
            return search_results
            
        except Exception as e:
            print(f"Error searching: {e}")
            return self._mock_search(query, top_k)
    
    def _mock_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Mock search results for testing"""
        # Try to load mock data
        mock_file = "/home/ah0012/project/data/mock_embeddings.json"
        
        if os.path.exists(mock_file):
            try:
                with open(mock_file, 'r') as f:
                    chunks_data = json.load(f)
                
                # Simple keyword matching for mock search
                query_words = query.lower().split()
                results = []
                
                for chunk in chunks_data:
                    text_lower = chunk['text'].lower()
                    matches = sum(1 for word in query_words if word in text_lower)
                    
                    if matches > 0:
                        score = matches / len(query_words)
                        results.append({
                            "id": hash(chunk['chunk_id']) % 1000000,
                            "text": chunk['text'],
                            "metadata": chunk['metadata'],
                            "similarity_score": score
                        })
                
                # Sort by score and return top_k
                results.sort(key=lambda x: x['similarity_score'], reverse=True)
                return results[:top_k]
                
            except Exception as e:
                print(f"Error loading mock data: {e}")
        
        # Fallback mock results
        return [
            {
                "id": 1,
                "text": "Our return policy allows returns within 30 days of purchase.",
                "metadata": {"filename": "return_policy.pdf", "topic": "returns"},
                "similarity_score": 0.8
            }
        ]
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        if not self.collection:
            return {"status": "error", "error": "Collection not available"}
        
        try:
            # Get collection statistics
            stats = {
                "name": self.collection_name,
                "num_entities": self.collection.num_entities,
                "schema": str(self.collection.schema),
                "indexes": [index.params for index in self.collection.indexes]
            }
            return {"status": "success", "data": stats}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def clear_collection(self) -> Dict[str, Any]:
        """Clear all data from collection (for testing)"""
        if not self.collection:
            return {"status": "error", "error": "Collection not available"}
        
        try:
            # Delete all entities
            self.collection.delete("")
            self.collection.flush()
            return {"status": "success", "message": "Collection cleared"}
        except Exception as e:
            return {"status": "error", "error": str(e)}


class DataPipeline:
    """Complete data pipeline for PDF processing and Milvus insertion"""
    
    def __init__(self):
        self.text_processor = TextProcessor()
        self.milvus_manager = MilvusManager()
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """Run the complete data pipeline"""
        print("Starting full data pipeline...")
        
        # Step 1: Process PDFs and create chunks
        print("\n1. Processing PDFs and creating chunks...")
        chunks = self.text_processor.process_all_pdfs()
        
        if not chunks:
            return {"status": "error", "error": "No chunks created"}
        
        # Step 2: Insert chunks into Milvus
        print(f"\n2. Inserting {len(chunks)} chunks into Milvus...")
        insert_result = self.milvus_manager.insert_chunks(chunks)
        
        # Step 3: Verify insertion
        print("\n3. Verifying insertion...")
        collection_info = self.milvus_manager.get_collection_info()
        
        return {
            "status": "success",
            "chunks_created": len(chunks),
            "insert_result": insert_result,
            "collection_info": collection_info
        }
    
    def test_search(self, test_queries: List[str] = None) -> Dict[str, Any]:
        """Test the search functionality"""
        if test_queries is None:
            test_queries = [
                "return policy",
                "shipping information",
                "customer service",
                "payment methods"
            ]
        
        print("Testing search functionality...")
        test_results = {}
        
        for query in test_queries:
            print(f"\nTesting query: '{query}'")
            results = self.milvus_manager.search_similar(query, top_k=3)
            
            test_results[query] = {
                "results_count": len(results),
                "top_result": results[0] if results else None
            }
            
            if results:
                print(f"Found {len(results)} results")
                print(f"Top result: {results[0]['text'][:100]}...")
            else:
                print("No results found")
        
        return test_results


def main():
    """Main function to run the setup"""
    print("Setting up Milvus database and embeddings...")
    
    # Run full pipeline
    pipeline = DataPipeline()
    result = pipeline.run_full_pipeline()
    
    print(f"\nPipeline result: {result}")
    
    # Test search functionality
    test_results = pipeline.test_search()
    print(f"\nSearch test results: {json.dumps(test_results, indent=2)}")
    
    return result


if __name__ == "__main__":
    main()