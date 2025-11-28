"""
LLM Processor Factory with singleton pattern for (provider, model) pairs
"""
import os
from typing import Dict, Tuple, Optional, Any
from abc import ABC, abstractmethod
import openai
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()


class LLMProcessor(ABC):
    """Abstract base class for LLM processors"""
    
    def __init__(self, model: str):
        self.model = model
    
    @abstractmethod
    def generate_completion(self, messages: list, **kwargs) -> str:
        """Generate a completion from messages"""
        pass
    
    @abstractmethod
    def generate_embedding(self, text: str) -> list:
        """Generate embeddings for text"""
        pass


class OpenAIProcessor(LLMProcessor):
    """OpenAI LLM Processor"""
    
    def __init__(self, model: str):
        super().__init__(model)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your_openai_key_here":
            print("Warning: No valid OpenAI API key found. Using mock responses.")
            self.client = None
        else:
            self.client = openai.OpenAI(api_key=api_key)
    
    def generate_completion(self, messages: list, **kwargs) -> str:
        """Generate completion using OpenAI"""
        if not self.client:
            # Return mock response when API is not available
            last_message = messages[-1].get("content", "")
            if "return policy" in last_message.lower():
                return "Our return policy allows returns within 30 days of purchase. Items must be in original condition with tags attached. Refunds are processed within 5-7 business days."
            elif "shipping" in last_message.lower():
                return "We offer standard shipping (5-7 business days) and expedited shipping (2-3 business days). Free shipping on orders over $50."
            elif "order" in last_message.lower():
                return "You can track your order using the order number and email address provided at checkout. Orders typically ship within 1-2 business days."
            else:
                return "Thank you for your question. I'm here to help with information about our policies, orders, returns, and products."
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API error: {str(e)}")
            return "I apologize, but I'm unable to process your request at the moment. Please try again later or contact customer support."
    
    def generate_embedding(self, text: str) -> list:
        """Generate embeddings using OpenAI"""
        if not self.client:
            # Return mock embedding when API is not available
            import hashlib
            hash_obj = hashlib.md5(text.encode())
            hash_int = int(hash_obj.hexdigest()[:8], 16)
            # Generate a deterministic 384-dimensional embedding
            embedding = [(hash_int >> i) % 2 - 0.5 for i in range(384)]
            return embedding
        
        try:
            response = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"OpenAI embedding error: {str(e)}")
            # Fallback to mock embedding
            import hashlib
            hash_obj = hashlib.md5(text.encode())
            hash_int = int(hash_obj.hexdigest()[:8], 16)
            embedding = [(hash_int >> i) % 2 - 0.5 for i in range(384)]
            return embedding


class AnthropicProcessor(LLMProcessor):
    """Anthropic LLM Processor"""
    
    def __init__(self, model: str):
        super().__init__(model)
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key or api_key == "your_anthropic_api_key_here":
            print("Warning: No valid Anthropic API key found. Using mock responses.")
            self.client = None
        else:
            self.client = Anthropic(api_key=api_key)
    
    def generate_completion(self, messages: list, **kwargs) -> str:
        """Generate completion using Anthropic"""
        if not self.client:
            # Return mock response when API is not available
            last_message = messages[-1].get("content", "")
            if "return policy" in last_message.lower():
                return "Our return policy allows returns within 30 days of purchase. Items must be in original condition with tags attached. Refunds are processed within 5-7 business days."
            elif "shipping" in last_message.lower():
                return "We offer standard shipping (5-7 business days) and expedited shipping (2-3 business days). Free shipping on orders over $50."
            elif "order" in last_message.lower():
                return "You can track your order using the order number and email address provided at checkout. Orders typically ship within 1-2 business days."
            else:
                return "Thank you for your question. I'm here to help with information about our policies, orders, returns, and products."
        
        try:
            # Convert OpenAI-style messages to Anthropic format
            system_message = None
            formatted_messages = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    formatted_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            
            response = self.client.messages.create(
                model=self.model,
                messages=formatted_messages,
                system=system_message,
                max_tokens=kwargs.get("max_tokens", 1000),
                **{k: v for k, v in kwargs.items() if k != "max_tokens"}
            )
            return response.content[0].text
        except Exception as e:
            print(f"Anthropic API error: {str(e)}")
            return "I apologize, but I'm unable to process your request at the moment. Please try again later or contact customer support."
    
    def generate_embedding(self, text: str) -> list:
        """Anthropic doesn't provide embeddings, fallback to sentence-transformers or mock"""
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embedding = model.encode(text)
            return embedding.tolist()
        except Exception as e:
            print(f"Sentence transformer error: {e}")
            # Fallback to mock embedding
            import hashlib
            hash_obj = hashlib.md5(text.encode())
            hash_int = int(hash_obj.hexdigest()[:8], 16)
            embedding = [(hash_int >> i) % 2 - 0.5 for i in range(384)]
            return embedding


class LLMProcessorFactory:
    """Factory for creating and managing LLM processors with singleton pattern"""
    
    _instances: Dict[Tuple[str, str], LLMProcessor] = {}
    
    @classmethod
    def get_processor(cls, provider: str, model: str) -> LLMProcessor:
        """
        Get or create an LLM processor instance.
        Uses singleton pattern - only one instance per (provider, model) pair.
        """
        key = (provider.lower(), model)
        
        if key not in cls._instances:
            cls._instances[key] = cls._create_processor(provider, model)
        
        return cls._instances[key]
    
    @classmethod
    def _create_processor(cls, provider: str, model: str) -> LLMProcessor:
        """Create a new processor instance"""
        provider = provider.lower()
        
        if provider == "openai":
            return OpenAIProcessor(model)
        elif provider == "anthropic":
            return AnthropicProcessor(model)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    @classmethod
    def list_instances(cls) -> Dict[Tuple[str, str], LLMProcessor]:
        """Get all active processor instances"""
        return cls._instances.copy()
    
    @classmethod
    def clear_instances(cls):
        """Clear all instances (for testing)"""
        cls._instances.clear()


# Convenience function for getting default processor
def get_default_processor() -> LLMProcessor:
    """Get default LLM processor from environment"""
    provider = os.getenv("DEFAULT_PROVIDER", "openai")
    model = os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo")
    return LLMProcessorFactory.get_processor(provider, model)