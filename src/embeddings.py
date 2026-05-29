# src/embeddings.py
"""
Embedding generation module for converting text to vectors
"""
import logging
from typing import List, Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings, OpenAIEmbeddings

from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Manages embedding generation for text documents
    """
    
    def __init__(
        self,
        embedding_type: str = "huggingface",
        model_name: Optional[str] = None
    ):
        """
        Initialize EmbeddingGenerator
        
        Args:
            embedding_type: Type of embeddings ("huggingface" or "openai")
            model_name: Specific model name to use
        """
        self.embedding_type = embedding_type
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.embeddings = None
        
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize the embedding model"""
        try:
            if self.embedding_type == "huggingface":
                logger.info(f"Initializing HuggingFace embeddings: {self.model_name}")
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=self.model_name,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                
            elif self.embedding_type == "openai":
                if not settings.OPENAI_API_KEY:
                    raise ValueError("OpenAI API key not found")
                
                logger.info("Initializing OpenAI embeddings")
                self.embeddings = OpenAIEmbeddings(
                    openai_api_key=settings.OPENAI_API_KEY,
                    model="text-embedding-ada-002"
                )
            
            else:
                raise ValueError(f"Unknown embedding type: {self.embedding_type}")
            
            logger.info("Embedding model initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing embeddings: {str(e)}")
            raise
    
    def get_embeddings(self):
        """
        Get the embedding model instance
        
        Returns:
            Embeddings object
        """
        return self.embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query
        
        Args:
            text: Query text
            
        Returns:
            Embedding vector
        """
        try:
            embedding = self.embeddings.embed_query(text)
            return embedding
        except Exception as e:
            logger.error(f"Error embedding query: {str(e)}")
            raise
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple documents
        
        Args:
            texts: List of document texts
            
        Returns:
            List of embedding vectors
        """
        try:
            embeddings = self.embeddings.embed_documents(texts)
            logger.info(f"Generated embeddings for {len(texts)} documents")
            return embeddings
        except Exception as e:
            logger.error(f"Error embedding documents: {str(e)}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors
        
        Returns:
            Embedding dimension
        """
        test_embedding = self.embed_query("test")
        return len(test_embedding)