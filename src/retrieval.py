# src/retrieval.py
"""
Retrieval system for finding relevant documents
"""
import logging
from typing import List, Optional

from langchain.schema import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.llms import OpenAI

from config.settings import settings
from src.vector_store import VectorStoreManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetrieverManager:
    """
    Manages document retrieval with various strategies
    """
    
    def __init__(self, vector_store_manager: VectorStoreManager):
        """
        Initialize RetrieverManager
        
        Args:
            vector_store_manager: VectorStoreManager instance
        """
        self.vector_store_manager = vector_store_manager
        self.vector_store = vector_store_manager.get_vector_store()
        
        logger.info("RetrieverManager initialized")
    
    def get_retriever(self, k: int = settings.RETRIEVAL_K):
        """
        Get basic retriever
        
        Args:
            k: Number of documents to retrieve
            
        Returns:
            Retriever object
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": k}
        )
        
        logger.info(f"Created retriever with k={k}")
        return retriever
    
    def get_compression_retriever(
        self,
        k: int = settings.RETRIEVAL_K,
        llm_kwargs: Optional[dict] = None
    ):
        """
        Get retriever with contextual compression
        
        Args:
            k: Number of documents to retrieve
            llm_kwargs: LLM configuration
            
        Returns:
            Compressed retriever object
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        try:
            # Create LLM for compression
            llm = OpenAI(
                temperature=0,
                openai_api_key=settings.OPENAI_API_KEY,
                **(llm_kwargs or {})
            )
            
            # Create compressor
            compressor = LLMChainExtractor.from_llm(llm)
            
            # Create compression retriever
            retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=self.vector_store.as_retriever(
                    search_kwargs={"k": k * 2}
                )
            )
            
            logger.info("Created compression retriever")
            return retriever
            
        except Exception as e:
            logger.error(f"Error creating compression retriever: {str(e)}")
            # Fall back to basic retriever
            return self.get_retriever(k)
    
    def retrieve_documents(
        self,
        query: str,
        k: int = settings.RETRIEVAL_K
    ) -> List[Document]:
        """
        Retrieve documents for a query
        
        Args:
            query: Search query
            k: Number of documents
            
        Returns:
            List of relevant documents
        """
        try:
            retriever = self.get_retriever(k)
            documents = retriever.get_relevant_documents(query)
            
            logger.info(f"Retrieved {len(documents)} documents for query: {query[:50]}")
            return documents
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            raise
    
    def retrieve_with_scores(
        self,
        query: str,
        k: int = settings.RETRIEVAL_K
    ) -> List[tuple]:
        """
        Retrieve documents with relevance scores
        
        Args:
            query: Search query
            k: Number of documents
            
        Returns:
            List of (Document, score) tuples
        """
        try:
            results = self.vector_store_manager.similarity_search_with_score(
                query=query,
                k=k
            )
            
            logger.info(f"Retrieved {len(results)} documents with scores")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving with scores: {str(e)}")
            raise