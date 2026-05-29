"""
Document retrieval module using FAISS vector store
"""
import logging
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

logger = logging.getLogger(__name__)

class DocumentRetriever:
    """Retrieve relevant documents using semantic similarity"""
    
    def __init__(self, embeddings_object):
        """
        Initialize retriever with embeddings
        
        Args:
            embeddings_object: HuggingFace embeddings object
        """
        self.embeddings = embeddings_object
        self.vector_store = None
    
    def create_vector_store(self, documents: List[Document]) -> None:
        """Create FAISS vector store from documents"""
        try:
            self.vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            logger.info(f"Created vector store with {len(documents)} documents")
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise
    
    def retrieve_documents(self, query: str, k: int = 3) -> List[Document]:
        """
        Retrieve top-k most relevant documents for a query
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        try:
            results = self.vector_store.similarity_search(query, k=k)
            logger.info(f"Retrieved {len(results)} documents for query: {query[:50]}")
            return results
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            raise
    
    def retrieve_with_scores(self, query: str, k: int = 3) -> List[Tuple[Document, float]]:
        """Retrieve documents with similarity scores"""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            return results
        except Exception as e:
            logger.error(f"Error retrieving documents with scores: {str(e)}")
            raise