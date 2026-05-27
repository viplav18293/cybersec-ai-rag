# src/vector_store.py
"""
Vector store management for efficient similarity search
"""
import logging
from typing import List, Optional
from pathlib import Path

from langchain.vectorstores import FAISS, Chroma
from langchain.schema import Document

from config.settings import settings
from src.embeddings import EmbeddingGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Manages vector database operations for document retrieval
    """
    
    def __init__(
        self,
        store_type: str = "faiss",
        embedding_generator: Optional[EmbeddingGenerator] = None,
        persist_directory: Optional[str] = None
    ):
        """
        Initialize VectorStoreManager
        
        Args:
            store_type: Type of vector store ("faiss" or "chroma")
            embedding_generator: EmbeddingGenerator instance
            persist_directory: Directory to persist vector store
        """
        self.store_type = store_type
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        self.persist_directory = persist_directory or str(settings.VECTOR_STORE_DIR)
        self.vector_store = None
        
        logger.info(f"VectorStoreManager initialized: type={store_type}")
    
    def create_vector_store(self, documents: List[Document]) -> None:
        """
        Create vector store from documents
        
        Args:
            documents: List of Document objects
        """
        try:
            embeddings = self.embedding_generator.get_embeddings()
            
            if self.store_type == "faiss":
                logger.info("Creating FAISS vector store...")
                self.vector_store = FAISS.from_documents(
                    documents=documents,
                    embedding=embeddings
                )
                
            elif self.store_type == "chroma":
                logger.info("Creating Chroma vector store...")
                self.vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=embeddings,
                    persist_directory=self.persist_directory
                )
                self.vector_store.persist()
            
            else:
                raise ValueError(f"Unknown vector store type: {self.store_type}")
            
            logger.info(f"Vector store created with {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise
    
    def save_vector_store(self, path: Optional[str] = None) -> None:
        """
        Save vector store to disk
        
        Args:
            path: Custom save path (optional)
        """
        try:
            save_path = path or self.persist_directory
            
            if self.store_type == "faiss":
                self.vector_store.save_local(save_path)
                logger.info(f"FAISS vector store saved to {save_path}")
                
            elif self.store_type == "chroma":
                # Chroma auto-persists
                logger.info(f"Chroma vector store already persisted at {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            raise
    
    def load_vector_store(self, path: Optional[str] = None) -> None:
        """
        Load vector store from disk
        
        Args:
            path: Custom load path (optional)
        """
        try:
            load_path = path or self.persist_directory
            embeddings = self.embedding_generator.get_embeddings()
            
            if self.store_type == "faiss":
                if Path(f"{load_path}/index.faiss").exists():
                    self.vector_store = FAISS.load_local(
                        load_path,
                        embeddings,
                        allow_dangerous_deserialization=True
                    )
                    logger.info(f"FAISS vector store loaded from {load_path}")
                else:
                    logger.warning(f"No FAISS index found at {load_path}")
                    
            elif self.store_type == "chroma":
                if Path(load_path).exists():
                    self.vector_store = Chroma(
                        persist_directory=load_path,
                        embedding_function=embeddings
                    )
                    logger.info(f"Chroma vector store loaded from {load_path}")
                else:
                    logger.warning(f"No Chroma DB found at {load_path}")
            
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            raise
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add new documents to existing vector store
        
        Args:
            documents: List of Document objects to add
        """
        try:
            if self.vector_store is None:
                logger.warning("No vector store exists. Creating new one...")
                self.create_vector_store(documents)
            else:
                self.vector_store.add_documents(documents)
                logger.info(f"Added {len(documents)} documents to vector store")
                
                if self.store_type == "chroma":
                    self.vector_store.persist()
                    
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[dict] = None
    ) -> List[Document]:
        """
        Perform similarity search
        
        Args:
            query: Search query
            k: Number of results to return
            filter: Metadata filter (optional)
            
        Returns:
            List of similar documents
        """
        try:
            if self.vector_store is None:
                raise ValueError("No vector store loaded")
            
            results = self.vector_store.similarity_search(
                query=query,
                k=k,
                filter=filter
            )
            
            logger.info(f"Similarity search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            raise
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5
    ) -> List[tuple]:
        """
        Perform similarity search with relevance scores
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of (Document, score) tuples
        """
        try:
            if self.vector_store is None:
                raise ValueError("No vector store loaded")
            
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k
            )
            
            logger.info(f"Similarity search with scores returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search with scores: {str(e)}")
            raise
    
    def get_vector_store(self):
        """
        Get the vector store instance
        
        Returns:
            Vector store object
        """
        return self.vector_store