# src/document_loader.py
"""
Document loading and processing module for cyber security documents
"""
import os
from typing import List, Optional
from pathlib import Path
import logging
import tempfile

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

from config.settings import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Handles loading and processing of various document formats
    for cyber security threat intelligence
    """
    
    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP
    ):
        """
        Initialize DocumentProcessor
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        logger.info(f"DocumentProcessor initialized: chunk_size={chunk_size}, overlap={chunk_overlap}")
    
    def load_document(self, file_path: str) -> List[Document]:
        """
        Load a single document based on file extension
        
        Args:
            file_path: Path to the document
            
        Returns:
            List of Document objects
        """
        file_extension = Path(file_path).suffix.lower()
        
        try:
            if file_extension == ".pdf":
                loader = PyPDFLoader(file_path)
            elif file_extension == ".txt":
                loader = TextLoader(file_path, encoding='utf-8')
            elif file_extension == ".docx":
                loader = Docx2txtLoader(file_path)
            elif file_extension == ".md":
                loader = UnstructuredMarkdownLoader(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} pages from {Path(file_path).name}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            raise
    
    def load_multiple_documents(self, file_paths: List[str]) -> List[Document]:
        """
        Load multiple documents
        
        Args:
            file_paths: List of file paths
            
        Returns:
            Combined list of Document objects
        """
        all_documents = []
        
        for file_path in file_paths:
            try:
                docs = self.load_document(file_path)
                all_documents.extend(docs)
            except Exception as e:
                logger.warning(f"Skipping {file_path} due to error: {str(e)}")
                continue
        
        logger.info(f"Total documents loaded: {len(all_documents)}")
        return all_documents
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of chunked Document objects
        """
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Split into {len(chunks)} chunks")
        return chunks
    
    def load_and_process(self, file_paths: List[str]) -> List[Document]:
        """
        Complete pipeline: load and process documents
        
        Args:
            file_paths: List of file paths
            
        Returns:
            Processed document chunks
        """
        logger.info("Starting document loading and processing pipeline...")
        
        # Load documents
        documents = self.load_multiple_documents(file_paths)
        
        if not documents:
            logger.warning("No documents loaded")
            return []
        
        # Process into chunks
        chunks = self.process_documents(documents)
        
        logger.info(f"Pipeline complete: {len(chunks)} chunks ready")
        return chunks
    
    def add_metadata(self, documents: List[Document], metadata: dict) -> List[Document]:
        """
        Add custom metadata to documents
        
        Args:
            documents: List of Document objects
            metadata: Dictionary of metadata to add
            
        Returns:
            Documents with updated metadata
        """
        for doc in documents:
            doc.metadata.update(metadata)
        
        return documents
    
    def load_from_uploaded_file(self, uploaded_file) -> List[Document]:
        """
        Load document from Streamlit uploaded file
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            List of Document objects
        """
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'_{uploaded_file.name}') as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        try:
            documents = self.load_document(tmp_path)
            return documents
        finally:
            # Clean up temporary file
            os.unlink(tmp_path)


class CyberSecDocumentProcessor(DocumentProcessor):
    """
    Specialized document processor for cyber security content
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Cyber security specific separators
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[
                "\n## ",  # Markdown headers
                "\n### ",
                "\n\n",
                "\n",
                ". ",
                "! ",
                "? ",
                " ",
                ""
            ]
        )
    
    def extract_threat_info(self, documents: List[Document]) -> List[Document]:
        """
        Extract and tag threat-related information
        
        Args:
            documents: List of Document objects
            
        Returns:
            Documents with threat metadata
        """
        threat_keywords = [
            'malware', 'ransomware', 'phishing', 'ddos', 'distributed denial',
            'vulnerability', 'exploit', 'breach', 'attack', 'attacker',
            'threat', 'security', 'CVE', 'zero-day', 'zero day',
            'trojan', 'botnet', 'worm', 'virus', 'spyware',
            'intrusion', 'unauthorized access', 'data exfiltration',
            'encryption', 'decryption', 'authentication', 'credential',
            'network', 'firewall', 'antivirus', 'endpoint', 'detection'
        ]
        
        for doc in documents:
            content_lower = doc.page_content.lower()
            
            # Tag documents with threat types
            detected_threats = [
                keyword for keyword in threat_keywords
                if keyword in content_lower
            ]
            
            doc.metadata['threat_keywords'] = detected_threats
            doc.metadata['threat_count'] = len(detected_threats)
            doc.metadata['is_threat_related'] = len(detected_threats) > 0
        
        return documents