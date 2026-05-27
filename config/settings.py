# config/settings.py
"""
Configuration settings for CyberSec AI RAG System
"""
import os
from typing import Optional
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

class Settings:
    """Application settings and configuration"""
    
    # Project Info
    PROJECT_NAME: str = "CyberSec AI - Threat Intelligence RAG"
    VERSION: str = "1.0.0"
    AUTHOR: str = "Viplav - Viswam AI Intern"
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    VECTOR_STORE_DIR: Path = BASE_DIR / "vector_store"
    
    # API Keys
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    HUGGINGFACE_API_KEY: Optional[str] = os.getenv("HUGGINGFACE_API_KEY")
    
    # LangChain Settings
    LANGCHAIN_TRACING_V2: bool = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    LANGCHAIN_API_KEY: Optional[str] = os.getenv("LANGCHAIN_API_KEY")
    LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "cybersecai-rag")
    
    # Document Processing
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Retrieval Settings
    RETRIEVAL_K: int = int(os.getenv("RETRIEVAL_K", "5"))
    SEARCH_TYPE: str = "similarity"
    
    # LLM Settings
    MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.3"))
    MAX_TOKENS: int = 1000
    
    # Embedding Settings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    
    # Supported File Types
    SUPPORTED_EXTENSIONS: list = [".pdf", ".txt", ".docx", ".md"]
    
    # Streamlit Settings
    PAGE_TITLE: str = "CyberSec AI Threat Intelligence"
    PAGE_ICON: str = "🛡️"
    LAYOUT: str = "wide"
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        directories = [
            cls.DATA_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.VECTOR_STORE_DIR
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def validate_api_keys(cls) -> bool:
        """Validate that required API keys are set"""
        if not cls.OPENAI_API_KEY:
            return False
        return True
    
    def __repr__(self):
        return f"<Settings(project={self.PROJECT_NAME}, version={self.VERSION})>"


# Initialize settings
settings = Settings()
settings.create_directories()