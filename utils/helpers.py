# utils/helpers.py
"""
Helper functions for the RAG system
"""
import logging
from typing import List
from langchain.schema import Document

logger = logging.getLogger(__name__)


def format_documents_for_display(documents: List[Document]) -> str:
    """
    Format documents for display in UI
    
    Args:
        documents: List of Document objects
        
    Returns:
        Formatted string
    """
    if not documents:
        return "No documents found."
    
    formatted = "📚 **Retrieved Documents:**\n\n"
    
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get('source', 'Unknown source')
        page = doc.metadata.get('page', None)
        
        formatted += f"**Document {i}** (Source: {source}"
        if page is not None:
            formatted += f", Page: {page}"
        formatted += f")\n\n{doc.page_content[:300]}...\n\n"
    
    return formatted


def format_response_with_sources(response: dict) -> tuple:
    """
    Format LLM response with sources
    
    Args:
        response: Response dict from LLM chain
        
    Returns:
        Tuple of (answer, sources)
    """
    answer = response.get('result', response.get('answer', 'No answer generated'))
    source_documents = response.get('source_documents', [])
    
    sources_text = ""
    if source_documents:
        sources_text = format_documents_for_display(source_documents)
    
    return answer, sources_text


def extract_threat_entities(text: str) -> List[str]:
    """
    Extract threat-related entities from text
    
    Args:
        text: Input text
        
    Returns:
        List of threat entities
    """
    threat_keywords = [
        'malware', 'ransomware', 'phishing', 'ddos',
        'vulnerability', 'exploit', 'breach', 'attack',
        'threat', 'CVE', 'zero-day', 'trojan', 'botnet',
        'worm', 'virus', 'spyware', 'intrusion'
    ]
    
    text_lower = text.lower()
    found_threats = [keyword for keyword in threat_keywords if keyword in text_lower]
    
    return found_threats


def validate_query(query: str, min_length: int = 3) -> bool:
    """
    Validate user query
    
    Args:
        query: User query
        min_length: Minimum query length
        
    Returns:
        True if valid, False otherwise
    """
    if not query or len(query.strip()) < min_length:
        return False
    return True


def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to max length
    
    Args:
        text: Input text
        max_length: Maximum length
        
    Returns:
        Truncated text with ellipsis
    """
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."