# app.py
"""
Main Streamlit application for CyberSec AI RAG System
"""
import streamlit as st
import logging
from pathlib import Path
from typing import List

from config.settings import settings
from src.document_loader import CyberSecDocumentProcessor
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStoreManager
from src.retrieval import RetrieverManager
from src.llm_chain import LLMChainManager
from utils.helpers import (
    format_response_with_sources,
    extract_threat_entities,
    validate_query
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title=settings.PAGE_TITLE,
    page_icon=settings.PAGE_ICON,
    layout=settings.LAYOUT,
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .threat-badge {
        background-color: #FFE66D;
        padding: 0.3rem 0.6rem;
        border-radius: 0.3rem;
        margin: 0.2rem;
        display: inline-block;
    }
    .source-box {
        background-color: #F0F0F0;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("""
<h1 class="main-header">🛡️ CyberSec AI - Threat Intelligence RAG System</h1>
<p style="text-align: center; color: #666;">
    Intelligent analysis of cyber security threats using AI and document retrieval
</p>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'vector_store_manager' not in st.session_state:
    st.session_state.vector_store_manager = None

if 'llm_chain' not in st.session_state:
    st.session_state.llm_chain = None

if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration & Upload")
    
    # API Key
    st.subheader("🔑 API Configuration")
    api_key = st.text_input(
        "OpenAI API Key",
        value=settings.OPENAI_API_KEY or "",
        type="password",
        help="Enter your OpenAI API key"
    )
    
    if api_key:
        settings.OPENAI_API_KEY = api_key
    
    st.divider()
    
    # Document Upload
    st.subheader("📄 Document Upload")
    uploaded_files = st.file_uploader(
        "Upload cyber security documents",
        type=['pdf', 'txt', 'docx', 'md'],
        accept_multiple_files=True,
        help="Upload documents about cyber security threats"
    )
    
    if uploaded_files:
        if st.button("📤 Process Documents", type="primary", use_container_width=True):
            with st.spinner("Processing documents..."):
                try:
                    # Initialize processor
                    processor = CyberSecDocumentProcessor(
                        chunk_size=settings.CHUNK_SIZE,
                        chunk_overlap=settings.CHUNK_OVERLAP
                    )
                    
                    all_documents = []
                    
                    # Process each uploaded file
                    for uploaded_file in uploaded_files:
                        docs = processor.load_from_uploaded_file(uploaded_file)
                        all_documents.extend(docs)
                        st.success(f"✅ Loaded {uploaded_file.name}")
                    
                    if all_documents:
                        # Extract threat information
                        all_documents = processor.extract_threat_info(all_documents)
                        
                        # Create vector store
                        embedding_gen = EmbeddingGenerator(embedding_type="huggingface")
                        vs_manager = VectorStoreManager(
                            store_type="faiss",
                            embedding_generator=embedding_gen
                        )
                        
                        st.info(f"Creating vector store with {len(all_documents)} chunks...")
                        vs_manager.create_vector_store(all_documents)
                        vs_manager.save_vector_store()
                        
                        # Save to session state
                        st.session_state.vector_store_manager = vs_manager
                        st.session_state.documents_loaded = True
                        
                        # Initialize LLM chain
                        retriever_manager = RetrieverManager(vs_manager)
                        llm_chain = LLMChainManager(retriever_manager)
                        st.session_state.llm_chain = llm_chain
                        
                        st.success(f"✅ Successfully processed {len(all_documents)} document chunks!")
                        st.info("🎉 Ready for querying!")
                    else:
                        st.error("No documents were loaded")
                        
                except Exception as e:
                    st.error(f"❌ Error processing documents: {str(e)}")
                    logger.error(f"Error: {str(e)}")
    
    st.divider()
    
    # Status
    st.subheader("📊 System Status")
    if st.session_state.documents_loaded:
        st.success("✅ Documents loaded and indexed")
    else:
        st.warning("⏳ Waiting for documents to be uploaded")
    
    # Clear chat history
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main content area
if not st.session_state.documents_loaded:
    st.info("""
    👋 Welcome to CyberSec AI!
    
    To get started:
    1. Enter your **OpenAI API Key** in the sidebar
    2. **Upload** cyber security threat documents (PDF, TXT, DOCX, MD)
    3. Click **Process Documents** to create the vector database
    4. Start asking questions about threats!
    
    ### Features:
    - 📚 Process multiple document formats
    - 🔍 Semantic search over your documents
    - 💬 Interactive chat interface
    - 📊 Threat detection and analysis
    - 🎯 Source citation and verification
    """)
else:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if "sources" in message and message["sources"]:
                with st.expander("📚 View Sources"):
                    st.markdown(message["sources"])
    
    # Chat input
    if prompt := st.chat_input("Ask about cyber security threats..."):
        # Validate query
        if not validate_query(prompt):
            st.error("⚠️ Please enter a valid query (at least 3 characters)")
        elif not st.session_state.llm_chain:
            st.error("❌ LLM chain not initialized. Please upload documents first.")
        else:
            # Add user message to chat
            st.session_state.messages.append({
                "role": "user",
                "content": prompt
            })
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("🤔 Analyzing threat information..."):
                    try:
                        # Get response from LLM chain
                        response = st.session_state.llm_chain.query(prompt)
                        
                        # Format response
                        answer, sources = format_response_with_sources(response)
                        
                        # Display answer
                        st.markdown(answer)
                        
                        # Extract and display threats
                        threats = extract_threat_entities(answer + " " + prompt)
                        if threats:
                            st.markdown("### 🚨 Threats Detected:")
                            threat_html = " ".join([
                                f'<span class="threat-badge">{threat.upper()}</span>'
                                for threat in threats
                            ])
                            st.markdown(threat_html, unsafe_allow_html=True)
                        
                        # Display sources
                        if sources:
                            with st.expander("📚 View Sources"):
                                st.markdown(sources)
                        
                        # Save to chat history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        })
                        
                    except Exception as e:
                        st.error(f"❌ Error generating response: {str(e)}")
                        logger.error(f"Error: {str(e)}")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #888; margin-top: 2rem;">
    <p>🛡️ CyberSec AI RAG System | Built with LangChain & Streamlit</p>
    <p>Intern: Viplav | Organization: Viswam AI</p>
    <p>Version: 1.0.0</p>
</div>
""", unsafe_allow_html=True)