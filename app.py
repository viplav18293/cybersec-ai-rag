 import streamlit as st
import logging
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="CyberSec AI", page_icon="🛡️", layout="wide")

st.markdown("<h1 style='text-align: center; color: #FF6B6B;'>🛡️ CyberSec AI - Threat Intelligence RAG</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>AI-Powered Cyber Security Threat Analysis</p>", unsafe_allow_html=True)

if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'docs_loaded' not in st.session_state:
    st.session_state.docs_loaded = False

with st.sidebar:
    st.header("⚙️ Configuration")
    st.success("✅ Using Smart Document Search (No API Needed!)")
    
    st.divider()
    st.subheader("📄 Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Choose cyber security documents",
        type=['pdf', 'txt'],
        accept_multiple_files=True,
        help="Upload PDF or TXT files about cyber security"
    )
    
    if uploaded_files:
        if st.button("🚀 Process Documents", type="primary", use_container_width=True):
            with st.spinner("📥 Processing documents..."):
                try:
                    all_documents = []
                    
                    for uploaded_file in uploaded_files:
                        st.write(f"📖 Processing {uploaded_file.name}...")
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f'_{uploaded_file.name}') as tmp:
                            tmp.write(uploaded_file.getvalue())
                            tmp_path = tmp.name
                        
                        try:
                            if uploaded_file.name.lower().endswith('.pdf'):
                                loader = PyPDFLoader(tmp_path)
                            else:
                                loader = TextLoader(tmp_path, encoding='utf-8')
                            
                            docs = loader.load()
                            all_documents.extend(docs)
                            st.success(f"✅ Loaded {uploaded_file.name}")
                        except Exception as file_error:
                            st.error(f"❌ Error loading {uploaded_file.name}: {str(file_error)}")
                        finally:
                            try:
                                os.unlink(tmp_path)
                            except:
                                pass
                    
                    if all_documents:
                        st.info(f"📊 Total documents loaded: {len(all_documents)}")
                        
                        st.write("✂️ Splitting into chunks...")
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=1000,
                            chunk_overlap=200,
                            separators=["\n\n", "\n", ". ", " ", ""]
                        )
                        chunks = text_splitter.split_documents(all_documents)
                        st.success(f"✂️ Created {len(chunks)} chunks")
                        
                        st.write("🧮 Creating embeddings...")
                        embeddings = HuggingFaceEmbeddings(
                            model_name="sentence-transformers/all-MiniLM-L6-v2",
                            model_kwargs={'device': 'cpu'},
                            encode_kwargs={'normalize_embeddings': True}
                        )
                        st.success("✅ Embedding model loaded")
                        
                        st.write("📦 Building vector store...")
                        st.session_state.vectorstore = FAISS.from_documents(
                            documents=chunks,
                            embedding=embeddings
                        )
                        st.success("✅ Vector store created")
                        
                        st.session_state.docs_loaded = True
                        st.success(f"🎉 Ready! Successfully indexed {len(chunks)} chunks")
                        st.balloons()
                    else:
                        st.error("❌ No documents were processed successfully")
                        
                except Exception as e:
                    st.error(f"❌ Processing error: {str(e)}")
                    logger.error(f"Processing error: {str(e)}", exc_info=True)
    
    st.divider()
    
    # Status
    if st.session_state.docs_loaded:
        st.success("✅ Documents Indexed & Ready")
        st.write("💬 You can now ask questions!")
    else:
        st.info("⏳ Upload documents to get started")
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main content
if not st.session_state.docs_loaded:
    st.markdown("""
    ### 👋 Welcome to CyberSec AI!
    
    **🆓 Completely FREE - No API Keys Needed!**
    
    #### Features:
    - ✅ Upload cyber security documents (PDF/TXT)
    - ✅ AI-powered semantic search
    - ✅ Source citations and verification
    - ✅ Intelligent document analysis
    - ✅ No costs, no API key required
    
    #### How to use:
    1. **📄 Upload Documents** - Cyber security PDFs or text files
    2. **🚀 Process** - Click the process button and wait
    3. **💬 Ask Questions** - Chat about threats, attacks, prevention
    4. **📚 View Sources** - See where information comes from
    
    #### Example Questions:
    - What is ransomware and how does it work?
    - How can organizations prevent phishing attacks?
    - What are the different types of DDoS attacks?
    - What are the best cyber security practices?
    - Compare ransomware and phishing attacks
    
    #### Technology Stack:
    - **🔍 Search:** Semantic similarity with embeddings
    - **📊 Processing:** LangChain + FAISS vector store
    - **🎨 Interface:** Streamlit (this web app)
    - **⚡ Embeddings:** HuggingFace Sentence Transformers
    """)
    
else:
    st.subheader("💬 Chat with Your Documents")
    st.write(f"📚 **Documents loaded** | 💬 **{len(st.session_state.messages)//2}** conversations")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if "sources" in message and message["sources"]:
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.write(f"**Source {i}:**")
                        st.write(source)
                        st.divider()
    
    # Chat input
    if prompt := st.chat_input("Ask about cyber security threats..."):
        
        if not st.session_state.vectorstore:
            st.error("❌ Please upload and process documents first!")
        else:
            # Add user message
            st.session_state.messages.append({
                "role": "user",
                "content": prompt
            })
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("🔍 Searching documents..."):
                    try:
                        # Retrieve relevant documents
                        retriever = st.session_state.vectorstore.as_retriever(
                            search_kwargs={"k": 5}
                        )
                        docs = retriever.invoke(prompt)
                        
                        if not docs:
                            response = "❌ No relevant information found in documents. Please rephrase your question or upload more relevant documents."
                            sources = []
                        else:
                            # Build smart response from retrieved documents
                            response = build_answer(prompt, docs)
                            sources = [doc.page_content[:300] for doc in docs]
                        
                        # Display answer
                        st.markdown(response)
                        
                        # Extract threat keywords
                        threat_keywords = [
                            'malware', 'ransomware', 'phishing', 'ddos',
                            'vulnerability', 'exploit', 'breach', 'attack',
                            'threat', 'CVE', 'zero-day', 'trojan', 'botnet',
                            'worm', 'virus', 'spyware', 'intrusion'
                        ]
                        
                        detected_threats = [
                            keyword for keyword in threat_keywords
                            if keyword.lower() in (response + " " + prompt).lower()
                        ]
                        
                        if detected_threats:
                            st.markdown("### 🚨 Threats Mentioned:")
                            threat_badges = " ".join([
                                f"🔴 **{threat.upper()}**"
                                for threat in detected_threats[:5]
                            ])
                            st.markdown(threat_badges)
                        
                        # Show sources
                        if sources:
                            with st.expander("📚 View Sources"):
                                for i, source in enumerate(sources, 1):
                                    st.write(f"**Source {i}:**")
                                    st.write(source)
                                    if i < len(sources):
                                        st.divider()
                        
                        # Save to chat history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response,
                            "sources": sources
                        })
                        
                    except Exception as e:
                        error_response = f"❌ Error: {str(e)}\n\nPlease try again or rephrase your question."
                        st.error(error_response)
                        logger.error(f"Error: {str(e)}")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #888; padding: 1rem;">
    <p><strong>🛡️ CyberSec AI RAG System</strong></p>
    <p>Built with LangChain, HuggingFace Embeddings, and Streamlit</p>
    <p>Intern: S. Ravinder | Organization: Viswam AI</p>
    <p><small>Version 1.0.0 | Deployed on Streamlit Cloud | No API Keys Required</small></p>
</div>
""", unsafe_allow_html=True)

# Helper function to build smart answers
def build_answer(question: str, docs) -> str:
    """
    Build answer from retrieved documents
    """
    if not docs:
        return "No relevant information found."
    
    # Combine document content
    combined_context = "\n\n".join([doc.page_content for doc in docs])
    
    # Build response
    response = f"""Based on the documents provided, here's what I found:\n\n"""
    
    # Add key information from documents
    for i, doc in enumerate(docs, 1):
        content = doc.page_content[:250]
        response += f"**Information {i}:**\n{content}...\n\n"
    
    response += f"""

**Analysis:**
The retrieved documents contain {len(docs)} relevant sections about "{question}". 
Key points are highlighted above. Please review the sources for detailed information.

*Note: This response is generated from your uploaded documents using semantic search.*
"""
    
    return response