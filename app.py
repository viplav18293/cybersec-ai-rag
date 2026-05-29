import streamlit as st
import logging
import tempfile
import os

from utils.loader import DocumentLoader
from utils.embedder import EmbeddingGenerator
from utils.retriever import DocumentRetriever
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="CyberSec AI RAG", page_icon="🛡️", layout="wide")

st.markdown("<h1 style='text-align: center; color: #FF6B6B;'>🛡️ CyberSec AI - Threat Intelligence RAG System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>Powered by TinyLlama | Free & Local</p>", unsafe_allow_html=True)

if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'docs_loaded' not in st.session_state:
    st.session_state.docs_loaded = False

with st.sidebar:
    st.header("⚙️ Configuration")
    st.info("✅ Using TinyLlama (FREE - Local)")
    
    st.divider()
    st.subheader("📄 Upload Documents")
    
    uploaded_files = st.file_uploader("Choose cyber security documents", type=['pdf', 'txt'], accept_multiple_files=True)
    
    if uploaded_files and st.button("🚀 Process Documents", type="primary", use_container_width=True):
        with st.spinner("Processing documents..."):
            try:
                loader = DocumentLoader(chunk_size=1000, chunk_overlap=200)
                embedder = EmbeddingGenerator(model_name="sentence-transformers/all-MiniLM-L6-v2")
                retriever = DocumentRetriever(embedder.get_embeddings_object())
                
                all_chunks = []
                
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f'_{uploaded_file.name}') as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                    
                    try:
                        chunks = loader.load_and_chunk(tmp_path)
                        all_chunks.extend(chunks)
                        st.success(f"✅ Processed {uploaded_file.name}")
                    finally:
                        try:
                            os.unlink(tmp_path)
                        except:
                            pass
                
                if all_chunks:
                    st.info(f"Creating vector store with {len(all_chunks)} chunks...")
                    retriever.create_vector_store(all_chunks)
                    st.session_state.retriever = retriever
                    st.session_state.docs_loaded = True
                    
                    st.success(f"🎉 Ready! Indexed {len(all_chunks)} chunks")
                    st.balloons()
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                logger.error(f"Error: {str(e)}")
    
    st.divider()
    
    if st.session_state.docs_loaded:
        st.success("✅ Documents Ready")
    else:
        st.info("Upload documents to start")
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if not st.session_state.docs_loaded:
    st.markdown("""
    ### 👋 Welcome to CyberSec AI!
    
    **About this RAG System:**
    - 🎓 Built for Viswam AI internship
    - 📚 Analyzes cyber security threat documents
    - 🤖 Uses TinyLlama (local, free)
    - 🔍 Retrieval-Augmented Generation (RAG)
    
    **How RAG Works:**
    1. Load documents
    2. Split into chunks (1000 tokens each)
    3. Generate embeddings (HuggingFace)
    4. Store in vector database (FAISS)
    5. Retrieve relevant docs for each query
    6. Generate answer using LLM
    
    **Steps to Use:**
    1. Upload cyber security documents (PDF/TXT)
    2. Click "Process Documents"
    3. Ask questions about threats
    """)
else:
    st.subheader("💬 Chat with Your Documents")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("📚 Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.write(f"**Source {i}:** {source[:150]}...")
    
    if prompt := st.chat_input("Ask about cyber security threats..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Analyzing with TinyLlama..."):
                try:
                    docs = st.session_state.retriever.retrieve_documents(prompt, k=3)
                    context = "\n\n".join([doc.page_content for doc in docs])
                    
                    llm = Ollama(
                        model="tinyllama",
                        base_url="http://localhost:11434",
                        temperature=0.3
                    )
                    
                    prompt_template = PromptTemplate(
                        input_variables=["context", "question"],
                        template="""You are a cyber security expert. Answer based on the context.

Context:
{context}

Question: {question}

Answer:"""
                    )
                    
                    chain = prompt_template | llm | StrOutputParser()
                    answer = chain.invoke({"context": context, "question": prompt})
                    
                    st.markdown(answer)
                    
                    sources = [doc.page_content[:200] for doc in docs]
                    
                    with st.expander("📚 Retrieved Sources"):
                        for i, doc in enumerate(docs, 1):
                            st.write(f"**Source {i}:** {doc.page_content[:300]}...")
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    logger.error(f"Error: {str(e)}")

st.divider()
st.markdown("<p style='text-align: center; color: #888;'>🛡️ CyberSec AI RAG | TinyLlama | Viswam AI</p>", unsafe_allow_html=True)