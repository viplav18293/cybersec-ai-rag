import streamlit as st
import logging
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="CyberSec AI", page_icon="🛡️", layout="wide")

st.markdown("<h1 style='text-align: center; color: #FF6B6B;'>🛡️ CyberSec AI - Threat Intelligence RAG</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>Powered by HuggingFace | Free & Cloud</p>", unsafe_allow_html=True)

if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'docs_loaded' not in st.session_state:
    st.session_state.docs_loaded = False
if 'hf_token' not in st.session_state:
    st.session_state.hf_token = ""

with st.sidebar:
    st.header("⚙️ Configuration")
    
    hf_token = st.text_input(
        "HuggingFace API Token",
        type="password",
        help="Get free token from https://huggingface.co/settings/tokens"
    )
    
    if hf_token:
        st.session_state.hf_token = hf_token
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
        st.success("✅ Token configured!")
    else:
        st.warning("⚠️ Enter HuggingFace token")
        st.info("Get free token from:\nhttps://huggingface.co/settings/tokens")
    
    st.divider()
    st.subheader("📄 Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Choose files",
        type=['pdf', 'txt'],
        accept_multiple_files=True
    )
    
    if uploaded_files and st.button("🚀 Process Documents", type="primary", use_container_width=True):
        with st.spinner("Processing..."):
            try:
                all_documents = []
                
                for uploaded_file in uploaded_files:
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
                        st.success(f"✅ {uploaded_file.name}")
                    finally:
                        try:
                            os.unlink(tmp_path)
                        except:
                            pass
                
                if all_documents:
                    st.info("Splitting documents...")
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200
                    )
                    chunks = text_splitter.split_documents(all_documents)
                    
                    st.info("Creating embeddings...")
                    embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2",
                        model_kwargs={'device': 'cpu'}
                    )
                    
                    st.info("Building vector store...")
                    st.session_state.vectorstore = FAISS.from_documents(
                        documents=chunks,
                        embedding=embeddings
                    )
                    st.session_state.docs_loaded = True
                    
                    st.success(f"✅ Ready! {len(chunks)} chunks indexed")
                    st.balloons()
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    st.divider()
    
    if st.session_state.docs_loaded:
        st.success("✅ Documents Ready")
    else:
        st.info("Upload documents to start")
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if not st.session_state.docs_loaded:
    st.info("""
    ### 👋 Welcome to CyberSec AI!
    
    **Steps:**
    1. Enter HuggingFace API Token (free)
    2. Upload cyber security documents
    3. Click Process Documents
    4. Ask questions about threats!
    
    **Example:** What is ransomware?
    """)
else:
    st.subheader("💬 Chat Interface")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask about cyber security threats..."):
        if not st.session_state.hf_token:
            st.error("❌ Please enter HuggingFace token first!")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
                        docs = retriever.invoke(prompt)
                        context = "\n\n".join([doc.page_content for doc in docs])
                        
                        os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.session_state.hf_token
                        
                        llm = HuggingFaceHub(
                            repo_id="mistralai/Mistral-7B-Instruct-v0.1",
                            model_kwargs={"temperature": 0.3, "max_new_tokens": 500}
                        )
                        
                        prompt_template = PromptTemplate(
                            input_variables=["context", "question"],
                            template="""You are a cybersecurity expert. Answer based on context only.

Context: {context}

Question: {question}

Answer:"""
                        )
                        
                        chain = prompt_template | llm | StrOutputParser()
                        answer = chain.invoke({"context": context, "question": prompt})
                        
                        st.markdown(answer)
                        
                        with st.expander("📚 Sources"):
                            for i, doc in enumerate(docs, 1):
                                st.write(f"**Source {i}:** {doc.page_content[:200]}...")
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer
                        })
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        logger.error(f"Error: {str(e)}")

st.divider()
st.markdown("<p style='text-align: center; color: #888;'>🛡️ CyberSec AI | HuggingFace | Viswam AI</p>", unsafe_allow_html=True)