import streamlit as st
import logging
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="CyberSec AI", page_icon="🛡️", layout="wide")

st.markdown("<h1 style='text-align: center; color: #FF6B6B;'>🛡️ CyberSec AI - Threat Intelligence RAG</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>Powered by HuggingFace | Free & Cloud-Ready</p>", unsafe_allow_html=True)

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
    st.info("✅ Using HuggingFace (FREE & Cloud-Ready)")
    
    hf_token = st.text_input(
        "HuggingFace API Token",
        type="password",
        value=st.session_state.hf_token,
        help="Get free token from https://huggingface.co/settings/tokens"
    )
    
    if hf_token:
        st.session_state.hf_token = hf_token
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
        st.success("✅ Token configured!")
    else:
        st.warning("⚠️ Enter HuggingFace token to use AI")
        with st.expander("How to get token"):
            st.write("1. Go to: https://huggingface.co/settings/tokens")
            st.write("2. Click 'New token'")
            st.write("3. Name: cybersecai")
            st.write("4. Type: Read")
            st.write("5. Copy token and paste above")
    
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
    
    **🆓 Completely FREE RAG System**
    
    #### Features:
    - ✅ Upload cyber security documents (PDF/TXT)
    - ✅ AI-powered question answering
    - ✅ Source citations and verification
    - ✅ Semantic search over your documents
    - ✅ No API costs (uses free HuggingFace)
    
    #### How to use:
    1. **🔑 Get HuggingFace Token** (free from https://huggingface.co/settings/tokens)
    2. **📄 Upload Documents** - Cyber security PDFs or text files
    3. **🚀 Process** - Click the process button and wait
    4. **💬 Ask Questions** - Chat about threats, attacks, prevention
    
    #### Example Questions:
    - What is ransomware and how does it work?
    - How can organizations prevent phishing attacks?
    - What are the different types of DDoS attacks?
    - What are the best cyber security practices?
    
    #### Technology Stack:
    - **🤖 AI Model:** HuggingFace Mistral-7B (free)
    - **🔍 Search:** Semantic similarity with embeddings
    - **📊 Processing:** LangChain + FAISS vector store
    - **🎨 Interface:** Streamlit (this web app)
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
        
        # Validate inputs
        if not st.session_state.hf_token:
            st.error("❌ Please enter your HuggingFace API token in the sidebar first!")
        elif not st.session_state.vectorstore:
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
                with st.spinner("🤔 Analyzing with AI..."):
                    try:
                        # Retrieve relevant documents
                        retriever = st.session_state.vectorstore.as_retriever(
                            search_kwargs={"k": 3}
                        )
                        docs = retriever.invoke(prompt)
                        
                        # Prepare context
                        context = "\n\n".join([doc.page_content for doc in docs])
                        
                        # Set up HuggingFace LLM
                        os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.session_state.hf_token
                        
                        llm = HuggingFaceEndpoint(
                            repo_id="mistralai/Mistral-7B-Instruct-v0.1",
                            temperature=0.3,
                            max_new_tokens=500,
                            huggingfacehub_api_token=st.session_state.hf_token
                        )
                        
                        # Create prompt template
                        prompt_template = PromptTemplate(
                            input_variables=["context", "question"],
                            template="""You are a cybersecurity expert. Use the following context to answer the question accurately and concisely.

Context:
{context}

Question: {question}

Answer: Based on the provided context, """
                        )
                        
                        # Create and run chain
                        chain = prompt_template | llm | StrOutputParser()
                        answer = chain.invoke({
                            "context": context,
                            "question": prompt
                        })
                        
                        # Clean up answer
                        if answer.startswith("Based on the provided context, "):
                            answer = answer.replace("Based on the provided context, ", "")
                        
                        # Display answer
                        st.markdown(answer)
                        
                        # Extract and display threat keywords
                        threat_keywords = [
                            'malware', 'ransomware', 'phishing', 'ddos',
                            'vulnerability', 'exploit', 'breach', 'attack',
                            'threat', 'CVE', 'zero-day', 'trojan', 'botnet'
                        ]
                        
                        detected_threats = [
                            keyword for keyword in threat_keywords
                            if keyword.lower() in (answer + " " + prompt).lower()
                        ]
                        
                        if detected_threats:
                            st.markdown("### 🚨 Threats Detected:")
                            threat_badges = " ".join([
                                f"🔴 **{threat.upper()}**"
                                for threat in detected_threats[:5]
                            ])
                            st.markdown(threat_badges)
                        
                        # Show sources
                        sources = []
                        if docs:
                            with st.expander("📚 View Sources"):
                                for i, doc in enumerate(docs, 1):
                                    source_text = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                                    st.write(f"**Source {i}:**")
                                    st.write(source_text)
                                    if i < len(docs):
                                        st.divider()
                                    sources.append(source_text)
                        
                        # Save to chat history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        })
                        
                    except Exception as e:
                        error_msg = str(e)
                        st.error(f"❌ AI Error: {error_msg}")
                        
                        # Fallback response
                        st.write("🔄 **Fallback Response:**")
                        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 2})
                        docs = retriever.invoke(prompt)
                        
                        if docs:
                            fallback_answer = f"Based on the documents, here's relevant information:\n\n"
                            for doc in docs:
                                fallback_answer += doc.page_content[:200] + "...\n\n"
                            
                            st.markdown(fallback_answer)
                            
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": fallback_answer
                            })
                        else:
                            st.write("I couldn't find relevant information in the documents.")
                        
                        logger.error(f"LLM Error: {error_msg}")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #888; padding: 1rem;">
    <p><strong>🛡️ CyberSec AI RAG System</strong></p>
    <p>Built with LangChain, HuggingFace, and Streamlit</p>
    <p>Intern: S. Ravinder | Organization: Viswam AI</p>
    <p><small>Version 1.0.0 | Deployed on Streamlit Cloud</small></p>
</div>
""", unsafe_allow_html=True)