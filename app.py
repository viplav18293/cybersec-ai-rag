import streamlit as st
import os
from utils.loader import load_documents, chunk_documents
from utils.embedder import get_embeddings
from utils.retriever import create_vectorstore, retrieve_documents

# Page config
st.set_page_config(page_title="CyberSec AI", page_icon="🛡️", layout="wide")
st.title("🛡️ CyberSec AI - RAG System")

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Sidebar - Pre-loaded PDFs section
with st.sidebar:
    st.header("📚 Pre-loaded Cybersecurity Documents")

    # List of pre-loaded PDFs (add your PDFs here)
    preloaded_pdfs = {
        "DDoS Attacks": "data/ddos_attacks.pdf",
        "Phishing Threats": "data/phishing_attacks.pdf",
        "Ransomware Guide": "data/ransomware_threats.pdf"
    }

    # Button to load pre-loaded PDFs
    if st.button("🚀 Load Pre-loaded PDFs"):
        with st.spinner("Processing pre-loaded documents..."):
            try:
                documents = []
                for name, path in preloaded_pdfs.items():
                    if os.path.exists(path):
                        documents.extend(load_documents(path))

                # Split documents
                chunks = chunk_documents(documents)

                # Create embeddings
                embeddings = get_embeddings()

                # Create vector store
                st.session_state.vectorstore = create_vectorstore(chunks, embeddings)

                st.success(f"✅ Loaded {len(chunks)} chunks from pre-loaded PDFs!")
            except Exception as e:
                st.error(f"Error: {str(e)}")

    st.divider()

    # Upload section (keep existing)
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose files (PDF or TXT)",
        type=['pdf', 'txt'],
        accept_multiple_files=True
    )

    if st.button("Process Documents"):
        if uploaded_files:
            with st.spinner("Processing..."):
                try:
                    documents = []
                    for file in uploaded_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=file.name) as tmp:
                            tmp.write(file.getvalue())
                            tmp_path = tmp.name

                        try:
                            documents.extend(load_documents(tmp_path))
                        finally:
                            os.unlink(tmp_path)

                    chunks = chunk_documents(documents)
                    embeddings = get_embeddings()
                    st.session_state.vectorstore = create_vectorstore(chunks, embeddings)

                    st.success(f"✅ Processed {len(chunks)} chunks!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Please upload files first")

# Main Chat Interface (keep existing)
st.header("Chat Interface")

# Display messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
if prompt := st.chat_input("Ask about cybersecurity topics..."):
    if st.session_state.vectorstore is None:
        st.warning("Please load pre-loaded PDFs or upload documents first!")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.write(prompt)

        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Searching cybersecurity knowledge base..."):
                try:
                    docs = retrieve_documents(st.session_state.vectorstore, prompt, k=3)

                    if docs:
                        response = f"🛡️ Based on cybersecurity documents:\n\n"
                        for i, doc in enumerate(docs, 1):
                            response += f"{i}. {doc.page_content[:300]}...\n\n"
                    else:
                        response = "⚠️ No relevant cybersecurity information found."

                    st.write(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

                except Exception as e:
                    st.error(f"Error: {str(e)}")