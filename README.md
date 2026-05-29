# 🛡️ CyberSec AI - RAG System for Threat Intelligence

**An AI-powered Retrieval-Augmented Generation (RAG) system for analyzing cyber security threats**

---

## 📋 Project Information

- **Intern:** S. Ravinder
- **Organization:** Viswam AI
- **Platform:** code.swecha.org (GitLab)
- **Topic:** AI Cyber Security Threat Intelligence
- **Duration:** Internship Project
- **Status:** ✅ Completed & Working

---

## 🎯 Project Overview

CyberSec AI is a **Retrieval-Augmented Generation (RAG)** system that allows users to upload cyber security threat documents and ask intelligent questions about them. The system uses:

- **Document Processing:** Automatic PDF/TXT loading and chunking
- **Embeddings:** HuggingFace Sentence Transformers
- **Vector Store:** FAISS for efficient similarity search
- **LLM:** TinyLlama (local, free, no API costs)
- **UI:** Streamlit for interactive chat interface

---

## 📚 Document Used & Why

### Document Source
We use three cyber security threat documents:
1. **ransomware_threats.txt** - Ransomware attack patterns and prevention
2. **phishing_attacks.txt** - Phishing techniques and defense strategies
3. **ddos_attacks.txt** - DDoS attack types and mitigation methods

### Why These Documents?
- ✅ Comprehensive coverage of major cyber threats
- ✅ Practical prevention strategies
- ✅ Real-world attack statistics
- ✅ Relevant for threat intelligence analysis
- ✅ Good mix of technical and strategic information

---

## ⚙️ How Chunking Works

### Chunking Strategy
```python
RecursiveCharacterTextSplitter(
    chunk_size=1000,           # Maximum tokens per chunk
    chunk_overlap=200,         # Overlap for context preservation
    separators=["\n\n", "\n", ". ", " ", ""]  # Smart splitting priority
)