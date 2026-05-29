# src/llm_chain.py
"""
LLM Chain setup for RAG system
"""
import logging
from typing import Optional, Dict, Any

from langchain_community.llms import OpenAI
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

from config.settings import settings
from src.retrieval import RetrieverManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMChainManager:
    """
    Manages LLM chains for RAG system
    """
    
    def __init__(self, retriever_manager: RetrieverManager):
        """
        Initialize LLMChainManager
        
        Args:
            retriever_manager: RetrieverManager instance
        """
        self.retriever_manager = retriever_manager
        self.llm = None
        self.chat_llm = None
        
        self._initialize_llms()
    
    def _initialize_llms(self):
        """Initialize LLM instances"""
        try:
            if not settings.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY not set")
            
            # Standard LLM
            self.llm = OpenAI(
                temperature=settings.TEMPERATURE,
                openai_api_key=settings.OPENAI_API_KEY,
                model_name=settings.MODEL_NAME,
                max_tokens=settings.MAX_TOKENS
            )
            
            # Chat LLM (for conversational chains)
            self.chat_llm = ChatOpenAI(
                temperature=settings.TEMPERATURE,
                openai_api_key=settings.OPENAI_API_KEY,
                model_name=settings.MODEL_NAME,
                max_tokens=settings.MAX_TOKENS
            )
            
            logger.info("LLM instances initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing LLMs: {str(e)}")
            raise
    
    def get_qa_chain(self) -> RetrievalQA:
        """
        Get RetrievalQA chain
        
        Returns:
            RetrievalQA chain
        """
        try:
            retriever = self.retriever_manager.get_retriever()
            
            # Create custom prompt
            prompt_template = """Use the following pieces of context about cyber security threats to answer the question.
If you don't know the answer, say so - don't make up information.
Always cite the source of your information.

Context:
{context}

Question: {question}

Answer: I'll help you with this cyber security threat question based on the provided context."""
            
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": prompt}
            )
            
            logger.info("Created RetrievalQA chain")
            return qa_chain
            
        except Exception as e:
            logger.error(f"Error creating QA chain: {str(e)}")
            raise
    
    def get_conversational_chain(self) -> ConversationalRetrievalChain:
        """
        Get ConversationalRetrievalChain for multi-turn conversations
        
        Returns:
            ConversationalRetrievalChain
        """
        try:
            retriever = self.retriever_manager.get_retriever()
            
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key='answer'
            )
            
            # Create custom prompt
            condense_prompt = PromptTemplate(
                input_variables=["chat_history", "question"],
                template="""Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}

Follow Up Input: {question}

Standalone question:"""
            )
            
            qa_prompt = PromptTemplate(
                input_variables=["context", "question"],
                template="""Use the following pieces of context about cyber security threats to answer the question.
If you don't know the answer, say so - don't make up information.

Context:
{context}

Question: {question}

Answer:"""
            )
            
            conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=self.chat_llm,
                retriever=retriever,
                memory=memory,
                condense_question_prompt=condense_prompt,
                combine_docs_chain_kwargs={"prompt": qa_prompt},
                return_source_documents=True,
                output_key='answer'
            )
            
            logger.info("Created ConversationalRetrievalChain")
            return conversation_chain
            
        except Exception as e:
            logger.error(f"Error creating conversational chain: {str(e)}")
            raise
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the RAG system
        
        Args:
            question: User question
            
        Returns:
            Response with answer and sources
        """
        try:
            qa_chain = self.get_qa_chain()
            response = qa_chain({"query": question})
            
            logger.info(f"Query processed: {question[:50]}...")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise