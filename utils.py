import os
import glob
import re
import zipfile
import io
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
import streamlit as st

def add_metadata(doc, doc_type):
    """Add document type metadata to a document."""
    doc.metadata["doc_type"] = doc_type
    return doc

def load_documents():
    """Load documents from the knowledge base directory."""
    with st.spinner("Loading knowledge base..."):
        # Path to knowledge base
        folders = glob.glob("Horizon_Haven_Realty_Knowledge_Base/*")
        
        if not folders:
            st.error("Knowledge base folders not found! Please make sure the Horizon_Haven_Realty_Knowledge_Base directory exists.")
            return False
        
        text_loader_kwargs = {'encoding': 'utf-8'}
        
        # Read in documents using LangChain's loaders
        documents = []
        for folder in folders:
            doc_type = os.path.basename(folder)
            loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
            folder_docs = loader.load()
            documents.extend([add_metadata(doc, doc_type) for doc in folder_docs])
        
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        db_name = "hh_vector_db"
        
        # Check if vector store already exists
        if os.path.exists(db_name):
            st.session_state.vector_store = Chroma(persist_directory=db_name, embedding_function=st.session_state.embeddings)
            st.success(f"Loaded existing vector store with {st.session_state.vector_store._collection.count()} documents")
        else:
            # Create vector store
            st.session_state.vector_store = Chroma.from_documents(
                documents=chunks, 
                embedding=st.session_state.embeddings, 
                persist_directory=db_name
            )
            st.success(f"Created new vector store with {st.session_state.vector_store._collection.count()} documents")
        
        return True

def setup_conversation_chain():
    """Set up the conversation chain with the LLM and retriever."""
    with st.spinner("Setting up the conversation..."):
        try:
            # Using qwen3:4b with Ollama
            llm = ChatOpenAI(
                temperature=0.5, 
                model_name="qwen3:4b", 
                base_url='http://localhost:11434/v1', 
                api_key='ollama'
            )
            
            # Set up the conversation memory
            memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
            
            # The retriever is an abstraction over the VectorStore
            retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
            
            # Create the conversation chain
            st.session_state.conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=llm, 
                retriever=retriever, 
                memory=memory
            )
            
            return True
        except Exception as e:
            st.error(f"Error setting up conversation: {str(e)}")
            return False

def clean_response(response):
    """Clean the response from any XML/HTML tags."""
    # Clean <...> tags that might be in the response
    cleaned_response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    return cleaned_response

def create_knowledge_base_zip():
    """Create a zip file of the knowledge base."""
    knowledge_base_path = "Horizon_Haven_Realty_Knowledge_Base"
    
    # Check if the knowledge base exists
    if not os.path.exists(knowledge_base_path):
        return None
    
    # Create a BytesIO object to store the zip file
    zip_buffer = io.BytesIO()
    
    # Create the zip file
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Walk through the directory
        for root, _, files in os.walk(knowledge_base_path):
            for file in files:
                # Get the full path of the file
                file_path = os.path.join(root, file)
                # Add the file to the zip with a path relative to the knowledge base directory
                arc_name = os.path.relpath(file_path, os.path.dirname(knowledge_base_path))
                zip_file.write(file_path, arc_name)
    
    # Seek to the beginning of the BytesIO object
    zip_buffer.seek(0)
    return zip_buffer