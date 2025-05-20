import streamlit as st
from utils import load_documents, setup_conversation_chain, clean_response, create_knowledge_base_zip

# App title and configuration
st.set_page_config(page_title="Horizon Haven Realty Assistant", layout="wide")
st.title("Horizon Haven Realty Knowledge Assistant")

# Sidebar content
with st.sidebar:
    st.header("About")
    st.markdown("""
    This assistant uses RAG (Retrieval-Augmented Generation) to answer questions about Horizon Haven Realty 
    using the company's knowledge base.
    """)
    
    # Knowledge base download button
    st.header("Knowledge Base")
    zip_buffer = create_knowledge_base_zip()
    if zip_buffer:
        st.download_button(
            label="Download Knowledge Base",
            data=zip_buffer,
            file_name="Horizon_Haven_Realty_Knowledge_Base.zip",
            mime="application/zip"
        )
    else:
        st.error("Knowledge base not found. Cannot create download.")

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

if 'conversation_chain' not in st.session_state:
    st.session_state.conversation_chain = None

if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None

if 'db_loaded' not in st.session_state:
    st.session_state.db_loaded = False

# Function to display chat messages
def display_messages():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Load docs and setup conversation on first run
if not st.session_state.db_loaded:
    if load_documents():
        st.session_state.db_loaded = True
        setup_conversation_chain()

# Display chat interface
display_messages()

# Chat input
if prompt := st.chat_input("Ask about Horizon Haven Realty"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        
        try:
            if st.session_state.conversation_chain:
                result = st.session_state.conversation_chain.invoke({"question": prompt})
                raw_response = result["answer"]
                # Clean the response to remove any XML/HTML tags
                response = clean_response(raw_response)
                message_placeholder.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                message_placeholder.error("Conversation chain not initialized. Please check configuration.")
        except Exception as e:
            message_placeholder.error(f"Error generating response: {str(e)}")