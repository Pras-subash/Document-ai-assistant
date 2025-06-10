import streamlit as st
from doc_ai_assistant import (
    load_and_chunk_documents,
    get_embedding_model,
    setup_vector_store,
    get_ollama_llm,
    ask_ai_assistant
)

# Configure the Streamlit page
st.set_page_config(
    page_title="Document AI Assistant",
    page_icon="📚",
    layout="wide"
)

# Initialize session state variables with default values
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "vector_store" not in st.session_state:
    st.session_state["vector_store"] = None
if "llm" not in st.session_state:
    st.session_state["llm"] = None
if "initialized" not in st.session_state:
    st.session_state["initialized"] = False

# Title and description
st.title("📚 Document AI Assistant")
st.markdown("""
This AI assistant helps you analyze and understand your documents.
Ask questions about your loaded documents and get AI-powered responses.
""")

# Sidebar for system status
with st.sidebar:
    st.header("System Status")
    
    # Initialize components if not already done
    if not st.session_state["initialized"]:
        try:
            with st.spinner('Loading document knowledge base...'):
                documents_chunks = load_and_chunk_documents()
                embedding_model = get_embedding_model()
                st.session_state["vector_store"] = setup_vector_store(documents_chunks, embedding_model)
                st.success("✅ Document knowledge base loaded!")
            
            with st.spinner('Initializing AI model...'):
                st.session_state["llm"] = get_ollama_llm(model_name="llama2")
                st.success("✅ AI model ready!")
                
            st.session_state["initialized"] = True
        except Exception as e:
            st.error(f"Error initializing system: {str(e)}")
            st.info("Make sure Ollama is running: `ollama run llama2`")
    else:
        st.success("✅ System initialized and ready!")

# Main chat interface
st.markdown("### Chat Interface")

# Display chat history
for q, a in st.session_state.chat_history:
    st.markdown("**Question:**")
    st.info(q)
    st.markdown("**Answer:**")
    st.success(a)
    st.markdown("---")

# Input area
user_question = st.text_input("Your question:", key="user_input", 
                            placeholder="Ask about your documents...")
col1, col2 = st.columns([1, 5])

with col1:
    ask_button = st.button("Ask", type="primary")
with col2:
    if st.button("Clear History"):
        st.session_state.chat_history = []
        st.rerun()

# Process question
if ask_button and user_question:
    if st.session_state.vector_store is None or st.session_state.llm is None:
        st.error("⚠️ System not fully initialized. Check the sidebar for errors.")
    else:
        try:
            with st.spinner('Getting answer...'):
                response = ask_ai_assistant(
                    user_question,
                    st.session_state.vector_store,
                    st.session_state.llm
                )
                # Add to chat history
                st.session_state.chat_history.append((user_question, response))
                st.rerun()
        except Exception as e:
            st.error(f"Error getting response: {str(e)}")
            st.info("Try refreshing the page or checking if Ollama is running properly")