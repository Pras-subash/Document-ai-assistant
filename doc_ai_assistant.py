# doc_ai_assistant.py
"""
Document AI Assistant - Main Module
This module implements a conversational AI assistant for document-related questions
using RAG (Retrieval Augmented Generation) architecture.

Core functionality:
- Loads and processes various document formats (txt, pdf, docx, md, html)
- Uses vector embeddings for efficient document retrieval
- Provides conversational AI interface using Ollama LLM
- Maintains persistent vector storage using ChromaDB

The system supports:
- Interactive Q&A about document content
- Local document processing and storage
- Efficient document retrieval using vector similarity
- Contextual responses using RAG architecture

Key Functions:
    load_and_chunk_documents(knowledge_base_path="knowledge_base") -> List[Document]
        Loads and splits documents from knowledge base into chunks
    setup_vector_store(chunks, embeddings) -> Chroma
        Initializes or loads ChromaDB vector store with error handling
    get_ollama_llm(model_name="llama2") -> Ollama
        Sets up the Ollama language model
    ask_ai_assistant(query, vector_store, llm) -> str
        Processes user queries using RAG for contextual responses
    get_embedding_model() -> HuggingFaceEmbeddings
        Initializes the sentence transformer embedding model

Dependencies:
    - langchain and langchain_community
    - chromadb
    - ollama
    - huggingface-hub
    - sentence-transformers
    - unstructured (for document loading)

Usage:
    Run directly for CLI interface with example photography knowledge base
    Contains built-in error handling for corrupted vector stores
"""

import os

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Directory to store ChromaDB data
CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "document_knowledge"

def setup_vector_store(chunks, embeddings):
    """
    Initialize or load the ChromaDB vector store.
    """
    import shutil
    
    # If database exists but is corrupted, remove it
    if os.path.exists(CHROMA_DB_DIR):
        try:
            # Test if we can load the existing database
            vectorstore = Chroma(
                persist_directory=CHROMA_DB_DIR,
                embedding_function=embeddings,
                collection_name=COLLECTION_NAME
            )
            # Test if we can access the collection
            count = vectorstore._collection.count()
            print(f"Loading existing ChromaDB from {CHROMA_DB_DIR} with {count} documents...")
            return vectorstore
        except Exception as e:
            print(f"Error loading existing ChromaDB: {e}")
            print("Removing corrupted database...")
            shutil.rmtree(CHROMA_DB_DIR)
    
    # Create new database
    print(f"Creating new ChromaDB at {CHROMA_DB_DIR}...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR,
        collection_name=COLLECTION_NAME
    )
    vectorstore.persist()
    print(f"ChromaDB created and populated with {len(chunks)} chunks.")
    return vectorstore

def get_ollama_llm(model_name="llama2"):
    """
    Initializes and returns an Ollama LLM instance.
    Ensure you have 'ollama run <model_name>' executed at least once to download the model.
    """
    print(f"Initializing Ollama LLM with model: {model_name}")
    llm = Ollama(model=model_name)
    return llm

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader
)

def load_and_chunk_documents(knowledge_base_path="knowledge_base"):
    documents = []
    for filename in os.listdir(knowledge_base_path):
        filepath = os.path.join(knowledge_base_path, filename)
        if filename.endswith(".txt"):
            print(f"Loading text file: {filename}")
            loader = TextLoader(filepath)
        elif filename.endswith(".pdf"):
            print(f"Loading PDF file: {filename}")
            loader = PyPDFLoader(filepath)
        elif filename.endswith(".docx"):
            print(f"Loading Word document: {filename}")
            loader = UnstructuredWordDocumentLoader(filepath)
        elif filename.endswith(".md"):
            print(f"Loading Markdown file: {filename}")
            loader = UnstructuredMarkdownLoader(filepath)
        elif filename.endswith(".html"):
            print(f"Loading HTML file: {filename}")
            loader = UnstructuredHTMLLoader(filepath)
        else:
            print(f"Unsupported file format: {filename}")
            continue
        documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Loaded {len(documents)} documents and split into {len(chunks)} chunks.")
    return chunks

    
def get_embedding_model():
    # This model is good for general purpose embeddings and runs locally.
    # It will be downloaded the first time it's used.
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    print(f"Loaded embedding model: {model_name}")
    return embeddings

def ask_ai_assistant(query, vector_store, llm):
    # 1. Define the retriever: how many relevant documents to fetch
    retriever = vector_store.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 relevant chunks

    # 2. Define the RAG prompt
    template = """
    You are an AI document assistant. Use the following context to answer the question.
    If you don't know the answer based on the provided context, politely state that you don't have enough information.

    Context:
    {context}

    Question: {question}

    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)

    # 3. Create the RAG chain
    # The chain first retrieves documents, then formats them into the prompt,
    # then passes to the LLM, and finally parses the output.
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print(f"\n--- Asking AI Assistant ---")
    print(f"Query: {query}")
    response = rag_chain.invoke(query)
    print(f"\n--- AI Assistant's Answer ---")
    print(response)
    return response


if __name__ == "__main__":
    print("Starting CLI interface (run webui.py for web interface)...")
    # Create a dummy knowledge_base directory and files if they don't exist
    if not os.path.exists("knowledge_base"):
        os.makedirs("knowledge_base")
        with open("knowledge_base/aperture.txt", "w") as f:
            f.write("Aperture is a fundamental concept in photography that controls the amount of light entering the camera lens and affects the depth of field. It's measured in f-numbers (f-stops). A lower f-number (e.g., f/2.8) means a wider aperture, letting in more light and creating a shallow depth of field (blurry background). A higher f-number (e.g., f/16) means a narrower aperture, letting in less light and resulting in a larger depth of field (more of the scene in focus). Wide apertures are commonly used for portraits to isolate the subject, while narrow apertures are preferred for landscapes to ensure everything from foreground to background is sharp.")
        with open("knowledge_base/shutter_speed.txt", "w") as f:
            f.write("Shutter speed dictates how long the camera's shutter remains open, allowing light to hit the sensor. It is measured in fractions of a second (e.g., 1/1000s, 1/60s) or full seconds. Fast shutter speeds (e.g., 1/1000s) freeze motion, ideal for sports or fast-moving subjects. Slow shutter speeds (e.g., 1/30s or longer) blur motion, creating artistic effects like light trails or silky water. A tripod is essential for slow shutter speeds to prevent camera shake.")
    
    documents_chunks = load_and_chunk_documents()
    
    # You can inspect the first few chunks:
    for i, chunk in enumerate(documents_chunks[:3]):
        print(f"\n--- Chunk {i+1} ---")
        print(chunk.page_content)
        
    embedding_model = get_embedding_model()
    vector_store = setup_vector_store(documents_chunks, embedding_model)
    print(f"Vector store initialized with {vector_store._collection.count()} documents.")
    llm = get_ollama_llm(model_name="llama2")
    
    # Test the assistant
    while True:
        user_query = input("\nEnter your document question (or 'quit' to exit): ")
        if user_query.lower() == 'quit':
            break
        ask_ai_assistant(user_query, vector_store, llm)