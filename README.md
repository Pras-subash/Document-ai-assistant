# Document AI Assistant

An AI-powered document assistant that uses RAG (Retrieval Augmented Generation) architecture to answer questions about your documents. Simply drop your PDF, DOC, TXT, or other document files into the knowledge base directory, and the Llama2/Ollama/sentence-transformers based AI will analyze them and provide relevant answers to your questions.

## Features

- Processes multiple document formats (txt, pdf, docx, md, html)
- Interactive Q&A about document content
- Local document processing and storage
- Efficient document retrieval using vector similarity
- Contextual responses using RAG architecture
- Web interface and CLI options

## Requirements

### Core Dependencies
- Python 3.11+ (recommended for best compatibility)
- Ollama (for local LLM inference)
- LangChain and LangChain Community (for RAG pipeline)
- ChromaDB (for vector storage)
- HuggingFace Transformers (for embeddings)
- Streamlit (for web interface)

### Python Packages
```bash
langchain==0.1.5
langchain-community==0.0.18
chromadb==0.4.22
ollama==0.1.6
sentence-transformers==2.5.1
huggingface-hub==0.20.3
unstructured>=0.11.0
python-magic>=0.4.27
python-docx==1.1.0
streamlit==1.31.1
pypdf>=3.17.1
```

### System Requirements
- Memory: Minimum 8GB RAM (16GB recommended)
- Storage: 2GB for base installation, plus space for your documents
- CPU: Modern multi-core processor
- GPU: Optional, improves embedding generation speed

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/doc-ai-assistant.git
cd doc-ai-assistant
```

2. Create and activate virtual environment:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. Install system dependencies (macOS):
```bash
brew install libmagic
```

5. Install and start Ollama:
```bash
# Install from https://ollama.ai
ollama run llama2
```

## Usage

### Web Interface (Recommended)
The Streamlit web interface provides an interactive way to use the Document AI Assistant with the following features:

- Chat-like interface for Q&A
- Persistent chat history
- System status monitoring
- Document processing status
- Clear history option

#### Starting the Web Interface

1. First, make sure Ollama is running in a terminal:
```bash
ollama run llama2
```

2. In a new terminal, activate the virtual environment and start Streamlit:
```bash
# Navigate to project directory
cd /path/to/doc-ai-assistant

# Activate virtual environment
source .venv/bin/activate

# Run Streamlit app
streamlit run webui.py
```

The web interface will automatically open in your default browser at `http://localhost:8501`

#### Web Interface Components

1. **Sidebar**
   - Shows system initialization status
   - Displays document loading progress
   - Indicates AI model readiness

2. **Main Chat Area**
   - Input field for questions
   - Chat history with Q&A pairs
   - Clear history button
   - Loading indicators during processing

3. **Error Handling**
   - Clear error messages
   - System status updates
   - Troubleshooting suggestions

#### Troubleshooting Web Interface

If you encounter issues:
1. Check if Ollama is running (`ollama run llama2`)
2. Verify virtual environment is activated
3. Ensure all dependencies are installed
4. Check system resources (RAM, CPU usage)
5. Look for errors in the terminal running Streamlit

### Command Line Interface
Run the CLI version:
```bash
python doc_ai_assistant.py
```

### Adding Documents
1. Place your documents in the `knowledge_base` directory
2. Supported formats: PDF, DOCX, TXT, MD, HTML
3. Restart the application to process new documents

## Project Structure
```
doc_ai_assistant/
├── knowledge_base/    # Your documents go here
├── chroma_db/        # Vector database storage
├── doc_ai_assistant.py   # Main RAG implementation
├── webui.py         # Streamlit web interface
└── requirements.txt  # Python dependencies
```

## License

MIT License
