# 📚 Scientific Research AI Agent (RAG System)

A sophisticated AI-powered question-answering system that retrieves information from scientific research articles using Retrieval-Augmented Generation (RAG). This project implements multiple RAG architectures with various LLM backends and deployment-ready components.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-✓-orange.svg)](https://www.langchain.com/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-✓-yellow.svg)](https://huggingface.co/)
[![Docker](https://img.shields.io/badge/Docker-✓-blue.svg)](https://www.docker.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--3.5-green.svg)](https://openai.com/)

## 🔍 Overview

This AI agent specializes in answering questions based on scientific literature by implementing various RAG (Retrieval-Augmented Generation) architectures. The system processes PDF documents, creates semantic embeddings, and leverages large language models to provide accurate, context-aware responses grounded in research articles.

## 📋 Requirements

### Core Dependencies

Create a `requirements.txt` file with the following:

```txt
# LangChain Ecosystem
langchain>=0.3.10
langchain_community==0.3.27
langchain_chroma==0.1.4

# Vector Database & Storage
chromadb==0.4.24

# LlamaIndex Integration (Alternative RAG Framework)
llama-index==0.11.0
llama-index-embeddings-huggingface==0.6.1
llama_index_embeddings_langchain==0.4.1

# API & Web Framework
fastapi==0.111.0
uvicorn[standard]==0.24.0
python-multipart>=0.0.7

# LLM Providers & APIs
openai>=1.14.0

# Document Processing
pypdf>=4.0.1,<5.0.0

# Tokenization & Utilities
tiktoken==0.5.1
python-dotenv==1.0.0
aiofiles==23.2.1

# Data Validation
pydantic>=1.5.0

# ML & NLP Libraries
transformers==4.57.3
sentence-transformers==5.1.2
torch==2.2.2

# Scientific Computing
scikit-learn==1.6.1
scipy==1.13.1
threadpoolctl==3.6.0

# Development
notebook
```

### Installation

```bash
# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For GPU support with PyTorch (optional)
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu118
```

## 🏗️ Architecture

### Core Components
1. **Document Processing**: PDF loading and intelligent chunking using `RecursiveCharacterTextSplitter`
2. **Embedding Generation**: Multiple embedding models (OpenAI, HuggingFace sentence-transformers)
3. **Vector Database**: ChromaDB for efficient similarity search with optional persistence
4. **LLM Integration**: Multi-model support (OpenAI GPT-3.5, Ollama TinyLlama, Flan-T5)
5. **RAG Pipeline**: Modular pipeline for retrieval and generation

## 📂 Project Structure

```
AI-agent-for-answering-based-on-scientific-research-articles/
│
├── Notebooks/                              # Jupyter notebook implementations
│   ├── RAG-with-LangChain.ipynb           # LangChain RAG with multiple LLMs
│   ├── RAG-with-HuggingFace.ipynb         # Pure HuggingFace implementation
│   └── RAG-with-LangChain-with-functions.ipynb  # Modular function-based approach
│
├── RAGSystem/                             # Production-ready RAG system
│   ├── RAGSystem.py                       # Main RAG class with full pipeline
│   ├── RAGService.py                      # Service layer for API integration
│   ├── RAGClient.py                       # Client for system interaction
│   ├── RAGBatchProcessor.py              # Batch processing capabilities
│   ├── Dockerfile                         # Containerization setup
│   └── docker-compose.yaml               # Multi-service orchestration
│
└── README.md                             # This file
```

## 🛠️ Technical Stack

### Embedding Models
- **OpenAI Embeddings**: High-quality embeddings via OpenAI API
- **HuggingFace**: `sentence-transformers/all-MiniLM-L6-v2` for local embeddings
- **Custom Embeddings**: Support for any HuggingFace embedding model

### Vector Databases
- **ChromaDB**: Persistent and in-memory vector stores
- **FAISS** (optional): For production-scale similarity search

### Language Models
- **OpenAI GPT-3.5 Turbo**: Commercial LLM with excellent performance
- **Ollama TinyLlama**: Lightweight local LLM via Ollama
- **Google Flan-T5**: Open-source text-to-text generation model
- **Custom Models**: Any HuggingFace pipeline-compatible model

### Frameworks & Libraries
- **LangChain**: Orchestration framework for RAG pipelines
- **HuggingFace Transformers**: Local model inference
- **PyPDF2/PDFPlumber**: Document parsing
- **Docker**: Containerization and deployment

## 🚀 Getting Started

### Environment Setup

1. **Clone the repository**:
```bash
git clone https://github.com/imbilalbutt/AI-agent-for-answering-based-on-scientific-research-articles.git
cd AI-agent-for-answering-based-on-scientific-research-articles
```

2. **Set up environment variables**:
Create a `.env` file in the project root:
### Environment Variables
Create a `.env` file:
```env
OPENAI_API_KEY=your_openai_key_here
HUGGINGFACE_TOKEN=your_hf_token_here
MODEL_NAME=gpt-3.5-turbo
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
VECTOR_DB_PATH=./chroma_db
LANGCHAIN_TELEMETRY=false
HOST=0.0.0.0
PORT=8000
RELOAD=false
```

### Quick Start with Notebooks

1. **Basic RAG with LangChain**:
```python
# Open and run RAG-with-LangChain.ipynb
# Supports OpenAI, Ollama, and HuggingFace models
```

2. **Pure HuggingFace Implementation**:
```python
# Open RAG-with-HuggingFace.ipynb
# Uses local models only, no API dependencies
```

3. **Modular Function-Based RAG**:
```python
# Open RAG-with-LangChain-with-functions.ipynb
# Clean, reusable function-based architecture
```

### Production Deployment

1. **Build and Run with Docker**:
```bash
cd RAGSystem
docker-compose up --build
```

2. **Run the RAG System Locally**:
```python
from RAGSystem import RAGSystem

# To work with HuggingFace embedding and GPT model
openai_rag = rag = RAGSystem(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    llm_model="gpt-3.5-turbo"  # or "google/flan-t5-small" for local
)

openai_rag.initialize()

response = openai_rag.query("What are the key findings in this research?")
print(response)

# To work with HuggingFace embedding and models

huggingface_rag = RAGSystem(
        docs_dir="../docs/",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2", # "text-embedding-ada-002",
        llm_model="",
        persist_directory="./chroma_db5",
        llm_temperature = 0.0,
        llm_max_tokens = 512,
        use_hugging_face=True, hugging_face_model = "google/flan-t5-small", hf_task = "text2text-generation",
        use_ollama= False, ollama_model = "" )

huggingface_rag.initialize()

# Query the system
response = huggingface_rag.query("What are the key findings in this research?")
print(response)
```

3. **Use the Client-Server Architecture**:
```bash
# Start the service
python RAGService.py

# In another terminal, query using client
python RAGClient.py --query "Summarize the methodology section"
```

## 📊 Features

### 1. **Multi-Model Support**
- Switch between commercial and open-source LLMs seamlessly
- Compare performance across different model architectures

### 2. **Flexible Embedding Strategies**
- API-based embeddings (OpenAI)
- Local embeddings (HuggingFace sentence-transformers)
- Custom embedding model integration

### 3. **Intelligent Document Processing**
- Recursive text splitting preserving semantic boundaries
- Metadata extraction from PDFs
- Configurable chunk sizes and overlap

### 4. **Production-Ready Components**
- Dockerized deployment
- REST API service layer
- Batch processing capabilities
- Persistent vector stores

### 5. **API Endpoints**
#### When running the FastAPI service, available endpoints include:

- POST /ingest: Upload and process documents

- POST /query: Query the RAG system

- GET /health: Service health check

- GET /models: List available models

### 5. **Interactive Query Interface**
```python
# Interactive session
openai_rag.interactive_query()


huggingface_rag.interactive_query()
# Enter queries in real-time, exit with 'quit'
```

## 🔧 Docker Configuration

The `docker-compose.yaml` includes:
- RAG service with auto-reload
- Vector database persistence
- Optional GPU support for local models
- Volume mounts for document storage



## 📈 Performance & Evaluation

### Available Metrics
- Retrieval accuracy (top-k precision)
- Generation relevance scores
- Response time benchmarks
- Token usage optimization

### Optimization Features
- Caching mechanisms for frequent queries
- Dynamic chunk sizing based on document type
- Parallel document processing
- Batch embedding generation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Test your changes thoroughly
4. Submit a pull request

### Areas for Contribution
- Additional document formats (LaTeX, docx)
- More vector database backends (Pinecone, Weaviate)
- Advanced retrieval strategies (HyDE, multi-query)
- Evaluation framework integration

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [LangChain](https://www.langchain.com/) for the amazing RAG framework
- [HuggingFace](https://huggingface.co/) for open-source models and embeddings
- [Chroma](https://www.trychroma.com/) for the vector database
- The open-source AI community

## 📚 Citation

If you use this project in your research, please cite:
```bibtex
@software{ScientificRAGAgent2024,
  author = {Bilal Ahmad Butt},
  title = {AI Agent for Answering Based on Scientific Research Articles},
  year = {2025},
  url = {https://github.com/imbilalbutt/AI-agent-for-answering-based-on-scientific-research-articles}
}
```

---

**💡 Tip**: Start with `RAG-with-LangChain.ipynb` for a comprehensive overview, then explore the production system in the `RAGSystem/` directory for deployment-ready code.

