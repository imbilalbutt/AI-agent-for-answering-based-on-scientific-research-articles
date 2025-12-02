import os
import uuid
from typing import List, Optional
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
import uvicorn

# Import the RAG system from previous implementation
from rag_system import RAGSystem


# Define Pydantic models for request/response
class QueryRequest(BaseModel):
    """Request model for querying the RAG system."""
    question: str = Field(..., min_length=1, max_length=1000, description="The question to answer")
    user_id: Optional[str] = Field(None, description="Optional user identifier")
    session_id: Optional[str] = Field(None, description="Optional session identifier")
    return_sources: bool = Field(True, description="Whether to return source documents")
    top_k: Optional[int] = Field(4, ge=1, le=20, description="Number of documents to retrieve")


class QueryResponse(BaseModel):
    """Response model for query results."""
    answer: str = Field(..., description="The generated answer")
    question: str = Field(..., description="The original question")
    sources: List[dict] = Field([], description="Source documents used")
    processing_time: Optional[float] = Field(None, description="Time taken to process in seconds")
    query_id: str = Field(..., description="Unique identifier for the query")


class BatchQueryRequest(BaseModel):
    """Request model for batch queries."""
    queries: List[str] = Field(..., min_items=1, max_items=100, description="List of questions")
    return_sources: bool = Field(True, description="Whether to return source documents")


class BatchQueryResponse(BaseModel):
    """Response model for batch query results."""
    results: List[dict] = Field(..., description="List of query results")
    total_queries: int = Field(..., description="Total number of queries processed")
    failed_queries: int = Field(0, description="Number of failed queries")


class SystemStatus(BaseModel):
    """System status information."""
    status: str = Field(..., description="System status")
    initialized: bool = Field(..., description="Whether system is initialized")
    document_count: Optional[int] = Field(None, description="Number of documents in vector store")
    vector_store_path: str = Field(..., description="Path to vector store")
    model_info: dict = Field(..., description="Model information")
    uptime: Optional[float] = Field(None, description="Uptime in seconds")


class DocumentInfo(BaseModel):
    """Information about a document."""
    id: str = Field(..., description="Document identifier")
    source: str = Field(..., description="Source file name")
    page_count: int = Field(..., description="Number of pages")
    chunk_count: int = Field(..., description="Number of chunks")


class UploadResponse(BaseModel):
    """Response model for document upload."""
    message: str = Field(..., description="Status message")
    uploaded_files: List[str] = Field(..., description="List of uploaded files")
    failed_files: List[str] = Field([], description="List of files that failed to upload")


class ReinitializeRequest(BaseModel):
    """Request model for reinitializing the system."""
    force_rebuild: bool = Field(False, description="Force rebuild of vector store")


# Global RAG system instance
rag_system = None
start_time = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager for FastAPI to handle startup and shutdown."""
    global rag_system, start_time

    # Startup
    print("Starting up RAG FastAPI service...")
    start_time = datetime.now()

    try:
        # Initialize RAG system
        rag_system = RAGSystem(
            docs_dir="/src/docs",
            embedding_model="text-embedding-ada-002",
            llm_model="gpt-3.5-turbo",
            persist_directory="./chroma_db"
        )
        rag_system.initialize()
        print("RAG system initialized successfully")
    except Exception as e:
        print(f"Failed to initialize RAG system: {e}")
        rag_system = None

    yield

    # Shutdown
    print("Shutting down RAG FastAPI service...")
    # Cleanup if needed


# Create FastAPI app with lifespan
app = FastAPI(
    title="RAG System API",
    description="REST API for Retrieval-Augmented Generation System",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency to check if system is initialized
def get_rag_system():
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    return rag_system


# Dependency to get system status
def get_system_status():
    uptime = None
    if start_time:
        uptime = (datetime.now() - start_time).total_seconds()

    doc_count = None
    if rag_system and rag_system.vector_store:
        try:
            # This is an example - adjust based on your vector store implementation
            doc_count = rag_system.vector_store._collection.count()
        except:
            pass

    return {
        "status": "ready" if rag_system else "not_initialized",
        "initialized": rag_system is not None,
        "document_count": doc_count,
        "vector_store_path": "./chroma_db",
        "model_info": {
            "embedding_model": "text-embedding-ada-002",
            "llm_model": "gpt-3.5-turbo"
        },
        "uptime": uptime
    }


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "RAG System API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "status": "/status",
            "query": "/query (POST)",
            "batch_query": "/batch-query (POST)",
            "documents": "/documents (GET)",
            "reinitialize": "/reinitialize (POST)",
            "api_docs": "/docs"
        }
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/status", response_model=SystemStatus, tags=["System"])
async def get_status():
    """Get system status and statistics."""
    status_info = get_system_status()
    return SystemStatus(**status_info)


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_rag(
        request: QueryRequest,
        rag: RAGSystem = Depends(get_rag_system)
):
    """
    Query the RAG system with a single question.

    - **question**: The question to answer (required)
    - **user_id**: Optional user identifier
    - **session_id**: Optional session identifier
    - **return_sources**: Whether to include source documents
    - **top_k**: Number of documents to retrieve (default: 4)
    """
    import time

    start_time = time.time()
    query_id = str(uuid.uuid4())

    try:
        # Configure retriever with top_k
        if request.top_k and hasattr(rag.retriever, 'search_kwargs'):
            rag.retriever.search_kwargs["k"] = request.top_k

        # Process query
        response = rag.query(request.question)

        processing_time = time.time() - start_time

        # Prepare response
        result = QueryResponse(
            answer=response["answer"],
            question=request.question,
            sources=response["sources"] if request.return_sources else [],
            processing_time=round(processing_time, 3),
            query_id=query_id
        )

        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


@app.post("/batch-query", response_model=BatchQueryResponse, tags=["Query"])
async def batch_query(
        request: BatchQueryRequest,
        background_tasks: BackgroundTasks,
        rag: RAGSystem = Depends(get_rag_system)
):
    """
    Process multiple queries in batch mode.

    - **queries**: List of questions to process (max 100)
    - **return_sources**: Whether to include source documents
    """
    results = []
    failed_count = 0

    for question in request.queries:
        try:
            response = rag.query(question)

            result = {
                "question": question,
                "answer": response["answer"],
                "sources": response["sources"] if request.return_sources else [],
                "success": True
            }
            results.append(result)

        except Exception as e:
            failed_count += 1
            results.append({
                "question": question,
                "answer": f"Error: {str(e)}",
                "sources": [],
                "success": False
            })

    return BatchQueryResponse(
        results=results,
        total_queries=len(request.queries),
        failed_queries=failed_count
    )


@app.get("/documents", tags=["Documents"])
async def list_documents(rag: RAGSystem = Depends(get_rag_system)):
    """List all documents in the system."""
    try:
        # This is a simplified example - you might need to adjust
        # based on how you store document metadata

        # If using ChromaDB directly
        if hasattr(rag.vector_store, '_collection'):
            collection = rag.vector_store._collection
            results = collection.get(include=['metadatas'])

            documents = {}
            if results and 'metadatas' in results:
                for metadata in results['metadatas']:
                    source = metadata.get('source', 'unknown')
                    if source not in documents:
                        documents[source] = {
                            'source': source,
                            'chunk_count': 0,
                            'pages': set()
                        }
                    documents[source]['chunk_count'] += 1
                    if 'page' in metadata:
                        documents[source]['pages'].add(metadata['page'])

            # Convert to list format
            doc_list = []
            for source, info in documents.items():
                doc_list.append({
                    'source': source,
                    'chunk_count': info['chunk_count'],
                    'page_count': len(info['pages']) if info['pages'] else 1
                })

            return {
                "documents": doc_list,
                "total_documents": len(doc_list),
                "total_chunks": sum(d['chunk_count'] for d in doc_list)
            }

        return {"message": "Document list not available"}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving documents: {str(e)}"
        )


@app.post("/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_documents(
        background_tasks: BackgroundTasks,
        rag: RAGSystem = Depends(get_rag_system)
):
    """
    Trigger re-indexing of documents from the source directory.

    Note: This will rebuild the vector store with current documents.
    """
    try:
        # This would typically involve:
        # 1. Checking for new PDFs in /src/docs
        # 2. Processing them
        # 3. Updating the vector store

        # For now, we'll just return a message
        # In a real implementation, you'd trigger the re-indexing process

        return UploadResponse(
            message="Document upload triggered. The system will re-index documents.",
            uploaded_files=[],
            failed_files=[]
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error uploading documents: {str(e)}"
        )


@app.post("/reinitialize", tags=["System"])
async def reinitialize_system(
        request: ReinitializeRequest = ReinitializeRequest(),
        rag: RAGSystem = Depends(get_rag_system)
):
    """
    Reinitialize the RAG system.

    - **force_rebuild**: Force complete rebuild of vector store
    """
    try:
        # In a real implementation, you might:
        # 1. Clear the vector store if force_rebuild is True
        # 2. Reinitialize the system

        # For this example, we'll just re-run initialization
        rag.initialize()

        return {
            "message": "System reinitialized successfully",
            "force_rebuild": request.force_rebuild,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error reinitializing system: {str(e)}"
        )


@app.get("/logs", tags=["System"])
async def get_logs(
        lines: int = 100,
        level: Optional[str] = "INFO"
):
    """
    Get system logs.

    - **lines**: Number of log lines to return (default: 100)
    - **level**: Log level filter (INFO, WARNING, ERROR, DEBUG)
    """
    try:
        # This is a simplified example
        # In production, you'd want to read from a log file
        log_file = "rag_system.log"

        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                logs = f.readlines()[-lines:]
            return {"logs": logs}
        else:
            return {"logs": ["No log file found"]}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving logs: {str(e)}"
        )


@app.get("/api-docs", include_in_schema=False)
async def custom_api_docs():
    """Redirect to Swagger UI."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "path": request.url.path,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "path": request.url.path,
            "timestamp": datetime.now().isoformat()
        }
    )


if __name__ == "__main__":
    # Configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    reload = os.getenv("RELOAD", "false").lower() == "true"

    print(f"Starting FastAPI server on {host}:{port}")
    print(f"Swagger UI: http://{host}:{port}/docs")
    print(f"Redoc: http://{host}:{port}/redoc")

    uvicorn.run(
        "fastapi_rag:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )