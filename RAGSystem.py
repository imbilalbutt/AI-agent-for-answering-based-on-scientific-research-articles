import os
os.environ["LANGCHAIN_TELEMETRY"] = "false"

from dotenv import load_dotenv
load_dotenv()
import glob
from typing import List
import logging

# Import required libraries
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.schema import Document


# Optional: Import llama_index if you want to use it alongside LangChain
try:
    from llama_index import (
        GPTVectorStoreIndex,
        SimpleDirectoryReader,
        LLMPredictor,
        ServiceContext
    )
    from llama_index.vector_stores import ChromaVectorStore
    from llama_index.storage.storage_context import StorageContext

    HAVE_LLAMA_INDEX = True
except ImportError:
    HAVE_LLAMA_INDEX = False
    print("LlamaIndex not installed. Install with: pip install llama-index")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """
    Main function to run the RAG system.
    """


    # Check for OpenAI API key
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError("Please set OPENAI_API_KEY environment variable")

    # Initialize RAG system
    rag_system = RAGSystem(
        docs_dir="docs/",
        embedding_model="text-embedding-ada-002",
        llm_model="gpt-3.5-turbo",
        persist_directory="./chroma_db"
    )

    # Initialize the system
    rag_system.initialize()

    # Run interactive query session
    rag_system.interactive_query()


class RAGSystem:
    def __init__(self,
                 docs_dir: str = "docs/",
                 embedding_model: str = "text-embedding-ada-002",
                 llm_model: str = "gpt-3.5-turbo",
                 persist_directory: str = "./chroma_db"):
        """
        Initialize the RAG system.

        Args:
            docs_dir: Directory containing PDF files
            embedding_model: OpenAI embedding model name
            llm_model: OpenAI LLM model name
            persist_directory: Directory to persist ChromaDB
        """
        self.docs_dir = docs_dir
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.persist_directory = persist_directory

        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(model=embedding_model)

        # Initialize components (will be set later)
        self.vector_store = None
        self.retriever = None
        self.qa_chain = None

    def load_pdf_documents(self) -> List[Document]:
        """
        Load PDF documents from the specified directory.

        Returns:
            List of Document objects
        """
        logger.info(f"Loading PDF files from {self.docs_dir}")

        # Get all PDF files in the directory
        pdf_files = glob.glob(os.path.join(self.docs_dir, "*.pdf"))

        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in {self.docs_dir}")

        logger.info(f"Found {len(pdf_files)} PDF files")

        all_documents = []

        for pdf_file in pdf_files[:10]:  # Limit to 10 files as specified
            try:
                logger.info(f"Loading: {os.path.basename(pdf_file)}")

                # Load PDF using PyPDFLoader
                loader = PyPDFLoader(pdf_file)
                pages = loader.load_and_split()

                # Add metadata about the source file
                for page in pages:
                    page.metadata["source"] = os.path.basename(pdf_file)

                all_documents.extend(pages)
                logger.info(f"Loaded {len(pages)} pages from {os.path.basename(pdf_file)}")

            except Exception as e:
                logger.error(f"Error loading {pdf_file}: {str(e)}")
                continue

        logger.info(f"Total pages loaded: {len(all_documents)}")
        return all_documents

    def split_documents(self, documents: List[Document],
                        chunk_size: int = 1000,
                        chunk_overlap: int = 200) -> List[Document]:
        """
        Split documents into chunks for better retrieval.

        Args:
            documents: List of Document objects
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks

        Returns:
            List of chunked Document objects
        """
        logger.info(f"Splitting {len(documents)} documents into chunks")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        chunked_documents = text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunked_documents)} chunks")

        return chunked_documents

    def create_vector_store(self, documents: List[Document]):
        """
        Create and persist vector store from documents.

        Args:
            documents: List of Document objects
        """
        logger.info("Creating vector store...")

        # Create Chroma vector store
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )

        # Persist the database
        self.vector_store.persist()
        logger.info(f"Vector store created and persisted to {self.persist_directory}")

        # Create retriever
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 4}  # Retrieve top 4 relevant chunks
        )

    def create_qa_chain(self):
        """
        Create the question-answering chain.
        """
        logger.info("Creating QA chain...")

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=OpenAI(model=self.llm_model, temperature=0),
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            verbose=True
        )
        logger.info("QA chain created successfully")

    def setup_llama_index(self):
        """
        Optional: Setup LlamaIndex alongside LangChain.
        This provides an alternative indexing approach.
        """
        if not HAVE_LLAMA_INDEX:
            logger.warning("LlamaIndex not available. Skipping LlamaIndex setup.")
            return None

        logger.info("Setting up LlamaIndex...")

        # Load documents with LlamaIndex
        documents = SimpleDirectoryReader(self.docs_dir).load_data()

        # Setup Chroma vector store for LlamaIndex
        chroma_client = chromadb.PersistentClient(path=self.persist_directory)
        chroma_collection = chroma_client.create_collection("llama_index_collection")

        vector_store = ChromaVectorStore(
            chroma_collection=chroma_collection,
            embed_model=self.embeddings
        )

        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Create index
        index = GPTVectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context
        )

        logger.info("LlamaIndex setup complete")
        return index

    def initialize(self):
        """
        Initialize the complete RAG system.
        """
        logger.info("Initializing RAG system...")

        # Step 1: Load PDF documents
        documents = self.load_pdf_documents()

        # Step 2: Split documents into chunks
        chunked_documents = self.split_documents(documents)

        # Step 3: Create vector store
        self.create_vector_store(chunked_documents)

        # Step 4: Create QA chain
        self.create_qa_chain()

        # Step 5: Optional - Setup LlamaIndex
        if HAVE_LLAMA_INDEX:
            self.llama_index = self.setup_llama_index()

        logger.info("RAG system initialized successfully")

    def query(self, question: str) -> dict:
        """
        Query the RAG system with a question.

        Args:
            question: User's question

        Returns:
            Dictionary containing answer and source documents
        """
        if not self.qa_chain:
            raise ValueError("RAG system not initialized. Call initialize() first.")

        logger.info(f"Processing query: {question}")

        try:
            # Get response from QA chain
            result = self.qa_chain({"query": question})

            # Format response
            response = {
                "answer": result["result"],
                "sources": []
            }

            # Add source information
            if "source_documents" in result:
                for doc in result["source_documents"]:
                    source_info = {
                        "content": doc.page_content[:200] + "...",  # First 200 chars
                        "source": doc.metadata.get("source", "Unknown"),
                        "page": doc.metadata.get("page", "N/A")
                    }
                    response["sources"].append(source_info)

            return response

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "answer": f"Error: {str(e)}",
                "sources": []
            }

    def interactive_query(self):
        """
        Run an interactive query session.
        """
        print("\n" + "=" * 50)
        print("RAG System - Interactive Query Mode")
        print("Type 'exit' or 'quit' to end the session")
        print("=" * 50)

        while True:
            try:
                question = input("\nEnter your question: ").strip()

                if question.lower() in ['exit', 'quit']:
                    print("Exiting...")
                    break

                if not question:
                    continue

                # Get answer
                response = self.query(question)

                # Display answer
                print("\n" + "=" * 50)
                print("ANSWER:")
                print(response["answer"])
                print("=" * 50)

                # Display sources
                if response["sources"]:
                    print("\nSOURCES:")
                    for i, source in enumerate(response["sources"], 1):
                        print(f"\nSource {i}:")
                        print(f"  Document: {source['source']}")
                        print(f"  Page: {source['page']}")
                        print(f"  Content: {source['content']}")
                print("=" * 50)

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {str(e)}")


if __name__ == "__main__":
    # Install required packages if not already installed
    # pip install langchain openai chromadb pypdf

    main()