import requests
import json
from typing import List


class RAGClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')

    def query(self, question: str, **kwargs) -> dict:
        """Send a single query to the RAG API."""
        url = f"{self.base_url}/query"

        payload = {
            "question": question,
            **kwargs
        }

        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()

    def batch_query(self, questions: List[str], **kwargs) -> dict:
        """Send multiple queries in batch."""
        url = f"{self.base_url}/batch-query"

        payload = {
            "queries": questions,
            **kwargs
        }

        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()

    def get_status(self) -> dict:
        """Get system status."""
        url = f"{self.base_url}/status"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

    def list_documents(self) -> dict:
        """List available documents."""
        url = f"{self.base_url}/documents"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()


# Example usage
if __name__ == "__main__":
    client = RAGClient()

    # Check system status
    status = client.get_status()
    print(f"System status: {status['status']}")

    # Query the system
    response = client.query(
        question="What is the main topic of the documents?",
        return_sources=True,
        top_k=3
    )

    print(f"Question: {response['question']}")
    print(f"Answer: {response['answer']}")
    print(f"Processing time: {response['processing_time']}s")

    if response['sources']:
        print("\nSources used:")
        for i, source in enumerate(response['sources'], 1):
            print(f"{i}. {source['source']} (Page {source.get('page', 'N/A')})")
            print(f"   Excerpt: {source['content']}")