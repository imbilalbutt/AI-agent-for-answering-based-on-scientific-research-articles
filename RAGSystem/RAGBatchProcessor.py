import json
from datetime import datetime


class RAGBatchProcessor:
    def __init__(self, rag_system):
        self.rag_system = rag_system

    def process_queries_from_file(self, input_file: str, output_file: str):
        """
        Process multiple queries from a file.

        Args:
            input_file: JSON file containing queries
            output_file: JSON file to save results
        """
        with open(input_file, 'r') as f:
            queries = json.load(f)

        results = []
        for query_data in queries:
            question = query_data.get("question", "")
            query_id = query_data.get("id", "")

            if question:
                print(f"Processing query {query_id}: {question[:50]}...")

                response = self.rag_system.query(question)

                result = {
                    "id": query_id,
                    "question": question,
                    "answer": response["answer"],
                    "sources": response["sources"],
                    "timestamp": datetime.now().isoformat()
                }
                results.append(result)

        # Save results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Processed {len(results)} queries. Results saved to {output_file}")

# Example usage in main:
# if __name__ == "__main__":
#     rag_system = RAGSystem()
#     rag_system.initialize()
#
#     processor = RAGBatchProcessor(rag_system)
#     processor.process_queries_from_file("queries.json", "results.json")