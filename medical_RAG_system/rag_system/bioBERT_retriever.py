from elasticsearch import Elasticsearch
import os
import requests
import json
from medical_RAG_system.information_retrieval.document_encoding.bioBERT_encoder import bioBERTEncoder
from dotenv import load_dotenv

class BioBERTRetriever:
    def __init__(self):
        load_dotenv(dotenv_path="pass.env", override=True)
        elastic_password = "7n2xK2kELC0GYsOCyi9+"
        ca_cert=r"C:\Users\Dung\Downloads\elasticsearch-9.2.0-windows-x86_64\elasticsearch-9.2.0\config\certs\http_ca.crt"
        self.es = Elasticsearch(
            ['https://localhost:9200'],
            basic_auth=('elastic', elastic_password),
            verify_certs=True,
            ca_certs=ca_cert,
            request_timeout=60
        )
        self.index = "test_index"
        self.faiss_url = "http://localhost:5000/search"
        self.query_encoder = bioBERTEncoder()

    def query_to_vector(self, text: str):
        """Converts text query to a vector using the BioBERT encoder."""
        embedding = self.query_encoder.encode(text)
        return embedding

    def faiss_query(self, query: str, k: int = 2):
        """Performs a vector search using FAISS with the given query and k."""
        vec = self.query_to_vector(query).tolist()  # Convert numpy array to list
        data = {
            'queries': [vec],  # List of vectors
            'k': k
        }
        response = requests.post(self.faiss_url, headers={'Content-Type': 'application/json'}, data=json.dumps(data))
        return response.json()

    def get_docs_via_IDs(self, IDs: list):
        query = {
            "size": len(IDs),
            "query": {
                "terms": {
                    "id": IDs
                }
            },
            "_source": ["id", "title", "text_chunked"]
        }
        return self.es.search(index=self.index, body=query)

    def retrieve_docs(self, query: str, k: int = 5):
        """Retrieves documents relevant to the query using both FAISS and Elasticsearch."""
        response = self.faiss_query(query, k)
        IDs = response['ids'][0]  # Assumes PMIDs are returned in a structured list
        es_response = self.get_docs_via_IDs(IDs)
        # Có thể thêm re-rank vào đây
        results = {}

        # Formatting the response as required
        for idx, hit in enumerate(es_response['hits']['hits'], 1):
            doc_key = f"doc{idx}"
            results[doc_key] = {
                'id': hit['_source']['id'],
                'title': hit['_source']['title'],
                'text_chunked': hit['_source']['text_chunked']
            }

        return json.dumps(results, indent=4)

if __name__ == '__main__':
    retriever = BioBERTRetriever()
    query = "Tell me about multi teacher KD?"
    n_docs = 5
    response = retriever.retrieve_docs(query, n_docs)
    print(response)