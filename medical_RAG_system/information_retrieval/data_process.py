from pathlib import Path
from document_encoding.text_chunking import TextChunking
from document_encoding.bioBERT_encoder import bioBERTEncoder
from elastic_container.elastic_indexing import ElasticIndexing
from faiss_container.faiss_insert_data import FaissData

class DataProcess:
    def __init__(self):
        self.pdf_document_path = Path('../data/pdf_document')

    def process(self):
        pdf_paths = [str(p.resolve()) for p in self.pdf_document_path.glob("*.pdf")]

        # Text chunking lấy dữ liệu từ pdf_documents để xử lý và thêm vào source/text_chunked.jsonl
        text_chunking = TextChunking()
        text_chunking.pdf_chungking(pdf_paths=pdf_paths)

        # ElasticSearch lấy dữ liệu từ source/text_chunked.jsonl để thêm vào index
        elastic_index_name = "med_index"
        elastic_indexing = ElasticIndexing(elastic_index_name)
        elastic_indexing.indexing_documents()

        # bioBERT lấy dữ liệu từ source/text_chunked.jsonl để encode và lưu vào target/text_chunked.jsonl
        encoder = bioBERTEncoder()
        encoder.embed_file()

        # FAISS lấy dữ liệu từ target/text_chunked.jsonl
        faiss_insert_data = FaissData()
        faiss_insert_data.insert_data()

if __name__ == "__main__":
    data_process = DataProcess()
    data_process.process()