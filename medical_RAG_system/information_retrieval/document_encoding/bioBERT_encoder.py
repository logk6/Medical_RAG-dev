import json
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer, models

class bioBERTEncoder:
    def __init__(self, max_length=512):
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        self.max_length = max_length
        
        word_embedding_model = models.Transformer('dmis-lab/biobert-v1.1', max_seq_length=self.max_length)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                       pooling_mode_mean_tokens=True,
                                       pooling_mode_cls_token=False,
                                       pooling_mode_max_tokens=False)

        self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=self.device)

    def pdf_encode(self, item):
        contents = item["text_chunked"]
        embeddings = self.model.encode(contents, batch_size=256, show_progress_bar=False)
        return{
            "id": str(item["id"]),
            "title": item["title"],
            "embeddings": embeddings.tolist()
        }

    # Source file là file pdf đã được chunking và chuyển thành jsonl
    # Target là file jsonl giống source nhưng thêm embedding, thay cho cột text_chunked
    def embed_file(self):
        source_file = Path('../../data/embed_data/source/text_chunked.jsonl')
        target_file = Path('../../data/embed_data/target/text_chunked.jsonl')
        with open(target_file, 'w', encoding='utf-8') as target:
            with open(source_file, 'r', encoding='utf-8') as source:
                for line in source:
                    # Mỗi dòng là một đối tượng JSON
                    item = json.loads(line)
                    embedded_item = self.pdf_encode(item)
                    # Ghi đối tượng đã embed vào tệp đích
                    target.write(json.dumps(embedded_item) + '\n')

    def encode(self, text):
        # Biến đổi text(câu query) thành vector d=768 chiều
        embedding = self.model.encode([text], batch_size=1, show_progress_bar=False)
        return embedding[0]

if __name__ == "__main__":
    # Đây là phần test hàm
    encoder = bioBERTEncoder()
    encoder.embed_file()

