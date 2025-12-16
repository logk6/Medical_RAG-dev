import fitz
import os
import json
import re
import random
from pathlib import Path
from tqdm.auto import tqdm

class TextChunking:

    def text_formatter(self, text: str) -> str:
        """Xử lí clean text."""
        cleaned_text = text.replace("\n", " ").strip()
        return cleaned_text

    def open_and_read_pdf(self, pdf_path: str) -> list[dict]:
        doc = fitz.open(pdf_path)
        pages_and_texts = []
        for page_number, page in tqdm(enumerate(doc)):  # lặp từng trang trong doc
            text = page.get_text()  # Lấy text
            text = self.text_formatter(text) # Clean text
            pages_and_texts.append({"page_number": page_number,
                                    "page_char_count": len(text),
                                    "page_word_count": len(text.split(" ")),
                                    "page_sentence_count_raw": len(text.split(". ")),
                                    "page_token_count": len(text) / 4,  # 1 token = ~4 chars
                                    "text": text})
        return pages_and_texts

    def get_next_id(self):
        counter_file = Path("../../data/id_counter.txt")
        if counter_file.exists():
            next_id = int(counter_file.read_text())
        else:
            next_id = 1
        return next_id

    def set_next_id(self, next_id):
        counter_file = Path("../../data/id_counter.txt")
        counter_file.write_text(str(next_id))

    def pdf_chungking(self, pdf_paths: list):
        next_id = self.get_next_id()
        for idx, pdf_path in enumerate(tqdm(pdf_paths)):
            if not isinstance(pdf_path, str):
                raise TypeError("pdf_path must be a string")
            if not os.path.isfile(pdf_path):
                raise FileNotFoundError(f"File not found: {pdf_path}")
            pages_and_texts = self.open_and_read_pdf(pdf_path=pdf_path)

            from spacy.lang.en import English
            nlp = English()
            nlp.add_pipe("sentencizer")
            for item in tqdm(pages_and_texts):
                item["sentences"] = list(nlp(item["text"]).sents) # Biến giá trị trong text thành list các câu
                item["sentences"] = [str(sentence) for sentence in item["sentences"]] # Đảm bảo mỗi câu đều là str
                item["page_sentence_count_spacy"] = len(item["sentences"])

            num_sentence_chunk_size = 10
            def split_list(input_list: list, slice_size: int) -> list[list[str]]:
                return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]
            """Phải trả về list của các list[str] vì sau đó sẽ có bước tách mỗi chunk là 1 list[str].
               Cách tách: mỗi 10 câu sẽ thành 1 list. VD lúc đầu sentence có 17 câu, thì sẽ tách thành 1 list 10 câu và 1 list 7 câu: [ [10], [7] ]. """

            for item in tqdm(pages_and_texts):
                item["sentence_chunks"] = split_list(input_list=item["sentences"],
                                                     slice_size=num_sentence_chunk_size)
                item["num_chunks"] = len(item["sentence_chunks"])

            import re

            # Tách mỗi chunk thành 1 item riêng
            pages_and_chunks = []
            for item in tqdm(pages_and_texts):
                for sentence_chunk in item["sentence_chunks"]:
                    chunk_dict = {}
                    chunk_dict["page_number"] = item["page_number"]

                    # Trước đó thì mỗi chunk là 1 list chứa nhiều câu. Bây giờ sẽ ghép lại thành 1 câu, ko còn list nữa
                    joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
                    joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk) # ".A" -> ". A" for any full-stop/capital letter combo
                    chunk_dict["sentence_chunk"] = joined_sentence_chunk

                    # Get stats about the chunk
                    chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
                    chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
                    chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4 # 1 token = ~4 characters

                    pages_and_chunks.append(chunk_dict)

            # Lấy ra 1 list tất cả text, mỗi phần từ là 1 chunk, mỗi chunk là 1 string đã được xử lý.
            # text_chunks = [item["sentence_chunk"] for item in pages_and_chunks]
            file_stem = Path(pdf_path).stem
            for c_idx, item in enumerate(pages_and_chunks):
                data_chunked = {
                    "id": str(next_id),
                    "title": str(file_stem),
                    "text_chunked": item["sentence_chunk"],
                }
                source_text_chunked = Path('../../data/embed_data/source/text_chunked.jsonl')
                with open(source_text_chunked, "a", encoding="utf-8") as f:
                    f.write(json.dumps(data_chunked, ensure_ascii=False) + "\n")
                next_id += 1

        self.set_next_id(next_id)

if __name__ == "__main__":
    # Đây là phần test hàm
    pdf_paths = ["D:\DUNG\MasterProgram\Project_I\Medical_RAG_Project\medical_RAG_system\data\pdf_document\Sport_injury_prevent_2009.pdf",
                 "D:\DUNG\MasterProgram\Project_I\Medical_RAG_Project\medical_RAG_system\data\pdf_document\Sports_Rehabilitation_and_Injury_Prevention.pdf",
                 ]
    text_chunking = TextChunking()
    text_chunking.pdf_chungking(pdf_paths=pdf_paths)