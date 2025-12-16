import faiss
import json
import numpy as np
import pandas as pd
import os
import csv
from pathlib import Path
from tqdm import tqdm

class FaissData:
    def __init__(self, d=768):
        self.csv_file = "../../data/faiss_indices/faiss_csv.csv"
        self.index_file = "../../data/faiss_indices/faiss_index.index"
        self.index = faiss.IndexFlatL2(d)

    def insert_data(self):
        source_directory = Path('../../data/embed_data/target')
        sorted_files = list(source_directory.glob('*.jsonl'))
        csv_rows = []
        for file_name in tqdm(sorted_files, desc="Processing JSONL files"):
            with open(file_name, 'r') as file:
                for line in file:
                    try:
                        data = json.loads(line)
                        embeddings = data.get('embeddings')
                        id = data.get('id')

                        # If embeddings and ID are present, add them to the index
                        if embeddings and id:
                            embeddings = np.array(embeddings, dtype='float32').reshape(1, -1)  # Convert to NumPy array and reshape
                            self.index.add(embeddings)
                            # Add PMIDs, filenames, and index numbers for ordering to the CSV
                            index_num = self.index.ntotal - 1  # Lấy số thứ tự của index (trong FAISS chỉ có thể lấy stt)
                            csv_rows.append([id, file_name.name, index_num])
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON in file {file_name}: {e}")

        # Write the index to a file
        faiss.write_index(self.index, self.index_file)
        # Phần này ghi đè chứ không ghi tiếp

        with open(self.csv_file, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerows(csv_rows)

    def search_data(self, data):
        index = faiss.read_index(self.index_file)
        csv_df = pd.read_csv(self.csv_file)

        # Tạo 1 từ điển map giữa số index và ids trong target/text_chunked.jsonl
        index_to_ids = dict(zip(csv_df['Index'], csv_df['ID']))  # {index_1: id_1, index_2: id_2}

        queries = np.array(data['queries'], dtype='float32')
        k = int(data['k'])

        # Tìm kiếm trong Faiss index
        distances, indices = index.search(queries, k)

        # Lấy ra các id ứng với các index đã tìm được
        matched_ids = [[index_to_ids[idx] for idx in row] for row in indices]

        # Return the response as JSON
        return {
            "ids": matched_ids,
            "distances": distances.tolist()
        }
        # return jsonify(ids=matched_IDs, distances=distances.tolist())

if __name__ == "__main__":
    faiss_insert_data = FaissData()
    faiss_insert_data.insert_data()

