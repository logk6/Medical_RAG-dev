import faiss
import json
import numpy as np
import os
import csv
from pathlib import Path
from tqdm import tqdm

class FaissInsertData:
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
                            index_num = self.index.ntotal - 1  # Index number of the last added embedding
                            csv_rows.append([id, file_name.name, index_num])
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON in file {file_name}: {e}")

        # Write the index to a file
        faiss.write_index(self.index, self.index_file)

        with open(self.csv_file, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['ID', 'Filename', 'Index'])
            csv_writer.writerows(csv_rows)

if __name__ == "__main__":
    faiss_insert_data = FaissInsertData()
    faiss_insert_data.insert_data()

