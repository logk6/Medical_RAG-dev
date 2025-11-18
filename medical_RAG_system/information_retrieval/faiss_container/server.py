from flask import Flask, request, jsonify
import faiss
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the Faiss index (assuming the index has already been created and saved)
index_path = '../../data/faiss_indices/faiss_index.index'
index = faiss.read_index(index_path)

csv_path = "../../data/faiss_indices/faiss_csv.csv"
csv_df = pd.read_csv(csv_path)

# Create a dictionary to map index numbers to PMIDs
index_to_ids = dict(zip(csv_df['Index'], csv_df['ID']))

@app.route('/search', methods=['POST'])
def search():
    # Extract the query vectors and the value of k from the POST request
    data = request.get_json()
    queries = np.array(data['queries'], dtype='float32')
    
    # Get the number of nearest neighbors to search for
    k = int(data['k'])
    # Perform the search in the Faiss index
    distances, indices = index.search(queries, k)
    
    # Map the Faiss indices to PMIDs using the dictionary
    matched_IDs = [[index_to_ids[idx] for idx in row] for row in indices]

    # Return the response as JSON
    return jsonify(ids=matched_IDs, distances=distances.tolist())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Accessible over port 5000 on all network interfaces