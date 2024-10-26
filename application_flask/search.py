import hnswlib
import json

class HNSW_search_tool:

    def __init__(self, dim, space, hnsw_path, ef, json_path):
        
        self.hnsw_index = hnswlib.Index(space=space, dim=dim)
        self.hnsw_index.load_index(str(hnsw_path))
        self.hnsw_index.set_ef(ef)

        with open(str(json_path), 'r') as f:
            self.image_metadata = json.load(f)
        
        print('Search Engine has loaded')

    def search_in_hnsw(self, query_embedding, k=5):
        
        labels, distances = self.hnsw_index.knn_query(query_embedding, k=k)
        results = [self.image_metadata[str(label)] for label in labels[0]]
        
        return results