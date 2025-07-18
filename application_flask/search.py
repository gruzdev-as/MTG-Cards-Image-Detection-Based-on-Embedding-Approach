import json
from pathlib import Path
from typing import Literal

import hnswlib
from numpy import ndarray


class HNSWSearchTool:
    """Nearest neigbour searching tool."""

    def __init__(self, dim:int, space:Literal["cosine", "l2"], hnsw_path:Path, ef:int, json_path:Path) -> None:
        self.hnsw_index = hnswlib.Index(space=space, dim=dim)
        self.hnsw_index.load_index(str(hnsw_path))
        self.hnsw_index.set_ef(ef)

        with Path(json_path).open("r") as f:
            self.image_metadata = json.load(f)

        print("Search Engine has loaded")

    def search_in_hnsw(self, query_embedding: ndarray, k: int=5) -> list[dict[str, str]]:
        """Use hnsw index to retrieval info."""
        labels, distances = self.hnsw_index.knn_query(query_embedding, k=k)

        return [self.image_metadata[str(label)] for label in labels[0]]
