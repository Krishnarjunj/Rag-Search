from sentence_transformers import SentenceTransformer
from semantic_search import SemanticSearch
import numpy as np
from pathlib import Path
import json

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents):
        file_path_cache = Path("~/Krish/RAG/rag-search-engine/cache/movie_embeddings.npy").expanduser()
        self.documents = documents

        list_str = []
        chunk_metadata = {}


