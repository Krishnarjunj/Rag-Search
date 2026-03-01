from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
import json

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embeddings(self, text):
        if text.strip()=="":
            raise ValueError

        # input_list = text.strip().split(" ")

        output_list = self.model.encode([text])

        return output_list[0]

    def build_embeddings(self, documents):
        file_path_cache = Path("~/Krish/RAG/rag-search-engine/cache/movie_embeddings.npy").expanduser()
        self.documents = documents

        term_list = []

        for document in documents:
            self.document_map[document["id"]] = document
            term = f"{document['title']}: {document['description']}"
            term_list.append(term)

        self.embeddings = self.model.encode(term_list, show_progress_bar=True)
        np.save(file_path_cache, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        file_path_cache = Path("~/Krish/RAG/rag-search-engine/cache/movie_embeddings.npy").expanduser()
        self.documents = documents

        for document in documents:
            self.document_map[document['id']] = document

        if file_path_cache.exists():
            try:
                self.embeddings = np.load(file_path_cache)

                if len(self.embeddings) == len(documents):
                    return self.embeddings
                else:
                    self.build_embeddings(documents)
                    return
            except:
                print("Cache corrupted")
        return self.build_embeddings(documents)


def verify_embeddings():
    Obj = SemanticSearch()
    file_path_movies = Path("~/Krish/RAG/rag-search-engine/data/movies.json").expanduser()
    with open(file_path_movies, 'r') as f:
        data_json = json.load(f)

    Obj.load_or_create_embeddings(data_json["movies"])
    print(f"Number of docs:   {len(Obj.documents)}")
    print(f"Embeddings shape: {Obj.embeddings.shape[0]} vectors in {Obj.embeddings.shape[1]} dimensions")

def verify_model():
    Obj = SemanticSearch()
    print(f"Model loaded: {Obj.model}")
    print(f"Max sequence length: {Obj.model.max_seq_length}")

def embed_text(text):
    Obj = SemanticSearch()
    embedding = Obj.generate_embeddings(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")
