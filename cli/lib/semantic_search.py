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

        text = text.strip()

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

    def search(self, query, limit):
        if self.embeddings is None or self.documents is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")

        query_embedding = self.generate_embeddings(query)

        results = []

        for idx, doc_embedding in enumerate(self.embeddings):
            similarity = cosine_similarity(query_embedding, doc_embedding)
            document = self.documents[idx]

            results.append((similarity, document))

        results.sort(key=lambda x: x[0], reverse=True)

        top_results = []

        for score, document in results[:limit]:
            top_results.append(f"{document["title"]} (score: {float(score)})\n")

        return top_results


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

def embed_query_text(query):
    Obj = SemanticSearch()
    embedding = Obj.generate_embeddings(query)

    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)
