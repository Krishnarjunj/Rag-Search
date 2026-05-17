import json
import re
from pathlib import Path

import numpy as np

from lib.semantic_search import SemanticSearch, cosine_similarity


SCORE_PRECISION = 4


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    @staticmethod
    def semantic_chunk_text(text: str, chunk_size: int = 4, overlap: int = 1) -> list[str]:
        text = text.strip()
        if not text:
            return []

        sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text)]
        sentences = [sentence for sentence in sentences if sentence]

        if len(sentences) == 1 and not re.search(r"[.!?]$", sentences[0]):
            sentences = [text]

        if not sentences:
            return []

        step = chunk_size - overlap
        if step <= 0:
            step = chunk_size

        chunks = []
        for i in range(0, len(sentences), step):
            chunk_sentences = sentences[i:i + chunk_size]
            if not chunk_sentences:
                continue
            if len(chunk_sentences) == 1:
                chunks.append(chunk_sentences[0])
                continue
            chunks.append(" ".join(chunk_sentences))

        return chunks

    def semantic_chunk(self, text: str, chunk_size: int = 4, overlap: int = 1) -> list[str]:
        return self.semantic_chunk_text(text, chunk_size=chunk_size, overlap=overlap)

    def _build_chunk_records(self, documents: list[dict]) -> tuple[list[str], list[dict]]:
        all_chunks = []
        chunk_metadata = []

        for movie_idx, document in enumerate(documents):
            description = document.get("description", "")
            if description.strip() == "":
                continue

            chunks = self.semantic_chunk(description, chunk_size=4, overlap=1)
            total_chunks = len(chunks)

            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_metadata.append(
                    {
                        "movie_idx": movie_idx,
                        "chunk_idx": chunk_idx,
                        "total_chunks": total_chunks,
                    }
                )

        return all_chunks, chunk_metadata

    def build_chunk_embeddings(self, documents):
        embeddings_path = Path("cache/chunk_embeddings.npy")
        metadata_path = Path("cache/chunk_metadata.json")

        self.documents = documents
        self.document_map = {}

        all_chunks, chunk_metadata = self._build_chunk_records(documents)

        for document in documents:
            self.document_map[document["id"]] = document

        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata

        np.save(embeddings_path, self.chunk_embeddings)
        with open(metadata_path, "w") as f:
            json.dump({"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2)

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        embeddings_path = Path("cache/chunk_embeddings.npy")
        metadata_path = Path("cache/chunk_metadata.json")

        self.documents = documents
        self.document_map = {}

        for document in documents:
            self.document_map[document["id"]] = document

        if embeddings_path.exists() and metadata_path.exists():
            self.chunk_embeddings = np.load(embeddings_path)
            with open(metadata_path, "r") as f:
                metadata_json = json.load(f)
            self.chunk_metadata = metadata_json.get("chunks", [])
            _, expected_metadata = self._build_chunk_records(documents)
            expected_total = len(expected_metadata)
            cached_total = len(self.chunk_embeddings)

            if cached_total == expected_total and len(self.chunk_metadata) == expected_total:
                return self.chunk_embeddings
            return self.build_chunk_embeddings(documents)

        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10):
        if self.chunk_embeddings is None or self.chunk_metadata is None or self.documents is None:
            raise ValueError("No chunk embeddings loaded. Call `load_or_create_chunk_embeddings` first.")

        query_embedding = self.generate_embeddings(query)

        chunk_scores = []

        for chunk_vector, metadata in zip(self.chunk_embeddings, self.chunk_metadata):
            score = cosine_similarity(chunk_vector, query_embedding)
            chunk_scores.append(
                {
                    "chunk_idx": metadata["chunk_idx"],
                    "movie_idx": metadata["movie_idx"],
                    "score": score,
                }
            )

        movie_scores = {}

        for chunk_score in chunk_scores:
            movie_idx = chunk_score["movie_idx"]
            if movie_idx not in movie_scores or chunk_score["score"] > movie_scores[movie_idx]["score"]:
                movie_scores[movie_idx] = chunk_score

        sorted_movie_scores = sorted(
            movie_scores.values(),
            key=lambda item: item["score"],
            reverse=True,
        )

        results = []

        for movie_score in sorted_movie_scores[:limit]:
            document = self.documents[movie_score["movie_idx"]]
            doc_id = document["id"]
            title = document["title"]
            doc_text = document.get("description", "")
            metadata = document.get("metadata")

            results.append(
                {
                    "id": doc_id,
                    "title": title,
                    "document": doc_text[:100],
                    "score": round(movie_score["score"], SCORE_PRECISION),
                    "metadata": metadata or {},
                }
            )

        return results

    def search(self, query: str, limit: int = 10):
        return self.search_chunks(query, limit)
