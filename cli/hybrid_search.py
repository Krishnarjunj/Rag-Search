from pathlib import Path
from typing import Any

from keyword_search import InvertedIndex
from semantic_search import ChunkedSemanticSearch


def rrf_score(rank: int, k: int) -> float:
    return 1 / (k + rank)


class HybridSearch:
    def __init__(self, documents: list[dict[str, Any]]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        self.index_path = Path("~/Krish/RAG/rag-search-engine/cache/index.pkl").expanduser()

        if not self.index_path.exists():
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int):
        self.idx.load()
        return self.idx.bm25_search(query, limit, print_results=False)

    @staticmethod
    def normalize(scores: list[float]) -> list[float]:
        if not scores:
            return []

        min_score = min(scores)
        max_score = max(scores)

        if min_score == max_score:
            return [1.0] * len(scores)

        score_range = max_score - min_score
        return [(score - min_score) / score_range for score in scores]

    @staticmethod
    def hybrid_score(keyword_score: float, semantic_score: float, alpha: float) -> float:
        return (alpha * keyword_score) + ((1 - alpha) * semantic_score)

    @staticmethod
    def rrf_score(rank: int, k: int) -> float:
        return rrf_score(rank, k)

    def weighted_search(self, query: str, alpha: float, limit: int = 5):
        expanded_limit = max(limit * 500, limit)

        bm25_results = self._bm25_search(query, expanded_limit)
        semantic_results = self.semantic_search.search(query, expanded_limit)

        normalized_keyword_scores = self.normalize(
            [result["score"] for result in bm25_results]
        )
        normalized_semantic_scores = self.normalize(
            [result["score"] for result in semantic_results]
        )

        documents_by_id = {}

        for result, normalized_score in zip(bm25_results, normalized_keyword_scores):
            doc_id = result["id"]
            documents_by_id[doc_id] = {
                "id": doc_id,
                "title": result["title"],
                "document": result["document"],
                "metadata": result.get("metadata", {}),
                "keyword_score": normalized_score,
                "semantic_score": 0.0,
            }

        for result, normalized_score in zip(semantic_results, normalized_semantic_scores):
            doc_id = result["id"]

            if doc_id not in documents_by_id:
                documents_by_id[doc_id] = {
                    "id": doc_id,
                    "title": result["title"],
                    "document": result["document"],
                    "metadata": result.get("metadata", {}),
                    "keyword_score": 0.0,
                    "semantic_score": normalized_score,
                }
                continue

            documents_by_id[doc_id]["semantic_score"] = normalized_score

        results = []

        for document in documents_by_id.values():
            document["hybrid_score"] = self.hybrid_score(
                document["keyword_score"],
                document["semantic_score"],
                alpha,
            )
            results.append(document)

        results.sort(key=lambda document: document["hybrid_score"], reverse=True)
        return results

    def rrf_search(self, query: str, k: int, limit: int = 5):
        expanded_limit = max(limit * 500, limit)

        bm25_results = self._bm25_search(query, expanded_limit)
        semantic_results = self.semantic_search.search(query, expanded_limit)

        documents_by_id = {}

        for rank, result in enumerate(bm25_results, start=1):
            doc_id = result["id"]
            documents_by_id[doc_id] = {
                "id": doc_id,
                "title": result["title"],
                "document": result["document"],
                "metadata": result.get("metadata", {}),
                "bm25_rank": rank,
                "semantic_rank": None,
                "rrf_score": self.rrf_score(rank, k),
            }

        for rank, result in enumerate(semantic_results, start=1):
            doc_id = result["id"]
            semantic_rrf_score = self.rrf_score(rank, k)

            if doc_id not in documents_by_id:
                documents_by_id[doc_id] = {
                    "id": doc_id,
                    "title": result["title"],
                    "document": result["document"],
                    "metadata": result.get("metadata", {}),
                    "bm25_rank": None,
                    "semantic_rank": rank,
                    "rrf_score": semantic_rrf_score,
                }
                continue

            documents_by_id[doc_id]["semantic_rank"] = rank
            documents_by_id[doc_id]["rrf_score"] += semantic_rrf_score

        results = list(documents_by_id.values())
        results.sort(key=lambda document: document["rrf_score"], reverse=True)
        return results
