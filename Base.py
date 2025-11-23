import numpy as np
from typing import List, Tuple, Optional


class SimpleVectorDB:
    def __init__(self):
        self.vectors: List[np.ndarray] = []
        self.documents: List[str] = []
        self.ids: List[str] = []

    def add(self, vectors: List[List[float]], documents: List[str], ids: List[str]):
        """Add vectors to the database with validation.

        Ensures input lengths match and stores vectors as 1-D float arrays.
        """
        if not (len(vectors) == len(documents) == len(ids)):
            raise ValueError("`vectors`, `documents`, and `ids` must have the same length")

        processed = []
        for v in vectors:
            arr = np.asarray(v, dtype=float).ravel()
            if arr.ndim != 1:
                raise ValueError("Each vector must be one-dimensional")
            processed.append(arr)

        # If existing vectors present, ensure dimensional consistency
        if self.vectors:
            expected_dim = self.vectors[0].shape[0]
            for arr in processed:
                if arr.shape[0] != expected_dim:
                    raise ValueError(f"All vectors must have the same dimension ({expected_dim})")

        self.vectors.extend(processed)
        self.documents.extend(documents)
        self.ids.extend(ids)

    def cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors with zero-norm safety."""
        dot_product = float(np.dot(v1, v2))
        norm_v1 = float(np.linalg.norm(v1))
        norm_v2 = float(np.linalg.norm(v2))

        if norm_v1 == 0.0 or norm_v2 == 0.0:
            return 0.0

        return dot_product / (norm_v1 * norm_v2)

    def search(self, query_vector: List[float], top_k: int = 5) -> List[Tuple[str, str, float]]:
        """Search for most similar vectors to `query_vector` and return top_k results.

        Returns list of tuples: (id, document, similarity)
        """
        if not self.vectors:
            return []

        query_vec = np.asarray(query_vector, dtype=float).ravel()
        if query_vec.ndim != 1:
            raise ValueError("`query_vector` must be one-dimensional")

        expected_dim = self.vectors[0].shape[0]
        if query_vec.shape[0] != expected_dim:
            raise ValueError(f"`query_vector` dimension ({query_vec.shape[0]}) does not match stored vectors ({expected_dim})")

        similarities: List[Tuple[str, str, float]] = []
        for i, vec in enumerate(self.vectors):
            similarity = self.cosine_similarity(query_vec, vec)
            similarities.append((self.ids[i], self.documents[i], similarity))

        similarities.sort(key=lambda x: x[2], reverse=True)
        return similarities[:top_k]


# Example usage with sentence-transformers (guarded)
try:
    from sentence_transformers import SentenceTransformer
except ModuleNotFoundError:
    SentenceTransformer = None  # type: ignore


def _demo():
    if SentenceTransformer is None:
        print("Missing dependency: install with `pip install sentence-transformers numpy`")
        raise SystemExit(1)

    # Initialize
    model = SentenceTransformer("all-MiniLM-L6-v2")
    db = SimpleVectorDB()

    # Add documents
    documents = [
        "Python is a versatile programming language",
        "Machine learning requires data preprocessing",
        "Vector databases enable fast similarity search",
        "Natural language processing uses embeddings",
    ]

    embeddings = model.encode(documents, convert_to_numpy=True)
    ids = [f"doc_{i}" for i in range(len(documents))]

    db.add([e.tolist() for e in embeddings], documents, ids)

    # Search
    query = "What is Python used for?"
    query_embedding = model.encode(query, convert_to_numpy=True)

    results = db.search(query_embedding.tolist(), top_k=3)

    print("Search Results:")
    for id_, doc, score in results:
        print(f"\nID: {id_}")
        print(f"Document: {doc}")
        print(f"Similarity: {score:.4f}")


if __name__ == "__main__":
    _demo()


