import logging
from pathlib import Path
from typing import List, Tuple

try:
    import chromadb
except ModuleNotFoundError:
    raise SystemExit("Missing dependency: run `pip install chromadb`")

try:
    from sentence_transformers import SentenceTransformer
except ModuleNotFoundError:
    raise SystemExit("Missing dependency: run `pip install sentence-transformers`")

# configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"
KB_PATH = Path("./knowledge_base")
COLLECTION_NAME = "faq"

model = SentenceTransformer(MODEL_NAME)

def get_client_and_collection(path: Path, collection_name: str):
    # Use a persistent client (existing behaviour), but handle creation safely.
    client = chromadb.PersistentClient(path=str(path))
    collection = client.get_or_create_collection(name=collection_name)
    return client, collection

def build_kb(collection, knowledge: List[str], force_rebuild: bool = False):
    """Add knowledge entries if collection is empty or force_rebuild=True."""
    # if forced, you may want to clear existing contents (careful with production data)
    if force_rebuild:
        try:
            collection.delete()  # if API supports this; otherwise remove items by id
        except Exception:
            logger.warning("Collection delete not supported; continuing")

    # avoid duplicate inserts: only add if collection is empty
    try:
        existing_count = getattr(collection, "count", lambda: None)()
    except Exception:
        existing_count = None

    if existing_count:
        logger.info("Collection already has items, skipping add.")
        return

    # Batch-encode and add all at once (faster)
    embeddings = model.encode(knowledge, convert_to_numpy=True)
    ids = [f"kb_{i}" for i in range(len(knowledge))]
    # Convert embeddings to lists if required by chroma
    collection.add(embeddings=[e.tolist() for e in embeddings], documents=knowledge, ids=ids)
    logger.info("Knowledge base populated with %d items", len(knowledge))

def transform_distance_to_similarity(distance: float, metric: str = "cosine") -> float:
    """Convert returned distance to a similarity score in [0,1], best-effort.

    For cosine distance in chroma the returned distance is typically (1 - cosine_similarity),
    so similarity = 1 - distance. This routine centralizes that logic; adjust if you use another metric.
    """
    if metric == "cosine":
        return max(0.0, 1.0 - distance)
    # fallback: return a naive mapping (clamp)
    return max(0.0, min(1.0, 1.0 - distance))

def ask_question(collection, question: str, top_k: int = 2) -> List[Tuple[str, str, float]]:
    query_embedding = model.encode(question, convert_to_numpy=True)
    results = collection.query(query_embeddings=[query_embedding.tolist()], n_results=top_k)

    docs = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]

    metric = "cosine"  # document or config: ensure this matches how the collection was configured
    output = []
    for i, doc in enumerate(docs):
        dist = distances[i] if i < len(distances) else None
        score = transform_distance_to_similarity(dist, metric) if dist is not None else None
        output.append((f"doc_{i}", doc, score))
    return output

def main():
    client, collection = get_client_and_collection(KB_PATH, COLLECTION_NAME)

    knowledge = [
        "Vector databases store data as high-dimensional vectors for similarity search",
        "Embeddings are numerical representations of text, images, or other data",
        "Cosine similarity measures the angle between two vectors",
        "ChromaDB is an open-source vector database built for AI applications",
        "FAISS is a library by Facebook for efficient similarity search",
    ]

    build_kb(collection, knowledge)

    for q in ["What are embeddings?", "How do I measure vector similarity?"]:
        print(f"\nQuestion: {q}\n")
        results = ask_question(collection, q)
        print("Relevant information:")
        for rank, (_, doc, score) in enumerate(results, start=1):
            print(f"{rank}. {doc}")
            if score is not None:
                print(f"   (Estimated similarity: {score:.4f})")
            else:
                print("   (No score available)")

if __name__ == "__main__":
    main()