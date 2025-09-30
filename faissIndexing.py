# build_and_query_triples_faiss.py
import json
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from collections import Counter
import logging

# Add after imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('faiss_indexing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---------- CONFIG ----------
INPUT_TRIPLES_JSON = "kg_triples_validated.json"
FAISS_INDEX_FILE = "triples_faiss.index"
EMBEDDINGS_FILE = "triples_embeddings.npy"
METADATA_FILE = "triples_metadata.json"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 256
# ----------------------------

# Cache the model globally to avoid reloading
_model_cache = None

def get_model():
    """Get cached model or load it"""
    global _model_cache
    if _model_cache is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL}...")
        _model_cache = SentenceTransformer(EMBEDDING_MODEL)
        print("Model loaded successfully!")
    return _model_cache

def load_triples(input_path):
    """Load validated triples from JSON"""
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            triples = json.load(f)
        logger.info(f"Loaded {len(triples)} triples from {input_path}")
        return triples
    except FileNotFoundError:
        logger.error(f"Triples file not found: {input_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {input_path}: {e}")
        raise

def triple_to_text(triple):
    """Convert triple to text for embedding"""
    subj = triple.get("subject", "").strip()
    pred = triple.get("predicate", "").strip()
    obj = triple.get("object", "").strip()
    title = triple.get("title", "").strip()
    
    if title:
        return f"{subj} {pred} {obj} | {title}"
    return f"{subj} {pred} {obj}"

def build_faiss_index(input_json=INPUT_TRIPLES_JSON,
                      index_file=FAISS_INDEX_FILE,
                      embeddings_file=EMBEDDINGS_FILE,
                      metadata_file=METADATA_FILE,
                      batch_size=BATCH_SIZE):
    """Build FAISS index from validated triples"""
    
    print("="*60)
    print("BUILDING FAISS INDEX FOR KNOWLEDGE GRAPH")
    print("="*60)
    
    triples = load_triples(input_json)
    n = len(triples)
    print(f"Total triples to index: {n}")
    if n == 0:
        print("No triples found. Exiting.")
        return None, None, None
    
    model = get_model()
    
    # Sample embedding to determine dimension
    sample_emb = model.encode([triple_to_text(triples[0])], convert_to_numpy=True)
    dim = sample_emb.shape[1]
    print(f"Embedding dimension: {dim}")
    
    embeddings = np.zeros((n, dim), dtype=np.float32)
    metadata = []
    
    print("\nEncoding triples...")
    for start in tqdm(range(0, n, batch_size), desc="Processing batches"):
        end = min(n, start + batch_size)
        batch_texts = [triple_to_text(t) for t in triples[start:end]]
        batch_emb = model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False)
        embeddings[start:end] = batch_emb
        
        for t in triples[start:end]:
            metadata.append({
                "subject": t.get("subject"),
                "predicate": t.get("predicate"),
                "object": t.get("object"),
                "title": t.get("title"),
                "chunk_id": t.get("chunk_id")
            })
    
    print("\nNormalizing embeddings for cosine similarity...")
    faiss.normalize_L2(embeddings)
    
    print("Building FAISS index...")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    
    print(f"\nIndex built successfully!")
    print(f"Total vectors indexed: {index.ntotal}")
    
    print("\nSaving index files...")
    faiss.write_index(index, index_file)
    np.save(embeddings_file, embeddings)
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"FAISS index saved to: {index_file}")
    print(f"Embeddings saved to: {embeddings_file}")
    print(f"Metadata saved to: {metadata_file}")
    print("="*60)
    
    return index, embeddings, metadata

def load_index_and_metadata(index_file=FAISS_INDEX_FILE, 
                            metadata_file=METADATA_FILE, 
                            embeddings_file=EMBEDDINGS_FILE):
    """Load FAISS index and metadata"""
    if not os.path.exists(index_file):
        raise FileNotFoundError(f"FAISS index not found: {index_file}")
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata not found: {metadata_file}")
    
    index = faiss.read_index(index_file)
    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    embeddings = None
    if os.path.exists(embeddings_file):
        embeddings = np.load(embeddings_file)
    
    return index, metadata, embeddings

def search_triples(query, k=5, 
                  index_file=FAISS_INDEX_FILE, 
                  metadata_file=METADATA_FILE):
    """Search for relevant triples using semantic similarity"""
    index, metadata, _ = load_index_and_metadata(index_file, metadata_file)
    model = get_model()
    q_emb = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    
    scores, indices = index.search(q_emb, k)
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        meta = metadata[idx]
        results.append({
            "score": float(score),
            "subject": meta.get("subject"),
            "predicate": meta.get("predicate"),
            "object": meta.get("object"),
            "title": meta.get("title"),
            "chunk_id": meta.get("chunk_id"),
            "triple_text": f"{meta.get('subject')} {meta.get('predicate')} {meta.get('object')}"
        })
    return results

def pretty_print_search(query, k=5):
    """Pretty print search results"""
    print("\n" + "="*80)
    print(f"QUERY: {query}")
    print("="*80)
    
    results = search_triples(query, k=k)
    if not results:
        print("No results found.")
        return
    
    for i, r in enumerate(results, 1):
        print(f"\n{i}. Similarity Score: {r['score']:.4f}")
        print(f"   {r['subject']} → {r['predicate']} → {r['object']}")
        print(f"   Paper: {r.get('title', 'Unknown')}")
        print(f"   Chunk: {r.get('chunk_id', 'Unknown')}")
        print("-"*80)
    print()

# -------- TOP-K FREQUENCY ANALYSIS --------
def get_top_k_entities(k=10, role="all", metadata_file=METADATA_FILE):
    """Get top-k most frequent entities"""
    if not os.path.exists(metadata_file):
        return []
    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    if role == "subject":
        items = [m["subject"] for m in metadata]
    elif role == "object":
        items = [m["object"] for m in metadata]
    else:
        items = [m["subject"] for m in metadata] + [m["object"] for m in metadata]
    
    return Counter(items).most_common(k)

def get_top_k_predicates(k=10, metadata_file=METADATA_FILE):
    """Get top-k most frequent predicates"""
    if not os.path.exists(metadata_file):
        return []
    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    preds = [m["predicate"] for m in metadata]
    return Counter(preds).most_common(k)

def demo_searches():
    """Run demo searches for hackathon presentation"""
    print("\n" + "="*80)
    print("KNOWLEDGE GRAPH SEMANTIC SEARCH DEMO")
    print("="*80)
    
    demo_queries = [
        "What are the effects of microgravity on bone density?",
        "How can we prevent bone loss in astronauts?",
        "What happens to the immune system during spaceflight?",
        "How does radiation affect DNA?",
        "What are countermeasures for muscle atrophy?"
    ]
    
    for query in demo_queries:
        pretty_print_search(query, k=5)
        print("\n📊 Top Entities & Predicates (Global Context)")
        print("Top 5 Entities (all):", get_top_k_entities(k=5, role="all"))
        print("Top 5 Subjects:", get_top_k_entities(k=5, role="subject"))
        print("Top 5 Predicates:", get_top_k_predicates(k=5))
        input("\nPress Enter for next query...")

def get_statistics():
    """Get statistics about the indexed knowledge graph"""
    if not os.path.exists(METADATA_FILE):
        print("Index not built yet.")
        return
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    subjects = [m["subject"] for m in metadata]
    predicates = [m["predicate"] for m in metadata]
    objects = [m["object"] for m in metadata]
    papers = [m.get("title", "Unknown") for m in metadata]
    
    print("\n" + "="*60)
    print("KNOWLEDGE GRAPH STATISTICS")
    print("="*60)
    print(f"Total triples indexed: {len(metadata)}")
    print(f"Unique subjects: {len(set(subjects))}")
    print(f"Unique predicates: {len(set(predicates))}")
    print(f"Unique objects: {len(set(objects))}")
    print(f"Papers covered: {len(set(papers))}")
    
    print("\nTop 10 most common relationships:")
    for pred, count in Counter(predicates).most_common(10):
        print(f"  {pred}: {count}")
    
    print("\nTop 10 most common entities (subjects):")
    for subj, count in Counter(subjects).most_common(10):
        print(f"  {subj}: {count}")
    print("="*60)

if __name__ == "__main__":
    if not os.path.exists(FAISS_INDEX_FILE):
        print("FAISS index not found. Building index...")
        build_faiss_index()
    else:
        print("FAISS index found. Loading existing index...")
    
    get_statistics()
    print("\nStarting demo searches...")
    demo_searches()
