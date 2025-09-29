import json
import re
import os
import time
from sentence_transformers import SentenceTransformer
import faiss
from collections import Counter

# ---------------- CONFIG ----------------
INPUT_TRIPLES = "kg_triples.json"
INPUT_CHUNKS = "papers_metadata.json"
OUTPUT_VALIDATED = "kg_triples_validated.json"
OUTPUT_DROPPED = "kg_triples_dropped.json"
CHECKPOINT_FILE = "kg_faiss_checkpoint.json"

FAISS_THRESHOLD = 0.65
CHECKPOINT_EVERY = 1000  # save progress every N triples
# ----------------------------------------

print("Loading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded successfully!")

def is_valid_entity(entity: str) -> bool:
    """Check if entity is valid"""
    entity = entity.lower().strip()
    if not entity or entity in ["it", "this", "that", "they", "we", "study", "research", "author", "paper", "article"]:
        return False
    if entity.isdigit():
        return False
    if re.match(r'^[\d\W]+$', entity):  # only numbers/special chars
        return False
    if len(entity) < 2 or len(entity) > 100:
        return False
    return True

def is_valid_predicate(pred: str) -> bool:
    """Check if predicate is valid"""
    pred = pred.lower().strip()
    bad_preds = ["is", "are", "has", "have", "contains", "includes"]
    return pred not in bad_preds and 2 <= len(pred) <= 50

def verify_with_faiss(triple: dict, text: str, threshold: float = 0.65) -> bool:
    """Verify triple using FAISS semantic similarity"""
    sentences = [s.strip() for s in text.split(". ") if len(s.strip()) > 10]
    if not sentences:
        return False
    
    try:
        embeddings = embedding_model.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        
        triple_text = f"{triple['subject']} {triple['predicate']} {triple['object']}"
        triple_embedding = embedding_model.encode([triple_text], convert_to_numpy=True, normalize_embeddings=True)
        
        scores, _ = index.search(triple_embedding, k=1)
        return float(scores[0][0]) >= threshold
    except Exception as e:
        print(f"FAISS error: {e}")
        return False

def load_checkpoint():
    """Load checkpoint if exists"""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Convert processed_ids back to set
            data["processed_ids"] = set(data.get("processed_ids", []))
            return data
    return {"validated": [], "dropped": [], "processed_ids": set()}

def save_checkpoint(validated, dropped, processed_ids):
    """Save progress checkpoint"""
    checkpoint = {
        "validated": validated,
        "dropped": dropped,
        "processed_ids": list(processed_ids)  # Convert set to list for JSON
    }
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2, ensure_ascii=False)
    print(f"Checkpoint saved: {len(processed_ids)} triples processed")

if __name__ == "__main__":
    # Load triples
    print(f"Loading triples from {INPUT_TRIPLES}...")
    with open(INPUT_TRIPLES, "r", encoding="utf-8") as f:
        triples = json.load(f)
    print(f"Loaded {len(triples)} triples")

    # Load chunks
    print(f"Loading text chunks from {INPUT_CHUNKS}...")
    with open(INPUT_CHUNKS, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    text_lookup = {c["chunk_id"]: c["text"] for c in chunks}
    print(f"Loaded {len(text_lookup)} text chunks")

    # Load checkpoint
    checkpoint = load_checkpoint()
    validated_triples = checkpoint["validated"]
    dropped_triples = checkpoint["dropped"]
    processed_ids = checkpoint["processed_ids"]
    
    if processed_ids:
        print(f"Resuming from checkpoint: {len(processed_ids)} already processed")

    start_time = time.time()
    
    print("\nValidating triples (rule-based + FAISS)...")

    for i, t in enumerate(triples):
        # Create unique ID for this triple
        triple_id = f"{t.get('chunk_id','')}-{t.get('subject','')}-{t.get('predicate','')}-{t.get('object','')}"
        
        # Skip if already processed
        if triple_id in processed_ids:
            continue

        # Rule-based validation
        if not all(k in t for k in ["subject", "predicate", "object", "chunk_id"]):
            dropped_triples.append(t)
        elif not is_valid_entity(t["subject"]):
            dropped_triples.append(t)
        elif not is_valid_entity(t["object"]):
            dropped_triples.append(t)
        elif not is_valid_predicate(t["predicate"]):
            dropped_triples.append(t)
        else:
            # FAISS semantic validation
            text = text_lookup.get(t["chunk_id"], "")
            if text and verify_with_faiss(t, text, FAISS_THRESHOLD):
                validated_triples.append({**t, "faiss_verified": True})
            else:
                dropped_triples.append(t)

        processed_ids.add(triple_id)

        # Periodic checkpoint
        if (i + 1) % CHECKPOINT_EVERY == 0:
            save_checkpoint(validated_triples, dropped_triples, processed_ids)
            elapsed = time.time() - start_time
            avg_time = elapsed / len(processed_ids)
            remaining = (len(triples) - len(processed_ids)) * avg_time
            print(f"Progress: {len(processed_ids)}/{len(triples)} | ETA: {remaining/60:.1f} min")

    # Final save
    print("\nSaving final results...")
    with open(OUTPUT_VALIDATED, "w", encoding="utf-8") as f:
        json.dump(validated_triples, f, indent=2, ensure_ascii=False)
    
    with open(OUTPUT_DROPPED, "w", encoding="utf-8") as f:
        json.dump(dropped_triples, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start_time
    
    print("\n" + "="*60)
    print("FAISS VALIDATION COMPLETE")
    print("="*60)
    print(f"Original triples:    {len(triples)}")
    print(f"Validated triples:   {len(validated_triples)} ({len(validated_triples)/len(triples):.1%})")
    print(f"Dropped triples:     {len(dropped_triples)} ({len(dropped_triples)/len(triples):.1%})")
    print(f"Time taken:          {elapsed/60:.1f} minutes")
    print(f"Avg time per triple: {elapsed/len(triples):.3f}s")
    
    print(f"\nValidated triples saved to: {OUTPUT_VALIDATED}")
    print(f"Dropped triples saved to:   {OUTPUT_DROPPED}")

    # Remove checkpoint after success
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print("\nCheckpoint file removed")
    
    # Show sample validated triples
    if validated_triples:
        print("\nSample validated triples:")
        for i, triple in enumerate(validated_triples[:5], 1):
            print(f"{i}. {triple['subject']} --{triple['predicate']}--> {triple['object']}")