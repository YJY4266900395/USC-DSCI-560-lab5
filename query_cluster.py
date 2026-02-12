import argparse
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", help="keywords or a message, e.g. \"ransomware attack\"")
    ap.add_argument("--out_dir", default="outputs", help="where centroids/meta/summary are stored")
    ap.add_argument("--top_posts", type=int, default=5, help="how many representative posts to show")
    args = ap.parse_args()

    meta_path = os.path.join(args.out_dir, "meta.json")
    centroids_path = os.path.join(args.out_dir, "centroids.npy")
    summary_path = os.path.join(args.out_dir, "clusters_summary.json")

    if not (os.path.exists(meta_path) and os.path.exists(centroids_path) and os.path.exists(summary_path)):
        raise FileNotFoundError(
            "Missing one of required files: meta.json, centroids.npy, clusters_summary.json. "
            "Run embed_and_cluster.py first with the NEW centroid/meta saving."
        )

    meta = load_json(meta_path)
    summary = load_json(summary_path)
    centroids = np.load(centroids_path).astype(np.float32)

    model_name = meta.get("model", "all-MiniLM-L6-v2")
    model = SentenceTransformer(model_name)

    q_vec = model.encode([args.query], normalize_embeddings=True)
    q_vec = np.asarray(q_vec, dtype=np.float32)[0]

    # Normalize centroids (KMeans centroids are not necessarily unit vectors)
    c_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)

    # cosine similarity (since q_vec is normalized)
    scores = c_norm @ q_vec
    best = int(np.argmax(scores))
    best_score = float(scores[best])

    # Find best cluster summary
    best_item = None
    for item in summary:
        if int(item.get("cluster_id", -1)) == best:
            best_item = item
            break

    print("\n=== QUERY RESULT ===")
    print(f"Query: {args.query}")
    print(f"Model: {model_name}")
    print(f"Best cluster: {best}  (cosine_sim={best_score:.4f})")

    if not best_item:
        print("[WARN] Could not find cluster in clusters_summary.json")
        return

    kws = best_item.get("top_keywords", [])
    print("\nTop keywords:")
    print(" - " + ", ".join(kws) if kws else " (none)")

    reps = best_item.get("representative_posts", [])[: args.top_posts]
    print("\nRepresentative posts:")
    for i, p in enumerate(reps, 1):
        title = p.get("title", "")
        sub = p.get("subreddit", "")
        link = p.get("permalink", "")
        sim = p.get("cosine_similarity_to_centroid", None)
        sim_str = f"{sim:.4f}" if isinstance(sim, (int, float)) else "NA"
        print(f"\n[{i}] r/{sub} | sim_to_centroid={sim_str}")
        print(f"Title: {title}")
        if link:
            print(f"Link:  {link}")
        if p.get("is_image") and p.get("image_url"):
            print(f"Image: {p.get('image_url')}")


if __name__ == "__main__":
    main()