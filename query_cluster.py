import argparse
import json
import os
from datetime import datetime

import numpy as np
from sentence_transformers import SentenceTransformer


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_filename(s: str, max_len: int = 60) -> str:
    s = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s.strip())
    s = s.strip("_")
    return (s[:max_len] if len(s) > max_len else s) or "query"


def plot_highlight_cluster(pca_2d: np.ndarray, labels: np.ndarray, best: int, out_path: str, title: str):
    import matplotlib.pyplot as plt

    mask = labels == best
    other = ~mask

    plt.figure(figsize=(8, 6))
    # background points
    plt.scatter(pca_2d[other, 0], pca_2d[other, 1], s=8, alpha=0.25)
    # highlight best cluster
    plt.scatter(pca_2d[mask, 0], pca_2d[mask, 1], s=10, alpha=0.9)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", help='keywords or a message, e.g. "ransomware attack"')
    ap.add_argument("--out_dir", default="outputs", help="where centroids/meta/summary are stored")
    ap.add_argument("--top_posts", type=int, default=5, help="how many representative posts to show")
    ap.add_argument("--plot", action="store_true", help="Generate a highlight plot for the best cluster")
    args = ap.parse_args()

    query = (args.query or "").strip()

    meta_path = os.path.join(args.out_dir, "meta.json")
    centroids_path = os.path.join(args.out_dir, "centroids.npy")
    summary_path = os.path.join(args.out_dir, "clusters_summary.json")

    if not (os.path.exists(meta_path) and os.path.exists(centroids_path) and os.path.exists(summary_path)):
        raise FileNotFoundError(
            "Missing one of required files: meta.json, centroids.npy, clusters_summary.json. "
            "Run embed_and_cluster.py first."
        )

    meta = load_json(meta_path)
    summary = load_json(summary_path)
    centroids = np.load(centroids_path).astype(np.float32)

    model_name = meta.get("model", "all-MiniLM-L6-v2")
    model = SentenceTransformer(model_name)

    q_vec = model.encode([query], normalize_embeddings=True)
    q_vec = np.asarray(q_vec, dtype=np.float32)[0]

    c_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)
    scores = c_norm @ q_vec
    best = int(np.argmax(scores))
    best_score = float(scores[best])

    best_item = None
    for item in summary:
        if int(item.get("cluster_id", -1)) == best:
            best_item = item
            break

    print("\n=== QUERY RESULT ===")
    print(f"Query: {query}")
    print(f"Model: {model_name}")

    text_fields = meta.get("text_fields_used")
    if text_fields:
        print(f"Text fields used: {text_fields}")

    print(f"Best cluster: {best}  (cosine_sim={best_score:.4f})")

    if not best_item:
        print("[WARN] Could not find cluster in clusters_summary.json")
        return

    topic = best_item.get("topic")
    if topic:
        print(f"Topic: {topic}")

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

    # NEW: query-time graphical representation
    if args.plot:
        pca_path = os.path.join(args.out_dir, "pca_2d.npy")
        labels_path = os.path.join(args.out_dir, "labels.npy")

        if not (os.path.exists(pca_path) and os.path.exists(labels_path)):
            raise FileNotFoundError(
                "Missing pca_2d.npy / labels.npy. "
                "Re-run embed_and_cluster.py (updated version) once to generate them."
            )

        pca_2d = np.load(pca_path).astype(np.float32)
        labels = np.load(labels_path).astype(np.int32)

        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        qname = safe_filename(query)
        out_plot = os.path.join(args.out_dir, f"query_highlight_cluster{best}_{qname}_{stamp}.png")

        title = f"Query='{query}' → best cluster={best} (highlighted)"
        plot_highlight_cluster(pca_2d, labels, best, out_plot, title)

        print("\n[GRAPH] Highlight plot saved:")
        print(f"  {out_plot}")
        print("Open it to satisfy the 'graphical representation' requirement.")


if __name__ == "__main__":
    main()
