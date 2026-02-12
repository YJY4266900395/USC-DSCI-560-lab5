import argparse
import json
import os
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

# sentence-transformers is recommended for semantic embeddings
from sentence_transformers import SentenceTransformer


def load_records(path: str):
    if path.endswith(".jsonl"):
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    elif path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        raise ValueError("Input must be .json or .jsonl")


def safe_get(rec, key, default=""):
    v = rec.get(key, default)
    return v if v is not None else default


def compute_embeddings(texts, model_name="all-MiniLM-L6-v2", batch_size=64):
    model = SentenceTransformer(model_name)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # makes cosine similarity = dot product
    )
    return np.asarray(emb, dtype=np.float32)


def top_keywords_per_cluster(texts, labels, top_n=10, max_features=20000):
    """
    Use TF-IDF on final_text and aggregate within each cluster.
    """
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=3,
    )
    X = vectorizer.fit_transform(texts)
    terms = np.array(vectorizer.get_feature_names_out())

    cluster_keywords = {}
    for c in sorted(set(labels)):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            cluster_keywords[c] = []
            continue
        # mean tf-idf score across docs in cluster
        mean_scores = X[idx].mean(axis=0).A1
        top_idx = mean_scores.argsort()[::-1][:top_n]
        cluster_keywords[c] = terms[top_idx].tolist()

    return cluster_keywords


def representative_posts(embeddings, labels, centroids, records, per_cluster=3):
    """
    Find posts closest to centroid (cosine distance since embeddings normalized).
    For normalized embeddings, cosine similarity = dot(emb, centroid_normed).
    We'll compute cosine distance = 1 - similarity.
    """
    reps = {}
    # normalize centroids too
    c_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)

    for c in sorted(set(labels)):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            reps[c] = []
            continue
        E = embeddings[idx]
        sim = E @ c_norm[c]  # cosine similarity
        order = np.argsort(-sim)[:per_cluster]
        chosen = idx[order]

        out = []
        for i in chosen:
            r = records[i]
            out.append(
                {
                    "subreddit": safe_get(r, "subreddit"),
                    "title": safe_get(r, "title"),
                    "permalink": safe_get(r, "permalink"),
                    "is_image": bool(r.get("is_image", False)),
                    "image_url": safe_get(r, "image_url"),
                    "score": r.get("score", 0),
                    "num_comments": r.get("num_comments", 0),
                    "created": safe_get(r, "created"),
                    "final_text_preview": safe_get(r, "final_text")[:220],
                    "cosine_similarity_to_centroid": float(sim[np.where(chosen == i)[0][0]]),
                }
            )
        reps[c] = out
    return reps


def maybe_plot(embeddings, labels, outpath="clusters_plot.png", max_points=5000):
    """
    Simple PCA 2D plot.
    """
    import matplotlib.pyplot as plt

    n = embeddings.shape[0]
    if n > max_points:
        # subsample for readability
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=max_points, replace=False)
        E = embeddings[idx]
        y = labels[idx]
    else:
        E = embeddings
        y = labels

    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(E)

    plt.figure(figsize=(8, 6))
    plt.scatter(X2[:, 0], X2[:, 1], s=8, c=y, alpha=0.7)
    plt.title("Clusters (PCA 2D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to posts_lab5_5000.json or .jsonl")
    ap.add_argument("--k", type=int, default=8, help="Number of clusters (KMeans)")
    ap.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--top_keywords", type=int, default=10)
    ap.add_argument("--rep_posts", type=int, default=3)
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--plot", action="store_true", help="Generate PCA plot (clusters_plot.png)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    records = load_records(args.input)

    # Use final_text as primary text for semantic embedding
    texts = [safe_get(r, "final_text") for r in records]
    # Guard: drop empty
    keep = [i for i, t in enumerate(texts) if isinstance(t, str) and t.strip()]
    records = [records[i] for i in keep]
    texts = [texts[i] for i in keep]

    print(f"[INFO] Loaded {len(records)} records with non-empty final_text")

    # Embeddings
    embeddings = compute_embeddings(texts, model_name=args.model, batch_size=args.batch_size)
    print(f"[INFO] Embeddings shape: {embeddings.shape}")

    # KMeans clustering
    kmeans = KMeans(n_clusters=args.k, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(embeddings)
    centroids = kmeans.cluster_centers_
    print("[INFO] Cluster sizes:", dict(Counter(labels)))

    # --- NEW: save centroids + meta for query mode ---
    centroids_path = os.path.join(args.out_dir, "centroids.npy")
    np.save(centroids_path, centroids.astype(np.float32))

    meta = {
        "model": args.model,
        "k": int(args.k),
        "normalized_embeddings": True,  # because normalize_embeddings=True
        "created_at": datetime.utcnow().isoformat() + "Z",
        "input_file": os.path.basename(args.input),
    }
    meta_path = os.path.join(args.out_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Wrote {centroids_path}")
    print(f"[DONE] Wrote {meta_path}")
    # --- END NEW ---

    # Keywords per cluster (TF-IDF on final_text)
    keywords = top_keywords_per_cluster(texts, labels, top_n=args.top_keywords)

    # Representative posts per cluster
    reps = representative_posts(
        embeddings=embeddings,
        labels=labels,
        centroids=centroids,
        records=records,
        per_cluster=args.rep_posts,
    )

    # Write per-post CSV for easy inspection / DB insertion
    df = pd.DataFrame(records)
    df["cluster_id"] = labels
    csv_path = os.path.join(args.out_dir, "clusters_posts.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"[DONE] Wrote {csv_path}")

    # Write summary JSON
    summary = []
    for c in sorted(set(labels)):
        summary.append(
            {
                "cluster_id": int(c),
                "size": int((labels == c).sum()),
                "top_keywords": keywords.get(c, []),
                "representative_posts": reps.get(c, []),
            }
        )

    summary_path = os.path.join(args.out_dir, "clusters_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[DONE] Wrote {summary_path}")

    # Optional plot
    if args.plot:
        plot_path = os.path.join(args.out_dir, "clusters_plot.png")
        maybe_plot(embeddings, labels, outpath=plot_path)
        print(f"[DONE] Wrote {plot_path}")


if __name__ == "__main__":
    main()