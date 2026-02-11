# clustering.py
# Uses: TF-IDF + KMeans + PCA
# Input: scraped_posts.json (from reddit_scraper.py)
# Output: clustered_posts.json + cluster_plot.png

import json
import re
from collections import Counter
from typing import List, Dict, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


DEFAULT_INPUT_JSON = "scraped_posts.json"
DEFAULT_OUTPUT_JSON = "clustered_posts.json"
DEFAULT_PLOT_PNG = "cluster_plot.png"

STOP_WORDS = "english"


def _safe_text(x) -> str:
    if x is None:
        return ""
    return str(x)


def basic_clean(text: str) -> str:
    text = _safe_text(text)
    text = re.sub(r"http[s]?://\S+", " ", text)
    text = re.sub(r"www\.\S+", " ", text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def load_posts(path: str = DEFAULT_INPUT_JSON) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        posts = json.load(f)
    if not isinstance(posts, list):
        raise ValueError("Input JSON must be a list of posts.")
    return posts


def build_corpus(posts: List[Dict], text_field: str = "combined_text") -> List[str]:
    corpus = []
    for p in posts:
        txt = p.get(text_field, "")
        if not txt:
            # fallback if combined_text missing
            txt = f"{_safe_text(p.get('title',''))} {_safe_text(p.get('selftext',''))}"
        corpus.append(basic_clean(txt))
    return corpus


def fit_vectorizer(corpus: List[str], max_features: int = 5000) -> Tuple[TfidfVectorizer, np.ndarray]:
    vectorizer = TfidfVectorizer(
        stop_words=STOP_WORDS,
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
    )
    X = vectorizer.fit_transform(corpus)
    return vectorizer, X


def fit_kmeans(X, k: int = 8, random_state: int = 42) -> KMeans:
    model = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    model.fit(X)
    return model


def top_keywords_for_cluster(vectorizer: TfidfVectorizer, center_vec, top_n: int = 10) -> List[str]:

    feature_names = vectorizer.get_feature_names_out()
    top_idx = np.argsort(center_vec)[::-1][:top_n]
    return [feature_names[i] for i in top_idx if center_vec[i] > 0]


def representative_post_index(X, center_vec) -> int:

    # X: dense (n, d)
    diffs = X - center_vec
    dists = np.linalg.norm(diffs, axis=1)
    return int(np.argmin(dists))


def visualize_pca(X, labels, out_png: str = DEFAULT_PLOT_PNG) -> None:
    pca = PCA(n_components=2, random_state=42)
    reduced = pca.fit_transform(X)

    plt.figure(figsize=(9, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, s=12)
    plt.title("Reddit Posts Clusters (PCA 2D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def run_clustering_pipeline(
    input_json: str = DEFAULT_INPUT_JSON,
    output_json: str = DEFAULT_OUTPUT_JSON,
    out_png: str = DEFAULT_PLOT_PNG,
    k: int = 8,
    max_features: int = 5000,
    top_keywords: int = 10,
) -> Dict:
    posts = load_posts(input_json)
    corpus = build_corpus(posts)

    vectorizer, X_sparse = fit_vectorizer(corpus, max_features=max_features)
    model = fit_kmeans(X_sparse, k=k)

    labels = model.labels_.astype(int)

    # dense for centroid distance + PCA plot
    X_dense = X_sparse.toarray()
    centers = model.cluster_centers_

    cluster_info = {}
    for cid in range(k):
        idxs = np.where(labels == cid)[0]
        if len(idxs) == 0:
            cluster_info[cid] = {
                "size": 0,
                "keywords": [],
                "representative_post_id": None,
                "representative_title": None,
            }
            continue

        kw = top_keywords_for_cluster(vectorizer, centers[cid], top_n=top_keywords)

        rep_local = representative_post_index(X_dense[idxs], centers[cid])
        rep_idx = int(idxs[rep_local])

        rep_post = posts[rep_idx]
        cluster_info[cid] = {
            "size": int(len(idxs)),
            "keywords": kw,
            "representative_post_id": rep_post.get("post_id") or rep_post.get("id"),
            "representative_title": rep_post.get("title", "")[:120],
        }

    for i, p in enumerate(posts):
        cid = int(labels[i])
        p["cluster_id"] = cid
        p["cluster_keywords"] = cluster_info[cid]["keywords"]

    visualize_pca(X_dense, labels, out_png=out_png)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": {
                    "k": k,
                    "max_features": max_features,
                    "top_keywords": top_keywords,
                    "input_json": input_json,
                    "output_plot": out_png,
                },
                "cluster_info": cluster_info,
                "posts": posts,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    summary = {
        "total_posts": len(posts),
        "k": k,
        "output_json": output_json,
        "plot_png": out_png,
        "cluster_sizes": {str(cid): cluster_info[cid]["size"] for cid in range(k)},
    }
    return summary


if __name__ == "__main__":
    s = run_clustering_pipeline(k=8)
    print("Clustering done.")
    print(json.dumps(s, ensure_ascii=False, indent=2))
