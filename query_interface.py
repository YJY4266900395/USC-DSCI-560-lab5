# query_interface.py
# Interactive CLI: type keywords/message, find nearest cluster, show results.
# Depends on clustered_posts.json produced by clustering.py

import json
import sys
from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


DEFAULT_CLUSTERED_JSON = "clustered_posts.json"


def load_clustered(path: str = DEFAULT_CLUSTERED_JSON) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def rebuild_models_from_clustered(clustered: Dict):
    posts = clustered["posts"]
    k = int(clustered["meta"]["k"])
    max_features = int(clustered["meta"]["max_features"])

    corpus = []
    for p in posts:
        txt = p.get("combined_text") or f"{p.get('title','')} {p.get('selftext','')}"
        corpus.append(str(txt))

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
    )
    X = vectorizer.fit_transform(corpus)

    model = KMeans(n_clusters=k, random_state=42, n_init="auto")
    model.fit(X)

    return posts, vectorizer, model, X


def find_best_cluster(user_text: str, vectorizer: TfidfVectorizer, model: KMeans) -> int:
    vec = vectorizer.transform([user_text])
    cid = int(model.predict(vec)[0])
    return cid


def show_cluster(posts: List[Dict], clustered_info: Dict, cid: int, top_n: int = 5):
    info = clustered_info.get(str(cid)) or clustered_info.get(cid)  # handle int/str keys
    print("\n" + "=" * 70)
    print(f"[MATCHED CLUSTER] #{cid}")
    if info:
        print(f"Size: {info.get('size')}")
        print(f"Keywords: {', '.join(info.get('keywords', []))}")
        print(f"Representative: {info.get('representative_title')}")
    print("=" * 70)

    # show top posts in this cluster (simple: high score first)
    candidates = [p for p in posts if int(p.get("cluster_id", -1)) == cid]
    candidates.sort(key=lambda x: int(x.get("score", 0)), reverse=True)

    print(f"\nTop {top_n} posts in cluster #{cid} (sorted by score):\n")
    for i, p in enumerate(candidates[:top_n], start=1):
        title = (p.get("title") or "")[:120]
        subreddit = p.get("subreddit", "")
        score = p.get("score", 0)
        url = p.get("permalink") or p.get("url") or ""
        print(f"{i}. r/{subreddit} | score={score}")
        print(f"   {title}")
        if url:
            print(f"   {url}")
        print("")


def interactive_loop(clustered_path: str = DEFAULT_CLUSTERED_JSON):
    clustered = load_clustered(clustered_path)
    posts, vectorizer, model, _ = rebuild_models_from_clustered(clustered)

    cluster_info = clustered.get("cluster_info", {})

    print("\nInteractive Query Mode")
    print("Type a keyword/message to find the closest cluster.")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            user_text = input("> ").strip()
        except EOFError:
            break

        if not user_text:
            continue
        if user_text.lower() in ("exit", "quit", "q"):
            break

        cid = find_best_cluster(user_text, vectorizer, model)
        show_cluster(posts, cluster_info, cid, top_n=5)


if __name__ == "__main__":
    path = DEFAULT_CLUSTERED_JSON
    if len(sys.argv) > 1:
        path = sys.argv[1]
    interactive_loop(path)
