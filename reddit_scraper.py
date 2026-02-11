import argparse
import hashlib
import json
import os
import random
import re
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import requests

# Defaults / Config
UA_JSON = "DSCI560-Lab5-Scraper/2.0 (Educational Project; JSON endpoints; +your_email@example.com)"
DEFAULT_SUBS = ["tech", "cybersecurity", "technology", "artificial", "datascience", "computerscience"]
DEFAULT_SORTS = ["new", "hot", "top", "rising"]  # 多 sort 绕开单 listing ~1000 限制
DEFAULT_GLOBAL_QUERY = "cyber OR security OR malware OR ransomware OR vulnerability"

POSTS_PER_REQUEST = 100  # reddit max
MIN_TEXT_LEN = 30

SLEEP_RANGE = (1.0, 2.0)
MAX_RETRIES = 6
BACKOFF_BASE = 2


# Utilities
def clean_text(s: str) -> str:
    """最小但够用的清洗：去HTML、零宽、合并空白、去奇怪控制符。"""
    if not s:
        return ""
    s = s.replace("\u200b", " ")
    # 去 HTML tag（JSON selftext 通常没tag，但保险）
    s = re.sub(r"<[^>]+>", " ", s)
    # 去一些控制字符
    s = re.sub(r"[\x00-\x1f\x7f]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def mask_username(author: str, salt: str = "dsci560") -> str:
    """把用户名 mask/pseudonymize。"""
    if not author:
        author = "unknown"
    h = hashlib.sha256((salt + ":" + author).encode("utf-8")).hexdigest()
    return "user_" + h[:12]


def to_iso_utc(ts_utc: Optional[float]) -> Optional[str]:
    if not ts_utc:
        return None
    return datetime.fromtimestamp(ts_utc, tz=timezone.utc).isoformat()


def is_likely_ad_or_irrelevant(post: Dict) -> bool:
    # promoted / stickied 广告与置顶
    if post.get("stickied"):
        return True
    if post.get("promoted"):
        return True
    # 有些广告/推荐会有 weird domain/empty author 等，这里不做过度猜测
    return False


def extract_image_fields(d: Dict) -> Tuple[bool, bool, str, List[str], str]:
    """
    返回:
      is_image, is_gallery, image_url, gallery_urls, thumbnail
    """
    url = d.get("url") or ""
    url_l = url.lower()

    image_ext = (".jpg", ".jpeg", ".png", ".gif", ".webp")
    is_image = url_l.endswith(image_ext) or ("i.redd.it" in url_l) or ("v.redd.it" in url_l)

    is_gallery = bool(d.get("is_gallery", False))
    gallery_urls: List[str] = []

    if is_gallery and isinstance(d.get("media_metadata"), dict):
        for _, info in d["media_metadata"].items():
            if info.get("status") == "valid" and isinstance(info.get("s"), dict):
                u = info["s"].get("u", "")
                if u:
                    gallery_urls.append(u.replace("&amp;", "&"))
        if gallery_urls:
            is_image = True

    thumbnail = d.get("thumbnail") or ""
    if thumbnail in ("self", "default", "nsfw", "spoiler"):
        thumbnail = ""

    image_url = ""
    if is_image:
        image_url = url
    else:
        # preview 里有时也能拿到图
        pv = d.get("preview", {})
        if isinstance(pv, dict) and isinstance(pv.get("images"), list) and pv["images"]:
            try:
                image_url = pv["images"][0]["source"]["url"].replace("&amp;", "&")
                if image_url:
                    is_image = True
            except Exception:
                pass

    return is_image, is_gallery, image_url, gallery_urls, thumbnail


# HTTP with backoff
def fetch_json(session: requests.Session, url: str, params: Optional[Dict] = None) -> Optional[Dict]:
    for i in range(MAX_RETRIES):
        try:
            r = session.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()

            if r.status_code == 429 or (500 <= r.status_code < 600):
                wait = min(120, (BACKOFF_BASE ** (i + 1)) + random.uniform(0, 1.5))
                print(f"[WARN] HTTP {r.status_code}. sleep {wait:.1f}s then retry... url={r.url}")
                time.sleep(wait)
                continue

            print(f"[WARN] HTTP {r.status_code} for {r.url}")
            return None
        except Exception as e:
            wait = min(60, (BACKOFF_BASE ** (i + 1)) + random.uniform(0, 1.0))
            print(f"[WARN] request failed ({e}). sleep {wait:.1f}s then retry...")
            time.sleep(wait)
    return None


# Checkpoint
def load_checkpoint(path: str) -> Tuple[List[Dict], set, Dict]:
    """
    返回 collected, seen_ids, state
    state: 用于记录每个 source 的 after / counts，便于真正断点续跑
    """
    if not path or not os.path.exists(path):
        return [], set(), {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    collected = data.get("collected", [])
    seen = set(data.get("seen", []))
    state = data.get("state", {})
    return collected, seen, state


def save_checkpoint(path: str, collected: List[Dict], seen: set, state: Dict) -> None:
    if not path:
        return
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(
            {"n": len(collected), "collected": collected, "seen": list(seen), "state": state},
            f,
            ensure_ascii=False,
        )
    os.replace(tmp, path)


# Parsing
def parse_listing(payload: Dict) -> Tuple[List[Dict], Optional[str]]:
    """
    标准 listing: /r/{sub}/{sort}.json or /search/.json
    返回 posts(list of child["data"]) 和 after token
    """
    try:
        children = payload["data"]["children"]
        after = payload["data"].get("after")
    except Exception:
        return [], None

    out = []
    for ch in children:
        if ch.get("kind") != "t3":
            continue
        d = ch.get("data", {})
        if not isinstance(d, dict):
            continue
        out.append(d)
    return out, after


# Main scraping logic
def build_sources(subs: List[str], sorts: List[str], use_global: bool) -> List[Tuple[str, str]]:
    """
    sources: (sub, sort) + optional ("_global_", "global_search_year")
    """
    sources: List[Tuple[str, str]] = []
    for sub in subs:
        for srt in sorts:
            sources.append((sub, srt))
    if use_global:
        sources.append(("_global_", "global_search_year"))
    return sources


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("N", type=int, help="total posts to fetch (e.g., 5000)")
    ap.add_argument("--subs", nargs="*", default=DEFAULT_SUBS, help="subreddits (domain you chose)")
    ap.add_argument("--sorts", nargs="*", default=DEFAULT_SORTS, help="listing sorts to rotate")
    ap.add_argument("--use_global_search", action="store_true", help="use global search to fill remaining posts")
    ap.add_argument("--global_query", default=DEFAULT_GLOBAL_QUERY, help="query string for global search")
    ap.add_argument("--time_range", default="year", choices=["day", "week", "month", "year", "all"], help="t= in reddit")
    ap.add_argument("--per_source_cap", type=int, default=1000, help="cap per (sub,sort) source")
    ap.add_argument("--min_text_len", type=int, default=MIN_TEXT_LEN)
    ap.add_argument("--sleep_min", type=float, default=SLEEP_RANGE[0])
    ap.add_argument("--sleep_max", type=float, default=SLEEP_RANGE[1])
    ap.add_argument("--checkpoint", default="ck_scrape.json", help="checkpoint path")
    ap.add_argument("--out_prefix", default="posts_lab5", help="output prefix for json/jsonl")
    ap.add_argument("--salt", default="dsci560", help="salt for username masking")
    args = ap.parse_args()

    target_n = args.N
    per_source_cap = min(args.per_source_cap, 1000)  # 和 PDF 的 API 阈值一致思路
    sleep_range = (max(0.0, args.sleep_min), max(args.sleep_min, args.sleep_max))

    collected, seen, state = load_checkpoint(args.checkpoint)
    print(f"[INFO] resume: already have {len(collected)} posts; checkpoint={args.checkpoint}")

    # state schema:
    # state["after"][key_source] = token
    # state["count"][key_source] = int
    if "after" not in state:
        state["after"] = {}
    if "count" not in state:
        state["count"] = {}

    subs_set = set(args.subs)

    with requests.Session() as s:
        s.headers.update({
            "User-Agent": UA_JSON,
            "Accept-Language": "en-US,en;q=0.9",
        })

        sources = build_sources(args.subs, args.sorts, args.use_global_search)
        # init counts
        for sub, mode in sources:
            key = f"{sub}:{mode}"
            state["after"].setdefault(key, None)
            state["count"].setdefault(key, 0)

        while len(collected) < target_n:
            progressed = False

            for sub, mode in sources:
                if len(collected) >= target_n:
                    break

                key_source = f"{sub}:{mode}"
                if state["count"][key_source] >= per_source_cap:
                    continue

                after = state["after"][key_source]

                # build URL/params
                if mode == "global_search_year":
                    url = "https://www.reddit.com/search/.json"
                    params = {
                        "q": args.global_query,
                        "sort": "new",
                        "t": args.time_range,
                        "limit": POSTS_PER_REQUEST,
                    }
                    if after:
                        params["after"] = after
                else:
                    # subreddit listing
                    url = f"https://www.reddit.com/r/{sub}/{mode}.json"
                    params = {"limit": POSTS_PER_REQUEST}
                    if mode == "top":
                        params["t"] = args.time_range
                    if after:
                        params["after"] = after

                payload = fetch_json(s, url, params=params)
                if not payload:
                    print(f"[WARN] listing fetch failed for {key_source}; cool down 10s...")
                    time.sleep(10)
                    continue

                batch, after2 = parse_listing(payload)
                state["after"][key_source] = after2

                if not batch:
                    continue

                # consume batch
                for d in batch:
                    if len(collected) >= target_n:
                        break
                    if state["count"][key_source] >= per_source_cap:
                        break

                    # global search 时，确保 topic/domain 仍然合理：只保留你选的 subs（否则会混进奇怪的 94 subs）
                    real_sub = d.get("subreddit") or sub
                    if sub == "_global_":
                        if real_sub not in subs_set:
                            continue

                    # filter ads / stickied
                    if is_likely_ad_or_irrelevant(d):
                        continue

                    fullname = d.get("name")  # t3_xxx
                    if not fullname or fullname in seen:
                        continue
                    seen.add(fullname)

                    title = clean_text(d.get("title", ""))
                    selftext = clean_text(d.get("selftext", ""))
                    is_self = bool(d.get("is_self", False))

                    # message abstraction：正文不足就回退到标题（保证有文本可做聚类/关键词）
                    body = selftext if is_self else ""
                    final_text = body if len(body) >= args.min_text_len else title
                    if len(final_text) < args.min_text_len:
                        continue

                    is_image, is_gallery, image_url, gallery_urls, thumbnail = extract_image_fields(d)

                    author_raw = d.get("author") or "[deleted]"
                    author_masked = mask_username(author_raw, salt=args.salt)

                    created_utc = d.get("created_utc", 0)
                    created_iso = to_iso_utc(created_utc)

                    permalink = d.get("permalink") or ""
                    if permalink and not permalink.startswith("http"):
                        permalink = "https://www.reddit.com" + permalink

                    out_url = d.get("url") or ""

                    # 输出记录（方便后续入库 MySQL）
                    rec = {
                        "fullname": fullname,
                        "post_id": d.get("id", ""),
                        "subreddit": real_sub,
                        "title": title,
                        "author": author_masked,          # ✅ masked
                        "author_raw": None,               # 不存原始用户名（隐私）
                        "created_utc": created_utc,
                        "created": created_iso,           # ✅ timestamp converted
                        "permalink": permalink,
                        "out_url": out_url,
                        "domain": d.get("domain", ""),
                        "score": d.get("score", 0),
                        "num_comments": d.get("num_comments", 0),
                        "is_self": is_self,
                        "body": body,
                        "final_text": final_text,
                        "is_image": bool(is_image),
                        "is_gallery": bool(is_gallery),
                        "image_url": image_url,
                        "gallery_urls": gallery_urls,
                        "thumbnail": thumbnail,
                        "over_18": bool(d.get("over_18", False)),
                        "link_flair_text": d.get("link_flair_text", "") or "",
                        # 预留：OCR/keywords/topics
                        "ocr_text": "",
                        "keywords": [],
                        "topic": "",
                    }

                    collected.append(rec)
                    state["count"][key_source] += 1
                    progressed = True

                    if len(collected) % 50 == 0:
                        save_checkpoint(args.checkpoint, collected, seen, state)
                        print(f"[INFO] checkpoint saved: n={len(collected)}")

                    time.sleep(random.uniform(*sleep_range))

                print(f"[INFO] {key_source}: +{state['count'][key_source]} / cap={per_source_cap} | total={len(collected)}/{target_n}")
                time.sleep(random.uniform(*sleep_range))

            if not progressed:
                print("[WARN] no progress in this round; sleep 10s then retry...")
                save_checkpoint(args.checkpoint, collected, seen, state)
                time.sleep(10)

    # write outputs
    out_jsonl = f"{args.out_prefix}_{len(collected)}.jsonl"
    out_json = f"{args.out_prefix}_{len(collected)}.json"

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for rec in collected:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(collected, f, ensure_ascii=False, indent=2)

    save_checkpoint(args.checkpoint, collected, seen, state)

    print(f"\n[DONE] wrote {len(collected)} posts")
    print(f" - {out_jsonl}")
    print(f" - {out_json}")
    print("\n[SAMPLE] first 3 posts:")
    for p in collected[:3]:
        print("-", p["subreddit"], "|", p["title"][:60], "| is_image=", p["is_image"], "| created=", p.get("created"))


if __name__ == "__main__":
    main()
