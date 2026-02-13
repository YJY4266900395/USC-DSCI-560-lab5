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
from io import BytesIO  # used to wrap image bytes for OCR processing

# Defaults / Config
UA_JSON = "DSCI560-Lab5-Scraper/2.0 (Educational Project; JSON endpoints; +your_email@example.com)"
DEFAULT_SUBS = ["tech", "cybersecurity", "technology", "artificial", "datascience", "computerscience"]
DEFAULT_SORTS = ["new", "hot", "top", "rising"]  # rotating sorts helps bypass ~1000 listing cap
DEFAULT_GLOBAL_QUERY = "cyber OR security OR malware OR ransomware OR vulnerability"

POSTS_PER_REQUEST = 100  # Reddit listing API maximum
MIN_TEXT_LEN = 30

SLEEP_RANGE = (1.0, 2.0)
MAX_RETRIES = 6
BACKOFF_BASE = 2


# Utilities
def clean_text(s: str) -> str:
    """
    Minimal but sufficient text cleaning:
    - remove HTML tags
    - remove zero-width characters
    - collapse whitespace
    - strip control characters
    """
    if not s:
        return ""
    s = s.replace("\u200b", " ")
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"[\x00-\x1f\x7f]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def mask_username(author: str, salt: str = "dsci560") -> str:
    """
    Pseudonymize usernames using SHA256 hashing.
    """
    if not author:
        author = "unknown"
    h = hashlib.sha256((salt + ":" + author).encode("utf-8")).hexdigest()
    return "user_" + h[:12]


def to_iso_utc(ts_utc: Optional[float]) -> Optional[str]:
    """
    Convert Unix timestamp (UTC) to ISO8601 string.
    """
    if not ts_utc:
        return None
    return datetime.fromtimestamp(ts_utc, tz=timezone.utc).isoformat()


def is_likely_ad_or_irrelevant(post: Dict) -> bool:
    """
    Filter out promoted or stickied posts.
    """
    if post.get("stickied"):
        return True
    if post.get("promoted"):
        return True
    return False


def extract_image_fields(d: Dict) -> Tuple[bool, bool, str, List[str], str]:
    """
    Extract image-related metadata from a Reddit post.

    Returns:
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
        # sometimes preview contains usable image
        pv = d.get("preview", {})
        if isinstance(pv, dict) and isinstance(pv.get("images"), list) and pv["images"]:
            try:
                image_url = pv["images"][0]["source"]["url"].replace("&amp;", "&")
                if image_url:
                    is_image = True
            except Exception:
                pass

    return is_image, is_gallery, image_url, gallery_urls, thumbnail


# OCR (Fast two-stage scan)
def _maybe_import_ocr():
    """
    OCR is optional. We import dependencies only if --ocr is enabled,
    so the script still runs without Tesseract installed.
    """
    try:
        import pytesseract  # type: ignore
        from PIL import Image, ImageFilter, ImageOps  # type: ignore
        return pytesseract, Image, ImageFilter, ImageOps
    except Exception as e:
        raise RuntimeError(
            "OCR is enabled but dependencies are missing. Install: "
            "`sudo apt-get install -y tesseract-ocr` and `pip install pytesseract pillow`."
        ) from e


def _basic_ocr_preprocess(img, *, image_filter_mod, image_ops_mod):
    """
    Cheap preprocessing to improve OCR accuracy without blowing up runtime:
      - grayscale
      - autocontrast
      - 2x upscale (helps small text a lot)
      - light median denoise
      - simple thresholding
    """
    gray = img.convert("L")
    gray = image_ops_mod.autocontrast(gray)
    gray = gray.resize((gray.width * 2, gray.height * 2))
    gray = gray.filter(image_filter_mod.MedianFilter(size=3))
    # A simple global threshold. Works surprisingly well for screenshots/memes.
    gray = gray.point(lambda x: 255 if x > 180 else 0)
    return gray


def ocr_maybe_text(
    session: requests.Session,
    url: str,
    *,
    pytesseract_mod,
    pil_image_mod,
    image_filter_mod,
    image_ops_mod,
    timeout: int = 15,
    max_bytes: int = 5_000_000,
    quick_box: int = 400,
    min_chars: int = 8,
    min_alnum: int = 4,
) -> str:
    """
    Two-stage OCR:
      1) Run a quick scan on a small thumbnail to detect potential text.
      2) If text seems likely, run full OCR on the original image (with light preprocessing).

    Returns empty string on failure to keep pipeline robust.
    """
    if not url:
        return ""
    try:
        r = session.get(url, timeout=timeout)
        if r.status_code != 200:
            return ""

        # Skip non-image responses (e.g., HTML pages)
        ctype = (r.headers.get("Content-Type") or "").lower()
        if ctype and ("image/" not in ctype):
            return ""

        if len(r.content) > max_bytes:
            return ""

        img = pil_image_mod.open(BytesIO(r.content)).convert("RGB")

        # Stage 1: quick thumbnail scan (cheap)
        small = img.copy()
        small.thumbnail((quick_box, quick_box))
        small_pp = _basic_ocr_preprocess(small, image_filter_mod=image_filter_mod, image_ops_mod=image_ops_mod)

        quick_cfg = "--oem 1 --psm 6 -l eng"
        quick = clean_text(pytesseract_mod.image_to_string(small_pp, config=quick_cfg))

        alnum = sum(ch.isalnum() for ch in quick)
        if len(quick) < min_chars or alnum < min_alnum:
            return ""

        # Stage 2: full OCR (still pretty fast, but much more accurate)
        full_pp = _basic_ocr_preprocess(img, image_filter_mod=image_filter_mod, image_ops_mod=image_ops_mod)
        full_cfg = "--oem 1 --psm 6 -l eng"
        full = clean_text(pytesseract_mod.image_to_string(full_pp, config=full_cfg))
        return full
    except Exception:
        return ""


# HTTP with exponential backoff
def fetch_json(session: requests.Session, url: str, params: Optional[Dict] = None) -> Optional[Dict]:
    for i in range(MAX_RETRIES):
        try:
            r = session.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()

            if r.status_code == 429 or (500 <= r.status_code < 600):
                wait = min(120, (BACKOFF_BASE ** (i + 1)) + random.uniform(0, 1.5))
                print(f"[WARN] HTTP {r.status_code}. sleeping {wait:.1f}s before retry... url={r.url}")
                time.sleep(wait)
                continue

            print(f"[WARN] HTTP {r.status_code} for {r.url}")
            return None
        except Exception as e:
            wait = min(60, (BACKOFF_BASE ** (i + 1)) + random.uniform(0, 1.0))
            print(f"[WARN] request failed ({e}). sleeping {wait:.1f}s before retry...")
            time.sleep(wait)
    return None


# Checkpoint handling
def load_checkpoint(path: str) -> Tuple[List[Dict], set, Dict]:
    """
    Load previous scraping state:
    - collected records
    - seen post IDs
    - pagination state per source
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
    """
    Save progress safely (write temp file then replace).
    """
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


# Parse Reddit listing response
def parse_listing(payload: Dict) -> Tuple[List[Dict], Optional[str]]:
    """
    Parse standard Reddit listing JSON and return:
      - list of post data dictionaries
      - next "after" token
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


# Build list of (subreddit, sort) sources
def build_sources(subs: List[str], sorts: List[str], use_global: bool) -> List[Tuple[str, str]]:
    """
    Generate scraping sources:
      (subreddit, sort) pairs,
      plus optional global search source.
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
    ap.add_argument(
        "--time_range",
        default="year",
        choices=["day", "week", "month", "year", "all"],
        help="time filter for Reddit listings",
    )
    ap.add_argument("--per_source_cap", type=int, default=1000, help="cap per (sub,sort) source")
    ap.add_argument("--min_text_len", type=int, default=MIN_TEXT_LEN)
    ap.add_argument("--sleep_min", type=float, default=SLEEP_RANGE[0])
    ap.add_argument("--sleep_max", type=float, default=SLEEP_RANGE[1])
    ap.add_argument("--checkpoint", default="ck_scrape.json", help="checkpoint path")
    ap.add_argument("--out_prefix", default="posts_lab5", help="output prefix for json/jsonl")
    ap.add_argument("--salt", default="dsci560", help="salt for username masking")

    # OCR configuration (fast two-stage mode)
    ap.add_argument("--ocr", action="store_true", help="Enable OCR for image posts")
    ap.add_argument("--tesseract_cmd", default="", help="Optional path to tesseract binary")
    ap.add_argument("--ocr_timeout", type=int, default=15, help="Image download timeout (seconds)")
    ap.add_argument("--ocr_max_bytes", type=int, default=5_000_000, help="Maximum allowed image size in bytes")
    ap.add_argument("--ocr_quick_box", type=int, default=400, help="Thumbnail size for quick OCR scan")
    ap.add_argument("--ocr_min_chars", type=int, default=8, help="Minimum characters threshold in quick scan")
    ap.add_argument("--ocr_min_alnum", type=int, default=4, help="Minimum alphanumeric characters threshold")
    ap.add_argument("--ocr_max_images_per_post", type=int, default=3, help="Maximum images to OCR per post")
    ap.add_argument("--ocr_budget_images", type=int, default=200, help="Maximum total images to OCR")

    args = ap.parse_args()

    target_n = args.N
    per_source_cap = min(args.per_source_cap, 1000)  # aligned with Reddit listing API limit
    sleep_range = (max(0.0, args.sleep_min), max(args.sleep_min, args.sleep_max))

    collected, seen, state = load_checkpoint(args.checkpoint)
    print(f"[INFO] resume: already have {len(collected)} posts; checkpoint={args.checkpoint}")

    # Initialize pagination tracking structure if missing
    if "after" not in state:
        state["after"] = {}
    if "count" not in state:
        state["count"] = {}

    subs_set = set(args.subs)

    # Initialize OCR dependencies only if enabled
    pytesseract_mod = None
    pil_image_mod = None
    image_filter_mod = None
    image_ops_mod = None
    ocr_images_used = 0
    if args.ocr:
        pytesseract_mod, pil_image_mod, image_filter_mod, image_ops_mod = _maybe_import_ocr()
        if args.tesseract_cmd:
            pytesseract_mod.pytesseract.tesseract_cmd = args.tesseract_cmd
        print(
            "[INFO] OCR enabled (fast two-stage): "
            f"quick_box={args.ocr_quick_box}, "
            f"budget_images={args.ocr_budget_images}, "
            f"max_images_per_post={args.ocr_max_images_per_post}"
        )

    with requests.Session() as s:
        s.headers.update(
            {
                "User-Agent": UA_JSON,
                "Accept-Language": "en-US,en;q=0.9",
            }
        )

        sources = build_sources(args.subs, args.sorts, args.use_global_search)

        # Initialize source-specific counters and pagination tokens
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

                # Construct URL and parameters for listing endpoint
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
                    url = f"https://www.reddit.com/r/{sub}/{mode}.json"
                    params = {"limit": POSTS_PER_REQUEST}
                    if mode == "top":
                        params["t"] = args.time_range
                    if after:
                        params["after"] = after

                payload = fetch_json(s, url, params=params)
                if not payload:
                    print(f"[WARN] listing fetch failed for {key_source}; cooling down 10s...")
                    time.sleep(10)
                    continue

                batch, after2 = parse_listing(payload)
                state["after"][key_source] = after2

                if not batch:
                    continue

                for d in batch:
                    if len(collected) >= target_n:
                        break
                    if state["count"][key_source] >= per_source_cap:
                        break

                    # If using global search, restrict to selected subreddits only
                    real_sub = d.get("subreddit") or sub
                    if sub == "_global_" and real_sub not in subs_set:
                        continue

                    if is_likely_ad_or_irrelevant(d):
                        continue

                    fullname = d.get("name")
                    if not fullname or fullname in seen:
                        continue
                    seen.add(fullname)

                    title = clean_text(d.get("title", ""))
                    selftext = clean_text(d.get("selftext", ""))
                    is_self = bool(d.get("is_self", False))

                    # If body text is too short, fall back to title
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

                    # Fast two-stage OCR (only applied to image posts)
                    ocr_text = ""
                    if args.ocr and is_image and (ocr_images_used < args.ocr_budget_images):
                        urls_to_ocr: List[str] = []
                        if image_url:
                            urls_to_ocr.append(image_url)
                        if gallery_urls:
                            remain = max(0, args.ocr_max_images_per_post - len(urls_to_ocr))
                            urls_to_ocr.extend(gallery_urls[:remain])
                        urls_to_ocr = urls_to_ocr[: args.ocr_max_images_per_post]

                        texts: List[str] = []
                        for u in urls_to_ocr:
                            if ocr_images_used >= args.ocr_budget_images:
                                break
                            t = ocr_maybe_text(
                                s,
                                u,
                                pytesseract_mod=pytesseract_mod,
                                pil_image_mod=pil_image_mod,
                                image_filter_mod=image_filter_mod,
                                image_ops_mod=image_ops_mod,
                                timeout=args.ocr_timeout,
                                max_bytes=args.ocr_max_bytes,
                                quick_box=args.ocr_quick_box,
                                min_chars=args.ocr_min_chars,
                                min_alnum=args.ocr_min_alnum,
                            )
                            ocr_images_used += 1
                            if t:
                                texts.append(t)

                        if texts:
                            ocr_text = " ".join(texts)

                    # Construct record for storage / downstream processing
                    rec = {
                        "fullname": fullname,
                        "post_id": d.get("id", ""),
                        "subreddit": real_sub,
                        "title": title,
                        "author": author_masked,
                        "author_raw": None,
                        "created_utc": created_utc,
                        "created": created_iso,
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
                        "ocr_text": ocr_text,
                        "keywords": [],
                        "topic": "",
                    }

                    collected.append(rec)
                    state["count"][key_source] += 1
                    progressed = True

                    if len(collected) % 50 == 0:
                        save_checkpoint(args.checkpoint, collected, seen, state)
                        print(f"[INFO] checkpoint saved: n={len(collected)} (ocr_images_used={ocr_images_used})")

                    time.sleep(random.uniform(*sleep_range))

                print(
                    f"[INFO] {key_source}: +{state['count'][key_source]} / cap={per_source_cap} | "
                    f"total={len(collected)}/{target_n}"
                )
                time.sleep(random.uniform(*sleep_range))

            if not progressed:
                print("[WARN] no progress in this round; sleeping 10s before retry...")
                save_checkpoint(args.checkpoint, collected, seen, state)
                time.sleep(10)

    # Write final outputs
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
    print(f"[INFO] OCR images processed total: {ocr_images_used}")
    print("\n[SAMPLE] first 3 posts:")
    for p in collected[:3]:
        print("-", p["subreddit"], "|", p["title"][:60], "| is_image=", p["is_image"], "| created=", p.get("created"))


if __name__ == "__main__":
    main()
