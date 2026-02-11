import requests
import json
import time
import sys
from datetime import datetime, timezone

# ============ CONFIG ============
SUBREDDITS = ["tech", "cybersecurity", "technology", "artificial", "datascience", "computerscience"]
SORT_METHODS = ["hot", "new", "top", "rising"]  # Bypass post limitation of .json endpoints by using different sort methods
POSTS_PER_REQUEST = 100  # Reddit max per request

# Use a descriptive User-Agent (Reddit recommends this for .json endpoints)
HEADERS = {
    "User-Agent": "DSCI560-Lab5-Scraper/1.0 (Educational Project)"
}

# Rate limiting config
REQUEST_DELAY = 1.5      # seconds between requests (be polite)
MAX_RETRIES = 5
BACKOFF_FACTOR = 2        # exponential backoff multiplier


def fetch_json(url, params=None, retries=0):
    """
    Fetch a Reddit .json endpoint with exponential backoff on rate limits.
    
    The 'backoff' your teammate mentioned: when Reddit returns HTTP 429 
    (Too Many Requests), we wait an exponentially increasing amount of time 
    before retrying: 2s, 4s, 8s, 16s, 32s.
    """
    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=30)

        if response.status_code == 200:
            return response.json()

        elif response.status_code == 429:
            if retries < MAX_RETRIES:
                wait_time = BACKOFF_FACTOR ** (retries + 1)
                print(f"  [429 Rate Limited] Waiting {wait_time}s before retry ({retries+1}/{MAX_RETRIES})...")
                time.sleep(wait_time)
                return fetch_json(url, params, retries + 1)
            else:
                print(f"  [ERROR] Max retries reached. Skipping.")
                return None

        elif response.status_code >= 500:
            if retries < MAX_RETRIES:
                wait_time = BACKOFF_FACTOR ** (retries + 1)
                print(f"  [Server Error {response.status_code}] Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
                return fetch_json(url, params, retries + 1)
            else:
                print(f"  [ERROR] Max retries reached after server errors.")
                return None

        else:
            print(f"  [ERROR] HTTP {response.status_code} for {url}")
            return None

    except requests.exceptions.Timeout:
        if retries < MAX_RETRIES:
            wait_time = BACKOFF_FACTOR ** (retries + 1)
            print(f"  [Timeout] Waiting {wait_time}s before retry...")
            time.sleep(wait_time)
            return fetch_json(url, params, retries + 1)
        return None
    except requests.exceptions.RequestException as e:
        print(f"  [ERROR] Request exception: {e}")
        return None


def parse_posts_from_json(json_data, subreddit):
    """
    Parse posts from Reddit's JSON response.
    
    The JSON structure is:
    {
        "data": {
            "children": [
                {"data": { ...post fields... }},
                ...
            ],
            "after": "t3_xxxxx"   <-- pagination token
        }
    }
    """
    posts = []

    if not json_data or "data" not in json_data:
        return posts, None

    children = json_data["data"].get("children", [])
    after_token = json_data["data"].get("after", None)

    for child in children:
        if child.get("kind") != "t3":  # t3 = link/post, skip others
            continue

        post = child["data"]

        # Skip promoted/ad posts
        if post.get("promoted") or post.get("is_reddit_media_domain") is None:
            pass  # not all promoted posts have this flag, check stickied too
        if post.get("stickied", False):
            print(f"  [SKIP] Stickied post: {post.get('title', '')[:50]}")
            continue

        # Determine if this is an image post
        url = post.get("url", "")
        image_extensions = (".jpg", ".jpeg", ".png", ".gif", ".webp")
        is_image = url.lower().endswith(image_extensions)

        # Also check for Reddit-hosted images
        if not is_image and "i.redd.it" in url:
            is_image = True

        # Check for gallery posts (multiple images)
        is_gallery = post.get("is_gallery", False)

        # Extract gallery image URLs for multi-image posts.
        # Gallery posts store image metadata in 'media_metadata' field.
        gallery_urls = []
        if is_gallery and "media_metadata" in post:
            for media_id, media_info in post["media_metadata"].items():
                if media_info.get("status") == "valid" and "s" in media_info:
                    img_u = media_info["s"].get("u", "")
                    if img_u:
                        gallery_urls.append(img_u.replace("&amp;", "&"))
            if gallery_urls:
                is_image = True

        # Get thumbnail URL
        thumbnail = post.get("thumbnail", "")
        if thumbnail in ("self", "default", "nsfw", "spoiler", ""):
            thumbnail = ""

        # Get image URL from preview if available
        image_url = ""
        if is_image:
            image_url = url
        elif "preview" in post and "images" in post["preview"]:
            try:
                image_url = post["preview"]["images"][0]["source"]["url"]
                # Reddit HTML-encodes the URL in preview
                image_url = image_url.replace("&amp;", "&")
            except (IndexError, KeyError):
                pass

        # Build the post dict
        parsed = {
            "id": post.get("name", ""),           # fullname like t3_xxxxx
            "post_id": post.get("id", ""),         # short id
            "subreddit": subreddit,
            "title": post.get("title", ""),
            "author": post.get("author", "[deleted]"),
            "created_utc": post.get("created_utc", 0),
            "created_datetime": datetime.fromtimestamp(
                post.get("created_utc", 0), tz=timezone.utc
            ).strftime("%Y-%m-%d %H:%M:%S"),
            "score": post.get("score", 0),
            "upvote_ratio": post.get("upvote_ratio", 0),
            "num_comments": post.get("num_comments", 0),
            "url": url,
            "permalink": f"https://www.reddit.com{post.get('permalink', '')}",
            "domain": post.get("domain", ""),
            "selftext": post.get("selftext", ""),
            "is_self": post.get("is_self", False),
            "is_image": is_image,
            "is_gallery": is_gallery,
            "image_url": image_url,
            "gallery_urls": gallery_urls,
            "thumbnail": thumbnail,
            "link_flair_text": post.get("link_flair_text", ""),
            "over_18": post.get("over_18", False),
        }
        posts.append(parsed)

    return posts, after_token


def scrape_subreddit(subreddit, num_posts, sort="new"):
    """
    Scrape posts from a subreddit using the .json endpoint.
    
    Pagination: each request returns up to 100 posts and an 'after' token.
    We keep requesting until we have enough posts or there are no more.
    
    For 5000 posts: 5000 / 100 = 50 requests, at ~1.5s each = ~75 seconds.

    Tries multiple sort methods (hot, new, top, rising) to maximize
    unique posts. Reddit caps each sort listing at ~1000 posts, so using
    multiple sorts lets us get more. Posts are deduplicated by post_id.
    """
    seen_ids = set()
    all_posts = []

    for sort in SORT_METHODS:
        if len(all_posts) >= num_posts:
            break

        base_url = f"https://www.reddit.com/r/{subreddit}/{sort}.json"
        after = None
        page = 1

        # For 'top' sort, add timeframe param to get more historical posts
        extra_params = {}
        if sort == "top":
            extra_params["t"] = "all"

        print(f"\n{'='*60}")
        print(f"Scraping r/{subreddit} | Sort: {sort} | Target: {num_posts} | Have: {len(all_posts)}")
        print(f"{'='*60}")

        while len(all_posts) < num_posts:
            params = {"limit": POSTS_PER_REQUEST, **extra_params}
            if after:
                params["after"] = after

            print(f"  [Page {page}] Fetching (after={after or 'None'})...")
            json_data = fetch_json(base_url, params=params)

            if not json_data:
                print("  Failed to fetch page. Moving to next sort method.")
                break

            posts, after = parse_posts_from_json(json_data, subreddit)

            if not posts:
                print(f"  No more posts from sort={sort}. Moving on.")
                break

            # Deduplicate, only add posts we haven't seen before
            new_count = 0
            for p in posts:
                if p["post_id"] not in seen_ids:
                    seen_ids.add(p["post_id"])
                    all_posts.append(p)
                    new_count += 1

            print(f"  Got {len(posts)} posts, {new_count} new. Total unique: {len(all_posts)}")

            if not after:
                print(f"  No 'after' token for sort={sort}. End of listing.")
                break

            page += 1
            time.sleep(REQUEST_DELAY)

    return all_posts[:num_posts]


def main():
    # Parse command line argument for number of posts
    total_target = 5000  # default
    if len(sys.argv) > 1:
        try:
            num_posts = int(sys.argv[1])
        except ValueError:
            print("Usage: python reddit_scraper.py [total_num_posts]")
            print("Example: python reddit_scraper.py 5000")
            sys.exit(1)

    per_sub_target = (total_target // len(SUBREDDITS)) + 1

    print(f"Target: {num_posts} posts per subreddit")
    print(f"Subreddits: {SUBREDDITS}")

    all_posts = []
    for sub in SUBREDDITS:
        # Stop early if we already have enough total posts
        if len(all_posts) >= total_target:
            print(f"\nAlready reached {total_target} posts. Skipping r/{sub}.")
            break

        remaining = total_target - len(all_posts)
        target = min(per_sub_target, remaining)

        posts = scrape_subreddit(sub, target)
        all_posts.extend(posts)
        print(f"\nr/{sub} done: scraped {len(posts)} posts | Running total: {len(all_posts)}")

    # Final deduplicate across subreddits
    seen_ids = set()
    unique_posts = []
    for p in all_posts:
        if p["post_id"] not in seen_ids:
            seen_ids.add(p["post_id"])
            unique_posts.append(p)
    all_posts = unique_posts[:total_target]


    # Summary
    print(f"\n{'='*60}")
    print(f"SCRAPING COMPLETE")
    print(f"Total unique posts: {len(all_posts)}")
    print(f"{'='*60}")

    # Preview first 5 posts
    print("\n--- Preview (first 5 posts) ---")
    for i, post in enumerate(all_posts[:5]):
        print(f"\n[{i+1}] r/{post['subreddit']} | Score: {post['score']} | Comments: {post['num_comments']}")
        print(f"    Title: {post['title'][:80]}")
        print(f"    Author: {post['author']}")
        print(f"    Date: {post['created_datetime']}")
        print(f"    Image: {'Yes' if post['is_image'] else 'No'} | Self-post: {'Yes' if post['is_self'] else 'No'}")
        if post["selftext"]:
            print(f"    Text: {post['selftext'][:120]}...")

    # Count stats
    image_posts = sum(1 for p in all_posts if p["is_image"] or p["is_gallery"])
    self_posts = sum(1 for p in all_posts if p["is_self"])
    link_posts = len(all_posts) - self_posts
    # Per-subreddit breakdown
    sub_counts = {}
    for p in all_posts:
        sub_counts[p["subreddit"]] = sub_counts.get(p["subreddit"], 0) + 1

    print(f"\n--- Stats ---")
    print(f"Self-text posts: {self_posts}")
    print(f"Link posts:      {link_posts}")
    print(f"Image/gallery posts:     {image_posts}")
    for sub, count in sub_counts.items():
        print(f"  r/{sub}: {count}")

    # Save to JSON
    output_file = "scraped_posts.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_posts, f, ensure_ascii=False, indent=2)
    print(f"\nData saved to {output_file}")


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"\nTotal execution time: {time.time() - start:.2f} seconds")