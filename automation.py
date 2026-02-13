import argparse
import os
import subprocess
import time
from datetime import datetime
from typing import List


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ts_compact() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def log(msg: str):
    print(f"[{now_str()}] {msg}", flush=True)


def run_cmd(cmd: List[str], desc: str) -> bool:
    log(f"{desc} ...")
    try:
        r = subprocess.run(cmd, check=True)
        return r.returncode == 0
    except subprocess.CalledProcessError as e:
        log(f"[ERROR] {desc} failed (exit={e.returncode}). cmd={' '.join(cmd)}")
        return False
    except FileNotFoundError as e:
        log(f"[ERROR] {desc} failed: {e}")
        return False


def expected_scraped_json(prefix: str, n: int) -> str:
    # fetch_reddit.py writes: {out_prefix}_{N}.json
    return f"{prefix}_{n}.json"


def expected_scraped_jsonl(prefix: str, n: int) -> str:
    # fetch_reddit.py writes: {out_prefix}_{N}.jsonl
    return f"{prefix}_{n}.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("interval_min", type=int, help="Run every N minutes (e.g., 5)")
    ap.add_argument("--fetch_n", type=int, default=300, help="How many posts to fetch per cycle")
    ap.add_argument("--k", type=int, default=8, help="Number of clusters")
    ap.add_argument("--scraper", default="fetch_reddit.py", help="Scraper script filename")
    ap.add_argument("--embedder", default="embed_and_cluster.py", help="Embedding+clustering script filename")
    ap.add_argument("--out_dir", default="outputs", help="Output directory (default: outputs)")
    ap.add_argument("--sleep_min", type=float, default=0.8, help="Scraper sleep_min")
    ap.add_argument("--sleep_max", type=float, default=1.6, help="Scraper sleep_max")
    ap.add_argument("--plot", action="store_true", help="Generate PCA plot (clusters_plot.png)")

    # DB loader (auto ingest)
    ap.add_argument("--loader", default="load_jsonl_to_mysql.py", help="Loader script filename")
    ap.add_argument("--table", default="reddit_posts", help="MySQL table name for ingestion")
    ap.add_argument("--no_db", action="store_true", help="Disable DB ingestion step")

    # OCR passthrough — keeps automation aligned with fetch_reddit.py OCR feature
    ap.add_argument("--ocr", action="store_true", help="Enable OCR in the scraper step")
    ap.add_argument("--tesseract_cmd", default="", help="Optional path to tesseract binary, e.g. /usr/bin/tesseract")
    ap.add_argument("--ocr_timeout", type=int, default=15, help="OCR image download timeout (sec)")
    ap.add_argument("--ocr_max_bytes", type=int, default=5_000_000, help="Max bytes per image download")
    ap.add_argument("--ocr_max_images_per_post", type=int, default=3, help="Max images to OCR per post")
    ap.add_argument("--ocr_budget_images", type=int, default=200, help="Max total images to OCR per cycle")

    args = ap.parse_args()

    interval_sec = max(1, args.interval_min) * 60
    os.makedirs(args.out_dir, exist_ok=True)

    cycle = 0
    log(
        f"Automation started (FULL AUTO). interval={args.interval_min} min, "
        f"fetch_n={args.fetch_n}, k={args.k}, "
        f"db_ingest={'OFF' if args.no_db else 'ON'}, "
        f"ocr={'ON' if args.ocr else 'OFF'}"
    )

    while True:
        cycle += 1
        t0 = time.time()
        stamp = ts_compact()

        # Each cycle independent (fresh checkpoint & prefix) => "latest snapshot" mode
        ck = f"ck_scrape_auto_{stamp}.json"
        prefix = f"posts_auto_{stamp}"

        scraped_json = expected_scraped_json(prefix, args.fetch_n)
        scraped_jsonl = expected_scraped_jsonl(prefix, args.fetch_n)

        log(f"=== Cycle {cycle} start ===")

        # 1) Scrape
        cmd1 = [
            "python3", args.scraper, str(args.fetch_n),
            "--checkpoint", ck,
            "--out_prefix", prefix,
            "--sleep_min", str(args.sleep_min),
            "--sleep_max", str(args.sleep_max),
        ]

        # Pass OCR flags through to fetch_reddit.py if enabled
        if args.ocr:
            cmd1 += [
                "--ocr",
                "--ocr_timeout", str(args.ocr_timeout),
                "--ocr_max_bytes", str(args.ocr_max_bytes),
                "--ocr_max_images_per_post", str(args.ocr_max_images_per_post),
                "--ocr_budget_images", str(args.ocr_budget_images),
            ]
            if args.tesseract_cmd:
                cmd1 += ["--tesseract_cmd", args.tesseract_cmd]

        ok1 = run_cmd(cmd1, desc=f"Scraping latest posts (N={args.fetch_n})")

        if not ok1:
            log("[WARN] Scrape failed; skipping the rest of this cycle.")
        else:
            # verify outputs
            if not os.path.exists(scraped_json) or not os.path.exists(scraped_jsonl):
                if not os.path.exists(scraped_json):
                    log(f"[ERROR] Expected scraped JSON not found: {scraped_json}")
                if not os.path.exists(scraped_jsonl):
                    log(f"[ERROR] Expected scraped JSONL not found: {scraped_jsonl}")
                log("[ERROR] Check fetch_reddit.py naming; should write {out_prefix}_{N}.json and .jsonl")
            else:
                log(f"[INFO] Using scraped file: {scraped_json}")
                log(f"[INFO] Using scraped jsonl for DB: {scraped_jsonl}")

                # 2) Ingest into MySQL
                if args.no_db:
                    log("[INFO] DB ingestion disabled (--no_db).")
                    ok_db = True
                else:
                    ok_db = run_cmd(
                        ["python3", args.loader, scraped_jsonl, "--table", args.table],
                        desc=f"Ingesting into MySQL (table={args.table})"
                    )
                    if ok_db:
                        log("[OK] DB ingestion complete.")
                    else:
                        log("[WARN] DB ingestion failed; continuing to clustering anyway.")

                # 3) Embed + cluster
                cmd2 = [
                    "python3", args.embedder,
                    "--input", scraped_json,
                    "--k", str(args.k),
                    "--out_dir", args.out_dir,
                ]

                if args.plot:
                    cmd2.append("--plot")

                # Only write back to DB if DB ingestion is enabled
                if not args.no_db:
                    cmd2 += ["--write_db", "--table", args.table]

                ok2 = run_cmd(cmd2, desc=f"Embedding + clustering (k={args.k})")

                if ok2:
                    log("[OK] Cycle artifacts updated:")
                    log(f"  - {os.path.join(args.out_dir, 'clusters_posts.csv')}")
                    log(f"  - {os.path.join(args.out_dir, 'clusters_summary.json')}")
                    if args.plot:
                        log(f"  - {os.path.join(args.out_dir, 'clusters_plot.png')}")
                else:
                    log("[WARN] Clustering failed; will retry next cycle.")

        elapsed = int(time.time() - t0)
        log(f"=== Cycle {cycle} end (elapsed={elapsed}s) ===")
        log(f"Next cycle in ~{interval_sec} seconds.")
        time.sleep(interval_sec)


if __name__ == "__main__":
    main()
