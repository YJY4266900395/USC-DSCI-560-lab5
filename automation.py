# automation.py
# Runs clustering pipeline periodically.
# When NOT updating, provides interactive query prompt.
# (every 5 minutes: re-run clustering over latest scraped_posts.json)

import sys
import time
from datetime import datetime

from clustering import run_clustering_pipeline
from query_interface import interactive_loop


def main():
    if len(sys.argv) < 2:
        print("Usage: python automation.py [interval_minutes]")
        print("Example: python automation.py 5")
        sys.exit(1)

    try:
        interval_min = int(sys.argv[1])
        if interval_min <= 0:
            raise ValueError
    except ValueError:
        print("[ERROR] interval_minutes must be a positive integer.")
        sys.exit(1)

    interval_sec = interval_min * 60

    print("=" * 70)
    print("DSCI-560 Lab5 Automation Runner")
    print(f"Interval: every {interval_min} minute(s)")
    print("Data input: scraped_posts.json")
    print("Output: clustered_posts.json + cluster_plot.png")
    print("=" * 70)

    last_run = 0

    while True:
        now = time.time()
        if now - last_run >= interval_sec:
            print("\n" + "-" * 70)
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Updating clusters...")
            try:
                summary = run_clustering_pipeline(
                    input_json="scraped_posts.json",
                    output_json="clustered_posts.json",
                    out_png="cluster_plot.png",
                    k=8,
                )
                print("[OK] Update complete.")
                print(summary)
            except Exception as e:
                print(f"[ERROR] Update failed: {e}")

            last_run = time.time()
            print("-" * 70)

        print("\n[Idle Mode] You can query now. (Type 'exit' to return to scheduler loop)")
        try:
            interactive_loop("clustered_posts.json")
        except Exception as e:
            print(f"[WARN] Query interface error: {e}")

        # After user exits query loop, continue scheduler
        # Small sleep to avoid busy loop
        time.sleep(1)


if __name__ == "__main__":
    main()
