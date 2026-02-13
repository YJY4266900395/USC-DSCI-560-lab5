import argparse
import hashlib
import html
import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import mysql.connector

DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER", "lab5")
DB_PASS = os.getenv("DB_PASS", "lab5pass")
DB_NAME = os.getenv("DB_NAME", "dsci560_lab5")

# very small stopword set (no extra deps)
STOPWORDS = {
    "a","an","the","and","or","but","if","then","else","for","to","of","in","on","at","by","with","as",
    "is","are","was","were","be","been","being","it","this","that","these","those","i","you","he","she",
    "they","we","me","him","her","them","us","my","your","our","their","from","not","no","yes",
    "about","into","over","under","after","before","between","during","than","too","very",
    "can","could","should","would","will","just","also","more","most","some","any","all",
}

TAG_RE = re.compile(r"<[^>]+>")
NON_WORD_RE = re.compile(r"[^0-9A-Za-z_]+")
WS_RE = re.compile(r"\s+")


def clean_text(s: Optional[str]) -> str:
    if not s:
        return ""
    s = html.unescape(s)
    s = TAG_RE.sub(" ", s)
    s = s.replace("\u200b", " ").replace("\ufeff", " ")
    s = WS_RE.sub(" ", s).strip()
    return s


def mask_author(author: Optional[str]) -> str:
    if not author:
        return "user_unknown"
    # if upstream already masked, keep it
    if author.startswith("user_") or author.startswith("anon_") or author.startswith("masked_"):
        return author
    h = hashlib.sha256(author.encode("utf-8")).hexdigest()[:10]
    return f"user_{h}"


def extract_keywords(text: str, top_k: int = 12) -> List[str]:
    text = text.lower()
    tokens = [t for t in NON_WORD_RE.split(text) if t]
    tokens = [t for t in tokens if len(t) >= 3 and t not in STOPWORDS]
    if not tokens:
        return []
    freq: Dict[str, int] = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    # sort by freq desc then token
    items = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    return [w for (w, _) in items[:top_k]]


def ensure_table(cur, table: str):
    # Create a robust table that matches PDF expectations (extra fields)
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS `{table}` (
          `fullname`   VARCHAR(64)  NOT NULL,
          `subreddit`  VARCHAR(64)  NULL,
          `title`      TEXT         NULL,
          `body`       LONGTEXT     NULL,
          `final_text` LONGTEXT     NULL,
          `author`     VARCHAR(128) NULL,
          `created`    VARCHAR(64)  NULL,
          `permalink`  TEXT         NULL,
          `out_url`    TEXT         NULL,
          `is_image`   TINYINT      DEFAULT 0,

          -- PDF extra fields:
          `topic`      VARCHAR(128) NULL,
          `keywords`   JSON         NULL,
          `ocr_text`   LONGTEXT     NULL,

          `ingested_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

          PRIMARY KEY (`fullname`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """
    )

    # Add missing columns if you already had a simpler table
    cur.execute(f"SHOW COLUMNS FROM `{table}`;")
    existing = {row[0] for row in cur.fetchall()}

    def add_col(sql: str, col: str):
        if col not in existing:
            cur.execute(sql)

    add_col(f"ALTER TABLE `{table}` ADD COLUMN `topic` VARCHAR(128) NULL;", "topic")
    add_col(f"ALTER TABLE `{table}` ADD COLUMN `keywords` JSON NULL;", "keywords")
    add_col(f"ALTER TABLE `{table}` ADD COLUMN `ocr_text` LONGTEXT NULL;", "ocr_text")
    add_col(
        f"ALTER TABLE `{table}` ADD COLUMN `ingested_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP;",
        "ingested_at",
    )


def norm_row(row: Dict[str, Any]) -> Dict[str, Any]:
    subreddit = row.get("subreddit") or row.get("topic") or ""
    title = clean_text(row.get("title"))
    body = clean_text(row.get("body"))
    final_text = clean_text(row.get("final_text") or (title + "\n" + body))

    # OCR text might be stored by your scraper under different keys
    ocr_text = clean_text(
        row.get("ocr_text")
        or row.get("image_text")
        or row.get("ocr")
        or ""
    )

    # topic: if provided use it, else fallback to subreddit
    topic = clean_text(row.get("topic") or subreddit)

    # keywords: if JSONL provides them use; else compute from final_text + ocr_text
    kws = row.get("keywords")
    if isinstance(kws, list):
        keywords = [clean_text(str(x)).lower() for x in kws if str(x).strip()]
    else:
        keywords = extract_keywords((final_text + " " + ocr_text).strip(), top_k=12)

    author = mask_author(clean_text(row.get("author")))
    created = row.get("created")
    if isinstance(created, (int, float)):
        # convert unix seconds to ISO-like string
        created = datetime.utcfromtimestamp(created).isoformat()
    created = clean_text(str(created)) if created is not None else ""

    return {
        "fullname": clean_text(row.get("fullname")),
        "subreddit": clean_text(subreddit),
        "title": title,
        "body": body,
        "final_text": final_text,
        "author": author,
        "created": created,
        "permalink": clean_text(row.get("permalink")),
        "out_url": clean_text(row.get("out_url")),
        "is_image": 1 if row.get("is_image") else 0,
        "topic": topic,
        "keywords": json.dumps(keywords, ensure_ascii=False),
        "ocr_text": ocr_text,
    }


def build_upsert_sql(table: str) -> str:
    return f"""
    INSERT INTO `{table}`
      (fullname, subreddit, title, body, final_text, author, created, permalink, out_url, is_image, topic, keywords, ocr_text)
    VALUES
      (%(fullname)s, %(subreddit)s, %(title)s, %(body)s, %(final_text)s, %(author)s, %(created)s, %(permalink)s, %(out_url)s, %(is_image)s, %(topic)s, %(keywords)s, %(ocr_text)s)
    ON DUPLICATE KEY UPDATE
      subreddit=VALUES(subreddit),
      title=VALUES(title),
      body=VALUES(body),
      final_text=VALUES(final_text),
      author=VALUES(author),
      created=VALUES(created),
      permalink=VALUES(permalink),
      out_url=VALUES(out_url),
      is_image=VALUES(is_image),
      topic=VALUES(topic),
      keywords=VALUES(keywords),
      ocr_text=VALUES(ocr_text),
      ingested_at=CURRENT_TIMESTAMP;
    """


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl_path", help="Path to JSONL file")
    ap.add_argument("--table", default="reddit_posts", help="MySQL table name (default: reddit_posts)")
    ap.add_argument("--batch", type=int, default=200, help="Commit every N rows (default: 200)")
    args = ap.parse_args()

    if not os.path.exists(args.jsonl_path):
        raise FileNotFoundError(f"JSONL not found: {args.jsonl_path}")

    conn = mysql.connector.connect(
        host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASS, database=DB_NAME
    )
    cur = conn.cursor()

    ensure_table(cur, args.table)
    conn.commit()

    upsert_sql = build_upsert_sql(args.table)

    inserted = 0
    bad = 0
    with open(args.jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                data = norm_row(row)
                # skip rows without PK
                if not data["fullname"]:
                    bad += 1
                    continue
                cur.execute(upsert_sql, data)
                inserted += 1
                if inserted % args.batch == 0:
                    conn.commit()
            except Exception:
                bad += 1

    conn.commit()
    cur.close()
    conn.close()

    print(f"[DONE] upserted {inserted} rows from {args.jsonl_path} into {DB_NAME}.{args.table} (bad_lines={bad})")


if __name__ == "__main__":
    main()