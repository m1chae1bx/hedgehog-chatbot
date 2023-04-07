import sqlite3
from contextlib import contextmanager
from config import SQLITE_DB_PATH
import functools
from typing import Tuple


@contextmanager
def sqlite_connection():
    conn = sqlite3.connect(SQLITE_DB_PATH)
    c = conn.cursor()
    try:
        yield c
    finally:
        conn.close()


@functools.lru_cache(maxsize=10)
def fetch_article(article_id: int, c: sqlite3.Cursor) -> Tuple[str, str]:
    query = "SELECT title, published FROM articles WHERE id = ?;"
    c.execute(query, (article_id,))
    res = c.fetchone()
    title = res[0]
    date_published = res[1].split("T")[0]
    return title, date_published


def fetch_article_split_text(article_id: int, position: int, c: sqlite3.Cursor) -> str:
    query = "SELECT text FROM articles_split WHERE article_id = ? AND " "position = ?;"
    c.execute(query, (article_id, position))
    text = c.fetchone()[0]
    return text
