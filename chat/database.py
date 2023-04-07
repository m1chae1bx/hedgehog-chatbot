import sqlite3
from contextlib import contextmanager
from config import SQLITE_DB_PATH
import functools


@contextmanager
def sqlite_connection():
    with sqlite3.connect(SQLITE_DB_PATH) as conn:
        c = conn.cursor()
        try:
            yield c
        finally:
            conn.close()


@functools.lru_cache(maxsize=10)
def fetch_article_title(article_id: int, c) -> str:
    query = "SELECT title FROM articles WHERE id = ?;"
    c.execute(query, (article_id,))
    title = c.fetchone()[0]
    return title


def fetch_article_split_text(article_id: int, position: int, c) -> str:
    query = "SELECT text FROM articles_split WHERE article_id = ? AND " "position = ?;"
    c.execute(query, (article_id, position))
    text = c.fetchone()[0]
    return text
