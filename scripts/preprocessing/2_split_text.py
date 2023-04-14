import os
import sqlite3
from tqdm import tqdm

import pandas as pd
from transformers import AutoTokenizer
from langchain.text_splitter import CharacterTextSplitter

MODEL_PATH = "models/sentence-transformers/all-MiniLM-L6-v2"
CLEANED_JSON_DATA_PATH = "data/processed/cleaned_json"
SQLITE_DB_DIR = "data/processed/sqlite_db"
SQLITE_DB_NAME = "articles.db"


def save_df_to_json(df, filename, output_dir):
    output_path = os.path.join(output_dir, filename)
    df.to_json(output_path, orient="records", lines=True)


def prepare_sqlite_db(db_path, db_name):
    if not os.path.exists(db_path):
        os.makedirs(db_path)

    conn = sqlite3.connect(os.path.join(db_path, db_name))
    c = conn.cursor()
    c.execute("DROP TABLE IF EXISTS articles;")
    c.execute(
        """
        CREATE TABLE articles (
            id integer PRIMARY KEY AUTOINCREMENT,
            published datetime,
            title text
        );
        """
    )
    c.execute("DROP TABLE IF EXISTS articles_split;")
    c.execute(
        """
        CREATE TABLE articles_split (
            id integer PRIMARY KEY AUTOINCREMENT,
            article_id integer,
            text text,
            position integer,
            FOREIGN KEY (article_id) REFERENCES articles (id)
        );
        """
    )
    conn.commit()

    return conn


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
sqlite3_conn = prepare_sqlite_db(SQLITE_DB_DIR, SQLITE_DB_NAME)

for file in tqdm(os.listdir(CLEANED_JSON_DATA_PATH)):
    if file.endswith(".json"):
        df = pd.read_json(os.path.join(CLEANED_JSON_DATA_PATH, file), lines=True)

        for index, row in df.iterrows():
            c = sqlite3_conn.cursor()
            c.execute(
                "INSERT INTO articles (published, title) VALUES (?, ?)",
                (row["published"], row["title"]),
            )
            article_id = c.lastrowid

            text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
                tokenizer,
                chunk_size=50,
                chunk_overlap=0,
                separator="\n",
            )
            texts = text_splitter.split_text(row["text"])

            for i, text in enumerate(texts):
                c.execute(
                    "INSERT INTO articles_split (article_id, text, position) VALUES (?, ?, ?)",
                    (article_id, text, i),
                )

            sqlite3_conn.commit()

sqlite3_conn.close()
