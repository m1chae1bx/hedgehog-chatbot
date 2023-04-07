import os
import sqlite3
import pandas
from langchain.vectorstores import FAISS

from langchain.embeddings import HuggingFaceEmbeddings

# from langchain.embeddings.base import Embeddings
# from typing import Any, List

# from pydantic import BaseModel

# import sentence_transformers
# from transformers import AutoTokenizer


# SPLIT_TEXT_JSON_DATA_PATH = "data/processed/split_text_json"
SQLITE_DB_PATH = "data/processed/sqlite_db/articles.db"
SENTENCE_EMBEDDING_MODEL_PATH = "models/sentence-transformers/all-MiniLM-L6-v2"
# SENTENCE_EMBEDDING_MODEL_PATH = "models/microsoft/GODEL-v1_1-large-seq2seq"
DOCSTORE_PATH = "data/processed/docstore"


embeddings = HuggingFaceEmbeddings(model_name=SENTENCE_EMBEDDING_MODEL_PATH)

if not os.path.exists(DOCSTORE_PATH):
    os.makedirs(DOCSTORE_PATH)
else:
    for file in os.listdir(DOCSTORE_PATH):
        os.remove(os.path.join(DOCSTORE_PATH, file))


# Load rows from sqlite db in table articles_split
conn = sqlite3.connect(SQLITE_DB_PATH)
c = conn.cursor()
query = (
    "SELECT asplit.*, a.title FROM articles_split as asplit JOIN articles a "
    "ON a.id = asplit.article_id;"
)
df = pandas.read_sql_query(query, conn)
# Combine the text and title columns and store in a list
merged_title_text = "Title: " + df["title"] + ". " + df["text"]

merged_title_text_list = merged_title_text.tolist()
df = df.drop(columns=["text", "title"])
metadata = df.to_dict(orient="records")
faiss = FAISS.from_texts(merged_title_text_list, embeddings, metadata)
faiss.save_local(DOCSTORE_PATH)
