import os
import sqlite3
import pandas
from typing import Optional
from tqdm import tqdm

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

SQLITE_DB_PATH = "data/processed/sqlite_db/articles.db"
SENTENCE_EMBEDDING_MODEL_PATH = "models/sentence-transformers/all-MiniLM-L6-v2"
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

# loop through the dataframe per 1000 rows
faiss: Optional[FAISS] = None
for i in tqdm(range(0, len(df), 1000)):
    # get the 1000 rows
    df_subset = df[i : i + 1000]
    # merge the title and text columns
    merged_title_text = "Title: " + df_subset["title"] + ". " + df_subset["text"]
    merged_title_text_list = merged_title_text.tolist()
    df_subset = df_subset.drop(columns=["text", "title"])
    metadata = df_subset.to_dict(orient="records")
    if faiss is None:
        faiss = FAISS.from_texts(merged_title_text_list, embeddings, metadata)
    else:
        temp_faiss = FAISS.from_texts(merged_title_text_list, embeddings, metadata)
        faiss.merge_from(temp_faiss)

if faiss is not None:
    faiss.save_local(DOCSTORE_PATH)
else:
    raise Exception("No data found in the database")

# merged_title_text = "Title: " + df["title"] + ". " + df["text"]
# merged_title_text_list = merged_title_text.tolist()
# df = df.drop(columns=["text", "title"])
# metadata = df.to_dict(orient="records")
# faiss = FAISS.from_texts(merged_title_text_list, embeddings, metadata)
# faiss.save_local(DOCSTORE_PATH)
