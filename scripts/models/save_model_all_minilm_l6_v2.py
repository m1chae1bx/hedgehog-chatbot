import os
from sentence_transformers import SentenceTransformer

MODELS_DIR = "models"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_PATH = os.path.join(MODELS_DIR, MODEL_NAME)

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

model = SentenceTransformer(MODEL_NAME)
model.save(OUTPUT_PATH)
