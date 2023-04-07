import os
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM

MODELS_DIR = "models"
MODEL_NAME = "microsoft/GODEL-v1_1-large-seq2seq"
OUTPUT_PATH = os.path.join(MODELS_DIR, MODEL_NAME)

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = TFAutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, from_pt=True)
tokenizer.save_pretrained(OUTPUT_PATH)
model.save_pretrained(OUTPUT_PATH)
