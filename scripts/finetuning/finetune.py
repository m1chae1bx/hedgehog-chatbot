from chat.config import MODEL_PATH
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = TFAutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH, True)

# Train the model
