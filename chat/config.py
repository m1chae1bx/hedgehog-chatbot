from datetime import datetime

# Path to the pre-trained seq2seq model used for generating responses.
MODEL_PATH = "models/microsoft/GODEL-v1_1-large-seq2seq"

# Path to the processed document store containing indexed articles.
DOCSTORE_PATH = "data/processed/docstore"

# Path to the pre-trained sentence embedding model used for document similarity search.
SENTENCE_EMBEDDING_MODEL_PATH = "models/sentence-transformers/all-MiniLM-L6-v2"

# Path to the SQLite database containing article information.
SQLITE_DB_PATH = "data/processed/sqlite_db/articles.db"

# Instruction string for the chatbot model.
INSTRUCTION = "Instruction: given a dialog context and related knowledge, you need to response safely based on the knowledge."

# Starting dialog for the chatbot.
STARTING_DIALOG = "Hello, I'm here to answer your questions regarding financial news."

# Number of top similar documents to consider during the search.
SIMILARITY_SEARCH_K = 7

# Number of top documents to consider for max marginal relevance search.
TOP_K = 3

# Maximum token limit for the input sequence. The upper bound for this limit is based on the
# pre-trained model's maximum input sequence length.
TOKEN_LIMIT = 512

# Top-p value used for controlling diversity in model's response.
TOP_P = 0.9

# Boolean flag to control sampling in model's response.
DO_SAMPLE = False

# Maximum length of the generated response.
MAX_LENGTH = 512
