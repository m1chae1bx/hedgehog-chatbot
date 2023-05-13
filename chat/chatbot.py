import os
import logging

from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM

from config import (
    MODEL_PATH,
    DOCSTORE_PATH,
    SENTENCE_EMBEDDING_MODEL_PATH,
    INSTRUCTION,
    SIMILARITY_SEARCH_K,
    TOP_K,
    TOKEN_LIMIT,
    TOP_P,
    DO_SAMPLE,
    MAX_LENGTH,
)
from database import sqlite_connection, fetch_article, fetch_article_split_text
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

LOG_LEVEL = os.environ.get("LOG_LEVEL", "ERROR").upper()

transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(LOG_LEVEL)

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


class Chatbot:
    def __init__(self, starting_dialog: str):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.model = TFAutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH, True)
        self.docstore = self._load_docstore()
        self.dialog = [starting_dialog]
        self.knowledge = []
        logger.info("Chatbot initialized.")

    def chat(self, user_input):
        """
        Generate a response based on user input and relevant knowledge.

        This method searches for relevant knowledge in the document store, generates a response
        using the retrieved knowledge and dialog history, and adds the response to the dialog
        history.

        Args:
            user_input (str): The user's input to generate a response.

        Returns:
            str: The generated response.
        """
        self._add_to_dialog(user_input)

        self._get_knowledge(user_input)
        response = self._generate_response()

        self._add_to_dialog(response)
        return response

    def _generate_response(self):
        """
        Generate a response based on the given instruction, knowledge, and dialog.

        Returns:
            str: The generated response.
        """
        print("Generating response...")
        query = self._form_query()

        token_length = len(self.tokenizer.encode(query))
        prev_token_length = None

        while token_length > TOKEN_LIMIT and token_length != prev_token_length:
            prev_token_length = token_length
            self.knowledge.pop(0)
            query = self._form_query()
            token_length = len(self.tokenizer.encode(query))

        print("query:", query)
        input_ids = self.tokenizer(query, return_tensors="pt", truncation=True).input_ids
        outputs = self.model.generate(
            input_ids, max_length=MAX_LENGTH, min_length=8, top_p=TOP_P, do_sample=DO_SAMPLE
        )
        output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return output

    def _form_query(self) -> str:
        """
        Form a query from the instruction, dialog and knowledge.

        Returns:
            str: The query.
        """
        synthesized_knowledge = self._synthesize_knowledge()
        dialog_serialized = " EOS ".join(self.dialog)
        return f"{INSTRUCTION} [CONTEXT] {dialog_serialized} [KNOWLEDGE] {synthesized_knowledge}"

    def _get_knowledge(self, user_input: str):
        """
        Retrieve relevant knowledge from the document store based on the user's input and stores
        it in the knowledge list.

        The method searches for similar documents in the document store using the user input as a
        query. It filters the results based on a similarity threshold and then uses the max marginal
        relevance search to find the most relevant documents.

        Args:
            user_input (str): The user's input to search for relevant knowledge.

        Returns:
            None
        """
        with sqlite_connection() as c:
            print("\nSearching for relevant articles...")
            mmr_results = self.docstore.max_marginal_relevance_search(
                user_input, k=TOP_K, fetch_k=SIMILARITY_SEARCH_K
            )
            print("mmr_results:", mmr_results)
            for document in mmr_results:
                article_id = document.metadata["article_id"]
                position = document.metadata["position"]

                existing_item = None
                for item in self.knowledge:
                    if item[0] == article_id and item[1] == position:
                        existing_item = item
                        break

                if existing_item:
                    self.knowledge.remove(existing_item)

                text = fetch_article_split_text(article_id, position, c)
                knowledge_item = (article_id, position, text)
                self.knowledge.append(knowledge_item)

    def _synthesize_knowledge(self) -> str:
        """
        Synthesize the knowledge from the retrieved articles.

        The method organizes the knowledge from the retrieved articles by their titles and text.
        It concatenates the information in a structured format, including the article title and
        text. The synthesized knowledge is then used as context for the model's response.

        Returns:
            str: A synthesized string containing the knowledge from the retrieved articles.
        """

        with sqlite_connection() as c:
            articles = {}
            article_ids = []
            for article_id, position, text in self.knowledge:
                if article_id not in articles:
                    articles[article_id] = []
                    article_ids.append(article_id)
                articles[article_id].append((position, text))
                articles[article_id].sort(key=lambda x: x[0])

            synthesized_knowledge = ""

            for article_id in article_ids:
                title, date_published = fetch_article(article_id, c)

                synthesized_knowledge += f"Title: {title}. "

                for _, text in articles[article_id]:
                    synthesized_knowledge += f"{text} "

            return synthesized_knowledge

    def _add_to_dialog(self, message):
        """
        Add a message to the dialog history.

        The method adds a message to the dialog history and removes the oldest message if the
        history exceeds a certain length.

        Args:
            message (str): The message to add to the dialog history.

        Returns:
            None
        """
        self.dialog.append(message)
        if len(self.dialog) > 5:
            self.dialog = self.dialog[1:]

    def _load_docstore(self) -> FAISS:
        """
        Load the document store from the specified file path.

        Returns:
            FAISS: An instance of the FAISS document store.
        """
        if not os.path.exists(DOCSTORE_PATH):
            logger.error("No docstore exists")
            exit()

        embeddings = HuggingFaceEmbeddings(model_name=SENTENCE_EMBEDDING_MODEL_PATH)
        docstore = None
        logger.info("Loading docstore...")
        docstore = FAISS.load_local(DOCSTORE_PATH, embeddings)

        if docstore is None:
            logger.error("No docstore exists")
            exit()

        return docstore
