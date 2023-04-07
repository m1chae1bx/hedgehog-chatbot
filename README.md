# Hedgehog: A Transformer-Based Chatbot With Financial Information Retrieval

Hedgehog is a chatbot focused on answering questions about financial news. It uses a transformer-based model to generate responses by drawing from relevant news articles in its knowledge base. When a user asks a question, the chatbot searches for applicable news sources and crafts a response based on the information found.

## Scope

The scope of the Hedgehog project is to provide users with accurate and relevant answers to their financial news questions. The chatbot relies on its transformer-based model and a knowledge base of news articles to generate responses. The main focus is on handling financial news inquiries; however, the underlying architecture and design principles can be extended to support other domains in the future.

Please note that Hedgehog has some limitations. It can only provide information that is available in its knowledge base, so for up-to-date information, the knowledge base needs to be updated regularly.

Here is a sample conversation between the chatbot and a user:

```
Hedgehog: Hello, I'm here to answer your questions regarding financial news.
User: Has Abe decided whether to reappoint Kuroda for another five-year term?
Hedgehog: No, he hasn't made up his mind.
User: Is there a possibility that Kuroda will be reappointed?
Hedgehog: Many analysts see a good chance that Kuroda will be reappointed when the government selects successors to him and his two deputies in coming months, decisions that need parliament's approval.
```

## How It Works

Hedgehog uses a two-step process to answer user questions: retrieving relevant news articles and generating a response based on the retrieved information.

Retrieving Relevant News Articles: When a user submits a question, the chatbot first needs to find relevant news articles that may contain the answer. To achieve this, Hedgehog leverages the sentence-transformers/all-MiniLM-L6-v2 model to compute embeddings for both the user's question and the available news articles. Then, it calculates the similarity between the question and each article's embeddings. The chatbot selects the most relevant articles based on the similarity scores.

Generating a Response: Once the relevant articles are identified, Hedgehog uses the microsoft/GODEL-v1_1-large-seq2seq model to generate a response. This model is an encoder-decoder architecture specifically designed to ground responses in external information. The chatbot feeds the user's question and the retrieved articles into the model, which then generates an answer based on the provided context.

It is important to note that the quality of the response depends on the relevance and accuracy of the retrieved articles, as well as the performance of the models used.

## Data

Financial news articles were downloaded from https://www.kaggle.com/datasets/jeet2016/us-financial-news-articles. The zip archive needs to be extracted to the data/raw directory before running the preprocessing scripts. Only a partial of the data has been preprocessed and included in the repository. The news articles used in this project were published in 2018.

## Preprocessing

The preprocessing step is responsible for extracting the data from the raw data files and storing it in a format that is more suitable for the chat application. The preprocessing step is also responsible for creating the necessary indexes for the chat application to function properly. Preprocessing has already been done for the data in the `data/processed` directory. If you want to re-run the preprocessing step, you can do so by running the scripts in the `scripts/preprocessing` directory.

## Models

The models directory contains models used for the chatbot and for the similarity search.
These models were downloaded from Huggingface. The `microsoft/GODEL-v1_1-large-seq2seq` model is a transformer-based encoder-decoder model that can generate responses grounded on external information in a conversation [1]. In this case, the external information is a set of news articles.

The `sentence-transformers/all-MiniLM-L6-v2` model is a transformer-based model that can be used to compute sentence embeddings, mapping sentences and paragraphs to a 384 dimensional dense vector space [2]. These sentence embeddings are used to compute the similarity between the user's input and the news articles.

## Setup

Before running the chatbot, you need to install the dependencies by running `pip install -r requirements.txt` on your preferred Python environment. In addition, make sure you have a capable machine when running the chatbot, as the model is quite large. The `microsoft/GODEL-v1_1-large-seq2seq` model alone is at least 3 GB in size.

Make sure to download the models before running the chatbot. To download the models, you can run the scripts in the `scripts/models` directory.

## Running the Chatbot

You can run the chatbot by running `python chat/main.py` from the root directory.

## Sample Questions

Here are some sample questions that you can ask the chatbot:

Reference Article: Japan's Abe urges central bank's Kuroda to keep up efforts on economy
1. Has Abe decided whether to reappoint Kuroda for another five-year term?
2. Is there a possibility that Kuroda will be reappointed?
3. What was the goal of the massive stimulus program launched by Kuroda in 2013?
4. What factors do the Japanese government consider when determining whether the economy is out of deflation?

Reference Article: Under threat: Cyber security startups fall on harder times
1. What was David Cowan's comment regarding the cyber security startups in 2018?
2. What are some of the challenges the cyber security startups face?
3. How much did cyber security startup ForeScout raise in an IPO in October?

## References

[1] https://huggingface.co/microsoft/GODEL-v1_1-large-seq2seq
[2] https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
