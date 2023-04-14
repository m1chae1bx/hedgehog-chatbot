# Hedgehog: A Financial News Chatbot Powered by Transformers and Semantic Search

Hedgehog is a chatbot focused on answering questions about financial news. It uses a transformer-based model to generate responses by drawing from relevant news articles in its knowledge base. When a user asks a question, the chatbot searches for applicable news sources and crafts a response based on the information found.

## Motivation

Hedgehog is an experimental project aimed at evaluating the effectiveness of augmenting a semantic search-based information retrieval system with a smaller transformer model fine-tuned for answering user queries. The experiment seeks to determine if this combination can create a decent information retrieval chatbot despite the reduced size of the transformer model compared to bleeding-edge large language models like GPT-3 and GPT-4.

## Scope and Limitations

As mentioned previously, Hedgehog relies on a transformer-based model and a knowledge base of news articles to generate responses to user queries. Its primary focus is on handling financial news inquiries; however, the underlying architecture and design principles can be extended to support other domains in the future.

The knowledge base is currently limited to a small fixed set of financial news articles dating back to 2018, therefore, the chatbot is not capable of answering questions about recent events.

Other methods of information retrieval, such as keyword-based search exist but are currently not supported by Hedgehog. However, the underlying architecture and design principles can be extended to support other methods of information retrieval in the future.

It's important to note that Hedgehog is not a production-ready chatbot. It is an experimental project and not optimized for performance and is not designed to handle large amounts of traffic.

## How It Works

Hedgehog uses a two-step process to answer user questions: retrieving relevant news articles and generating a response based on the retrieved information.

### Retrieving Relevant News Articles

When a user submits a question, the chatbot first needs to find relevant news articles that may contain the answer. To achieve this, Hedgehog leverages a technique called semantic search. Semantic search is a method of information retrieval that uses a similarity metric to find documents that are semantically similar to a given query. In this case, the query is the user's question and the documents are the news articles.

In order to compare the similarity between the user's question and the news articles, Hedgehog needs a way to represent the semantic meaning of the question and the news articles. To do this, Hedgehog uses a sentence embedding model to compute a dense vector representation of the question and the news articles. The sentence embedding model used in this **project** is the sentence-transformers/all-MiniLM-L6-v2 model [2]. This model is a transformer-based model that can be used to compute sentence embeddings, mapping sentences and paragraphs to a 384 dimensional dense vector space Sentences with similar meanings should have similar embeddings.

In this project, Hedgehog uses the library FAISS to compute the similarity between the question's embedding and the embeddings of the news articles. FAISS is a library for efficient similarity search and clustering of dense vectors [3]. It is capable of performing similarity search on large datasets with high performance. FAISS provides many index types that can be used to compute the similarity between two embeddings. Since we are using a wrapper library called Langchain to interface with FAISS, we are limited to the default index type that is used by Langchain - IndexFlatL2 [6]. IndexFlatL2 is a brute-force search index that computes the L2 distance between two embeddings [4]. The L2 distance is the Euclidean distance between two embeddings [5]. The smaller the distance, the more similar the embeddings are.

To compute similarity between embeddings, we need to generate and store them in a FAISS index. However, news articles are usually lengthy, so we need to split them into smaller parts before generating embeddings. As the paragraphs in each of the article are already separated by newlines, we use the newlines to split the articles into chunks. We add the title at the beginning of each of the chunk to maintain context.  Although some paragraphs may exceed the maximum sequence length of the sentence embedding model, resulting in truncated text and loss of context, we accepted this limitation for our project. In the future, other methods of splitting the articles into smaller chunks may be explored, such as using overlaps between the chunks. The maximum sequence length for the sentence embedding model used in this project is 256 word pieces or tokens [2]. Each embedding is stored in the FAISS index along with some metadata.

Every time a user asks a question, Hedgehog first computes the embedding of the question using the same sentence embedding model. It then uses the FAISS index to retrieve the most similar articles to the question's embedding. ...



<!--
According to https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2, the maximum sequence length for this model is 256 word pieces or tokens.
 -->

Then, it calculates the similarity between the question and each article's embeddings. The chatbot selects the most relevant articles based on the similarity scores.

A sentence embedding is a dense vector that represents the semantic meaning of a sentence or paragraph. To compare the similarity between two sentences, we can simply compute


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

### Install Python Dependencies

Before running the chatbot, you need to install the dependencies by running `pip install -r requirements-<os>.txt` on your preferred Python environment.

### Download Models from Hugging Face

Make sure to download the models before running the chatbot. To download the models, you can run the scripts in the `scripts/models` directory. This will download the models to the `models` directory.

## Running the Chatbot

Before running the chatbot, make sure you have a capable machine when running the chatbot, as the model is quite large. The `microsoft/GODEL-v1_1-large-seq2seq` model alone is at least 3 GB in size.

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
2. What are some of the challenges that cyber security startups face?
3. How much did cyber security startup ForeScout raise in an IPO in October?

Reference Article: Race car driver Tucker gets more than 16 years for lending scheme
1. What happened to race car driver Scott Tucker in 2018?
2. What crime did he commit?
3. Who else was convicted along with Tucker?
4. Where has Tucker competed in the past as a race car driver?

Reference Article: Amazon's latest patents are focused on video, augmented reality and the connected home
1. What are the three main areas Amazon is focusing on in its recent patent activity, according to MCAM analysis?
2. What is the purpose of Amazon's augmented reality mirror patent?
3. How did early cloud computing patents help Amazon in the market?
4. What privacy feature is offered in Amazon's Echo devices patent for video conferencing?

## Sample Conversations

Here are some sample conversations with the chatbot based on the sample questions above:

Reference Article: Japan's Abe urges central bank's Kuroda to keep up efforts on economy
```
Hedgehog: Hello, I'm here to answer your questions regarding financial news.
User: Has Abe decided whether to reappoint Kuroda for another five-year term?
Hedgehog: No, he hasn't made up his mind.
User: Is there a possibility that Kuroda will be reappointed?
Hedgehog: Many analysts see a good chance that Kuroda will be reappointed when the government selects successors to him and his two deputies in coming months, decisions that need parliament's approval.
User: What was the goal of the massive stimulus program launched by Kuroda in 2013?
Hedgehog: Boost growth but failed to drive up inflation to the bank's target of 2 percent.
User: What factors do the Japanese government consider when determining whether the economy is out of deflation?
Hedgehog: The government looks at several factors, such as inflation data and the output gap, in determining whether the economy is sustainably out of deflation.
```

Reference Article: Under threat: Cyber security startups fall on harder times
```
Hedgehog: Hello, I'm here to answer your questions regarding financial news.
User: What was David Cowan's comment regarding the cyber security startups in 2018?
Hedgehog: “I have never seen such a fast-growing market with so many companies on the losing side,” said David Cowan, a partner at Bessemer Venture Partners, a venture capital firm that has invested in the cyber security sector.
User: What are some of the challenges that cyber security startups face?
Hedgehog: The result is a number of these start-ups have become corporate “zombies” with little prospect of fetching a good price in an initial public offering (IPO) or becoming acquisition targets, experts said.
User: How much did cyber security startup ForeScout raise in an IPO in October?
Hedgehog: $116 million in an IPO in October that valued the company at about $800 million,
```

Reference Article: Race car driver Tucker gets more than 16 years for lending scheme
```
Hedgehog: Hello, I'm here to answer your questions regarding financial news.
User: What happened to race car driver Scott Tucker in 2018?
Hedgehog: He was sentenced by U.S. District Judge Kevin Castel in Manhattan.
User: What crime did he commit?
Hedgehog: violating federal truth in lending and racketeering laws.
User: Who else was convicted along with Tucker?
Hedgehog: Timothy Muir, 46, a lawyer who worked with Tucker and was convicted of the same charges in October, to seven years in prison.
User: Where has Tucker competed in the past as a race car driver?
Hedgehog: U.S. and European racing circuits,
```

Reference Article: Amazon's latest patents are focused on video, augmented reality and the connected home
```
Hedgehog: Hello, I'm here to answer your questions regarding financial news.
User: What are the three main areas Amazon is focusing on in its recent patent activity, according to MCAM analysis?
Hedgehog: Video, augmented reality and the connected home.
User: What is the purpose of Amazon's augmented reality mirror patent?
Hedgehog: It allows users to try on virtual clothes in a virtual setting, using a system of cameras, mirrors, projectors and lights to project different backgrounds and clothes onto the user.
User: How did early cloud computing patents help Amazon in the market?
Hedgehog: Amazon was getting patents issued relating to advanced cloud management and deployment.
User: What privacy feature is offered in Amazon's Echo devices patent for video conferencing?
Hedgehog: The technology allows users to use video conferencing on their Echo and mute it by blurring or pixelating the video for user privacy.
```

<!-- Possible new article: Philippine lawmakers to press for inquiry into deadly mall fire -->
How many people died in a shopping mall fire in the Philippines in the city of Davao last 2018?
38 people died in the fire, which injured six people, although about 100 call center employees survived.
Which call center company was affected in the fire?
Research Now SSI, on the fourth floor of the same building in choking smoke.

## References

[1] https://huggingface.co/microsoft/GODEL-v1_1-large-seq2seq
[2] https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
[3] https://github.com/facebookresearch/faiss/wiki
[4] https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
[5] https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances
[6] https://python.langchain.com/en/latest/_modules/langchain/vectorstores/faiss.html
