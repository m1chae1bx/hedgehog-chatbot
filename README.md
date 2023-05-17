# Hedgehog: A Financial News Chatbot Powered by Transformers and Semantic Search

Hedgehog is a chatbot focused on answering questions about financial news. It uses a transformer-based model to generate responses by drawing from relevant news articles in its knowledge base. When a user asks a question, the chatbot searches for relevant news articles and crafts a response based on the information found.

## Motivation

Hedgehog is an experimental project aimed at evaluating the effectiveness of augmenting a semantic search-based information retrieval system with a smaller transformer model pre-trained for answering user queries. The experiment aims to evaluate if this combination can create a decent information retrieval chatbot despite the reduced size of the transformer model compared to bleeding-edge large language transformer models like GPT-3 and GPT-4.

Disclaimer:

This document discusses the development and assessment of the Hedgehog chatbot, an exercise that should not be seen as a comprehensive or strictly scientific experiment. The methodologies utilized, while designed to provide insights into the chatbot's performance, do not strictly adhere to the standard processes and stringent criteria usually associated with formal scientific experimentation. Consequently, the results and conclusions drawn from this project should be viewed within this context. The ultimate aim of this project is not to draw definitive conclusions about the chatbot's overall capabilities, but rather to provide a basis for potential enhancements and to stimulate further exploratory analysis. It's important to understand that this project did not fully conform to the typical procedures associated with scientific experiments.

## Scope and Limitations

As mentioned previously, Hedgehog relies on a transformer-based model and a knowledge base of news articles to generate responses to user queries. It's important to note that the transformer-based model used, specifically the sentence-transformers/all-MiniLM-L6-v2 model, has been utilized as is, without any fine-tuning tailored to this specific task.

The primary focus of Hedgehog is on handling financial news inquiries. However, the underlying architecture and design principles can be extended to support other domains in the future.

The knowledge base is currently limited to a small fixed set of financial news articles dating back to 2018. As such, the chatbot may not be capable of answering questions about very recent events or those that require information beyond the scope of the available articles.

While other methods of information retrieval, such as keyword-based search, exist, they are currently not supported by Hedgehog. However, the flexibility of the underlying architecture and design principles allows for potential future extensions to support other methods of information retrieval.

Lastly, it's important to understand that Hedgehog is not a production-ready chatbot. It is an experimental project and not optimized for performance. It is not designed to handle large amounts of traffic or provide real-time responses on a large scale.

## How It Works

Hedgehog uses a two-step process to answer user questions: retrieving relevant news articles and generating a response based on the retrieved information.

### Retrieving Relevant News Articles

When a user submits a question, the chatbot first needs to find relevant news articles that may contain the answer. To achieve this, Hedgehog leverages a technique called semantic search. Semantic search is a method of information retrieval that uses a similarity metric to find documents that are semantically similar to a given query. In this case, the query is the user's question and the documents are the news articles.

#### Sentence Embeddings

To compare the semantic similarity between the user's question and the news articles, Hedgehog utilizes sentence embeddings. Sentence embeddings are vector representations that encapsulate the semantic meaning of sentences. For this project, we employ the sentence-transformers/all-MiniLM-L6-v2 model [1] to compute these embeddings. This model, based on the transformer architecture, computes dense vector representations (embeddings) for sentences and paragraphs, mapping them to a 384-dimensional vector space. Consequently, sentences with similar meanings are expected to have proximate embeddings in this high-dimensional space.

In this project, Hedgehog employs the FAISS library to compute the similarity between the embedding of a user's question and the embeddings of the news articles. FAISS is a library specifically designed for efficient similarity search and clustering of dense vectors [2]. It stands out for its high performance, particularly when dealing with large datasets.

There are various index types offered by FAISS to calculate the similarity between two embeddings. In our case, since we are interfacing with FAISS via a wrapper library from LangChain, we default to using IndexFlatL2 [3]. IndexFlatL2 conducts a brute-force search, calculating the L2 (or Euclidean) distance between two embeddings [4][5]. The principle is simple: the smaller the distance, the higher the similarity between the embeddings.

However, Hedgehog's use of FAISS is not confined to a simple similarity search. Instead, we leverage the maximal marginal relevance algorithm that LangChain provides on top of FAISS. This algorithm integrates both Euclidean and Cosine similarity measures to achieve more nuanced search results. More in-depth information on this process can be found in the "Searching the FAISS Index" section.

Langchain, the framework being utilized here, is tailored for developing applications powered by language models. The framework is designed to enable applications to be data-aware and agentic, allowing language models to interact with their environment. Langchain provides modular abstractions for the components necessary to work with language models, and collections of implementations for all these abstractions. The components are designed to be easy to use, regardless of whether you are using the rest of the LangChain framework. Langchain provides an interface to interact with several Large Language Models (LLMs), Chat Models, and Text Embedding Models [6].

#### Preprocessing News Articles

To compute similarity between embeddings, we need to generate and store them in a FAISS index. However, news articles are usually lengthy, so we need to split them into smaller parts before generating embeddings. As the paragraphs in each of the article are already separated by newlines, we use the newlines to split the articles into chunks. We add the title at the beginning of each of the chunk to maintain context.  Although some paragraphs may exceed the maximum sequence length of the sentence embedding model, resulting in truncated text and loss of context, we accepted this limitation for our project. The maximum sequence length for the sentence embedding model used in this project is 256 word pieces or tokens [1]. Each embedding is stored in the FAISS index along with some metadata.

#### Searching the FAISS Index

When a user asks a question, Hedgehog first computes the embedding of the question using the same sentence embedding model used in the preprocessing stage. It then uses the FAISS index to retrieve the most similar articles to the question's embedding. However, instead of performing a straightforward similarity search with FAISS, the maximal marginal relevance algorithm that Langchain provides on top of FAISS is used. Maximal marginal relevance (MMR) optimizes for similarity to query and diversity among selected documents [3]. MMR was used due to duplicate articles existing in the dataset. As an example, for the topic of Abe's sentiment regarding central bank governor Haruhiko Kuroda, there exist two similar articles from Reuters and one from Bloomberg. This approach is important to ensure that the limited input size of the transformer model is used efficiently when generating a response as the results from the semantic search will form part of the model input.

How does maximal marginal relevance work? First, a set of candidate chunks are retrieved from the FAISS index based on L2 or Euclidean distance. Then, the candidate chunks are iteratively selected based on their cosine similarity to the input while penalizing chunks that are too similar to previously selected chunks. This ensures that the selected chunks are diverse and relevant to the input. Once the final set of chunks are selected, they are added to the chatbot's current knowledge list.

### Generating a Response

Once the chatbot's knowledge list is updated with the relevant articles chunks, Hedgehog synthesizes a query that will be fed to the
the microsoft/GODEL-v1_1-large-seq2seq model to generate a response. This model is an encoder-decoder architecture specifically trained to ground responses in external information provided in its input. Following the suggested prompt from the model's official Hugging Face model card, the query is constructed as follows:

`<instruction> [CONTEXT] <dialog> [KNOWLEDGE] <knowledge>`

Where:
- `<instruction>` is a pre-configured instruction to the model to generate a response
- `[CONTEXT]` is a special token that indicates the start of the context
- `<dialog>` is the history of the conversation between the user and the chatbot (limited to 5 recent messages). Messages are separated by the EOS token.
- `[KNOWLEDGE]` is a special token that indicates the start of the knowledge list
- `<knowledge>` is a synthesized version of the knowledge list containing the most relevant chunks to the user's query

Here is a sample query to be fed to the model:

```
Instruction: given a dialog context and related knowledge, you need to response safely based on the knowledge. [CONTEXT] Hello, I'm here to answer your questions regarding financial news. EOS Has Abe decided whether to reappoint Kuroda for another five-year term? [KNOWLEDGE] Title: Japan's Abe tells Kuroda to keep up efforts on economy. Kuroda was handpicked by Abe to take the helm of the central bank in 2013 to deploy a massive stimulus program - part of the premier's "Abenomics" reflationary policies - that helped boost growth but failed to drive up inflation to the bank's target of 2 percent. "I want him to keep up his efforts. But I haven't made up my mind," on who should succeed Kuroda when his term ends in April, he added. Many analysts see a good chance that Kuroda will be reappointed when the government selects successors to him and his two deputies in coming months, decisions that need parliament's approval.
```

The model has a maximum input length of 512 tokens, thus, if the query exceeds this limit, the knowledge list will be truncated. This is a limitation of the model due to the limited input size. In the future, it may be possible to use a larger model that can handle longer inputs. Once the response is generated, it is returned to the user, added to the dialog history, and the user can then proceed to ask another question, thus repeating the process.

It is important to note that the quality of the response depends on the relevance and accuracy of the retrieved articles, as well as the performance of the models used.

## Data

Financial news articles were downloaded from https://www.kaggle.com/datasets/jeet2016/us-financial-news-articles. The zip archive needs to be extracted to the data/raw directory before running the preprocessing scripts. Only a partial of the data has been preprocessed and included in the repository. The news articles used in this project were published in 2018.

## Preprocessing

The preprocessing step is responsible for extracting the data from the raw data files and storing it in a format that is more suitable for the chat application. The preprocessing step is also responsible for creating the necessary indexes for the chat application to function properly. The data in the data/processed directory has already been preprocessed. If you want to re-run the preprocessing step, you can do so by running the scripts in the `scripts/preprocessing` directory.

## Models

The models directory contains models used for the chatbot and for the similarity search.
These models were downloaded from Huggingface. The `microsoft/GODEL-v1_1-large-seq2seq` model is a transformer-based encoder-decoder model that can generate responses grounded on external information in a conversation [7]. In this case, the external information is a set of news articles.

The `sentence-transformers/all-MiniLM-L6-v2` model is a transformer-based model that can be used to compute sentence embeddings, mapping sentences and paragraphs to a 384 dimensional dense vector space [1]. These sentence embeddings are used to compute the similarity between the user's input and the news articles.

## Setup

### Install Python Dependencies

Before running the chatbot, you need to install the dependencies by running `pip install -r requirements-<os>.txt` on your preferred Python environment.

### Download Models from Hugging Face

Make sure to download the models before running the chatbot. To download the models, you can run the scripts in the `scripts/models` directory. This will download the models to the `models` directory.

## Running the Chatbot

Before running the chatbot, make sure you have a capable machine when running the chatbot, as the model is quite large. The `microsoft/GODEL-v1_1-large-seq2seq` model alone is at least 3 GB in size.

You can run the chatbot by running `python chat/main.py` from the root directory.

## Sample Conversations

Below are some sample conversations with the chatbot based on the sample questions above.

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

# Evaluation of Hedgehog Chatbot

Disclaimer: This evaluation of the Hedgehog chatbot has been conducted on a qualitative basis, which involves subjective analysis of the chatbot's responses. While it provides insights into the chatbot's ability to generate coherent and contextually relevant replies, it lacks the specificity and broad statistical backing of quantitative methods. The results should be interpreted with this in mind, and this document does not claim to offer a comprehensive assessment of the chatbot's capabilities. Further evaluations utilizing quantitative methodologies may be needed for a more detailed understanding of the chatbot's performance.

## Evaluation Questions

Here are the questions that were used to evaluate the performance of the chatbot:

Reference Article: Japan's Abe urges central bank's Kuroda to keep up efforts on economy
1. Has Abe decided whether to reappoint Kuroda for another five-year term?
2. Is there a possibility that Kuroda will be reappointed?
3. What was the goal of the massive stimulus program launched by Kuroda in 2013?
4. What factors do the Japanese government consider when determining whether the economy is out of deflation?

Reference Article: Under threat: Cyber security startups fall on harder times
1. What was David Cowan's comment regarding the cyber security startups in 2018?
2. What happened to the security products these startups have developed?
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

## Comparison with GPT-3.5-Turbo Model and the Reference Article

Based from the questions above, the responses from Hedgehog were compared with the responses from a large language model from OpenAI, GPT-3.5-Turbo, and a snippet from the reference article containing the answer. Please note that each of the topics is a separate conversation. Asking multiple topics in one conversation might result in a different response from Hedgehog. This is a limitation of the current implementation of the chatbot. This will be further discussed later.

Each of the sample conversations are compared with a response from OpenAI's gpt-3.5-turbo model, and a snippet from the original article that contains the answer to the question.

### Model Configuration

The same query given to the GODEL model is given to the OpenAI gpt-3.5-turbo model except that it is formatted to utilize the fields provided in the OpenAI playground. The OpenAI playground (https://platform.openai.com/playground?mode=chat) is a tool that allows users to interact with the GPT-3.5-Turbo model from the browser.

An example query sent to GODEL to is shown below:
```
Instruction: given a dialog context and related knowledge, you need to response safely based on the knowledge. [CONTEXT] Hello, I'm here to answer your questions regarding financial news. EOS Has Abe decided whether to reappoint Kuroda for another five-year term? [KNOWLEDGE] Title: Japan's Abe tells Kuroda to keep up efforts on economy. Kuroda was handpicked by Abe to take the helm of the central bank in 2013 to deploy a massive stimulus program - part of the premier's "Abenomics" reflationary policies - that helped boost growth but failed to drive up inflation to the bank's target of 2 percent. "I want him to keep up his efforts. But I haven't made up my mind," on who should succeed Kuroda when his term ends in April, he added. Many analysts see a good chance that Kuroda will be reappointed when the government selects successors to him and his two deputies in coming months, decisions that need parliament's approval.
```

A similar query is sent to the OpenAI gpt-3.5-turbo model adapted to the inputs required and should look like this:

System:
```
Instruction: given a dialog context and related knowledge, you need to response safely based on the knowledge. [KNOWLEDGE] Title: Japan's Abe tells Kuroda to keep up efforts on economy. Kuroda was handpicked by Abe to take the helm of the central bank in 2013 to deploy a massive stimulus program - part of the premier's "Abenomics" reflationary policies - that helped boost growth but failed to drive up inflation to the bank's target of 2 percent. "I want him to keep up his efforts. But I haven't made up my mind," on who should succeed Kuroda when his term ends in April, he added. Many analysts see a good chance that Kuroda will be reappointed when the government selects successors to him and his two deputies in coming months, decisions that need parliament's approval.
```

Assistant:
```
Hello, I'm here to answer your questions regarding financial news.
```

User:
```
Has Abe decided whether to reappoint Kuroda for another five-year term?
```

The OpenAI Playground already manages the dialog history, so we only need to update the system message with the updated knowledge every time a new question is asked.

GODEL has the following parameters:
- TOP_P = 0.9
- DO_SAMPLE = False
- max_length = 512

GPT-3.5-Turbo has the following parameters:
- Top P = 0.9
- Temperature = 0.2
- Maximum Length = 512
- Frequency Penalty = 0
- Presence Penalty = 0

### Results and Observations

#### Reference Article: Japan's Abe urges central bank's Kuroda to keep up efforts on economy

| Question | Hedgehog | GPT-3.5-Turbo | Reference Article |
| --- | --- | --- | --- |
| Has Abe decided whether to reappoint Kuroda for another five-year term? | No, he hasn't made up his mind. | According to the recent news, Abe has not made up his mind yet on who should succeed Kuroda when his term ends in April. However, many analysts see a good chance that Kuroda will be reappointed when the government selects successors to him and his two deputies in the coming months, decisions that need parliament's approval. | Japanese Prime Minister Shinzo Abe on Sunday called on central bank governor Haruhiko Kuroda to keep up efforts to reflate the economy, but added he was undecided on whether to reappoint Kuroda for another five-year term. ... "I want him to keep up his efforts. But I haven't made up my mind," on who should succeed Kuroda when his term ends in April, he added.  |
| Is there a possibility that Kuroda will be reappointed? | Many analysts see a good chance that Kuroda will be reappointed when the government selects successors to him and his two deputies in coming months, decisions that need parliament's approval. | Yes, there is a possibility that Kuroda will be reappointed as the governor of the Bank of Japan. Many analysts see a good chance that he will be reappointed when the government selects successors to him and his two deputies in the coming months, decisions that need parliament's approval. | Many analysts see a good chance that Kuroda will be reappointed when the government selects successors to him and his two deputies in coming months, decisions that need parliament's approval. |
| What was the goal of the massive stimulus program launched by Kuroda in 2013? | Boost growth but failed to drive up inflation to the bank's target of 2 percent. | The massive stimulus program launched by Kuroda in 2013 was part of the premier's "Abenomics" reflationary policies, which aimed to boost growth and drive up inflation to the Bank of Japan's target of 2 percent. The stimulus program involved deploying a range of monetary policy tools, including negative interest rates and massive purchases of government bonds and other assets, in order to stimulate the economy and encourage lending and investment. While the program helped boost growth, it failed to drive up inflation to the bank's target. | Kuroda was handpicked by Abe to take the helm of the central bank in 2013 to deploy a massive stimulus program - part of the premier's "Abenomics" reflationary policies - that helped boost growth but failed to drive up inflation to the bank's target of 2 percent. |
| What factors do the Japanese government consider when determining whether the economy is out of deflation? | The government looks at several factors, such as inflation data and the output gap, in determining whether the economy is sustainably out of deflation. | The Japanese government considers several factors when determining whether the economy is out of deflation. These factors include inflation data, the output gap, and other economic indicators such as employment and consumer spending. The government looks for sustained improvement in these indicators over time to determine whether the economy is sustainably out of deflation. Policymakers are keen to declare an end to deflation, which would help the government argue that its policies have succeeded in reflating the economy. However, some analysts caution that prematurely declaring an end to deflation could give the Bank of Japan justification to withdraw crisis-mode stimulus. | The government looks at several factors, such as inflation data and the output gap, in determining whether the economy is sustainably out of deflation. |

For this reference article, both models were able to answer the questions correctly. While GPT-3.5-Turbo was able to provide more details, some of the details were made up and were not present from the article. GPT-3.5-Turbo also tends to add other information that were not really asked by the user. Hedgehog, on the other hand, was able to provide more concise answers that were directly from the article. While it's generally beneficial for a language model to provide a comprehensive answer, it's essential to ensure the accuracy of the information provided, especially when the model is explicitly instructed to base its response on a specific piece of knowledge or article.

#### Reference Article: Under threat: Cyber security startups fall on harder times

| Question | Hedgehog | GPT-3.5-Turbo | Reference Article |
| --- | --- | --- | --- |
| What was David Cowan's comment regarding the cyber security startups in 2018? | "I have never seen such a fast-growing market with so many companies on the losing side," | David Cowan, a partner at Bessemer Venture Partners, commented that he had never seen such a fast-growing market with so many companies on the losing side in the cyber security sector. | "I have never seen such a fast-growing market with so many companies on the losing side," said David Cowan, a partner at Bessemer Venture Partners, a venture capital firm that has invested in the cyber security sector. |
| What happened to the security products these startups have developed? | have been overtaken by advances in cyber hacking, | According to industry executives and venture capitalists, in some cases, the security products developed by these startups have been overtaken by advances in cyber hacking. In others, larger competitors have come out with similar technology and locked down customers. | In some cases, the security products they developed have been overtaken by advances in cyber hacking, according to industry executives and venture capitalists. |
| How much did cyber security startup ForeScout raise in an IPO in October? | $116 million in an IPO in October that valued the company at about $800 million, | ForeScout raised $116 million in an IPO in October that valued the company at about $800 million, down from its $1 billion valuation in the private markets a year earlier. | ForeScout raised $116 million in an IPO in October that valued the company at about $800 million, down from its $1 billion valuation in the private markets a year earlier. |

Both Hedgehog and GPT-3.5-Turbo were able to accurately provide David Cowan's comment regarding the cyber security startups in 2018. However, Hedgehog looks incomplete with the dangling comma at the end.

For the second question, both were able to provide the correct answer, but GPT-3.5-Turbo added additional information from the article that was not asked.

For the third question, both were able to provide the correct answer, but Hedgehog again has a dangling comma at the end.

#### Reference Article: Race car driver Tucker gets more than 16 years for lending scheme

| Question | Hedgehog | GPT-3.5-Turbo | Reference Article |
| --- | --- | --- | --- |
| What happened to race car driver Scott Tucker in 2018? | He was sentenced by U.S. District Judge Kevin Castel in Manhattan. | In 2018, race car driver Scott Tucker was sentenced to more than 16 years in prison by U.S. District Judge Kevin Castel in Manhattan. He was found guilty by a jury in October of violating federal truth in lending and racketeering laws. Timothy Muir, a lawyer who worked with Tucker and was convicted of the same charges in October, was also sentenced to seven years in prison. | A Kansas businessman and race car driver was sentenced to 16 years and 8 months in prison on Friday for crimes related to his online payday lending business, which prosecutors said made more $3.5 billion as it exploited millions of cash-strapped consumers. \nScott Tucker, 55, was sentenced by U.S. District Judge Kevin Castel in Manhattan. He was found guilty by a jury in October of violating federal truth in lending and racketeering laws. \nTucker did not make a statement at his sentencing, referring instead to a letter he submitted to the court in December. |
| What crime did he commit? | violating federal truth in lending and racketeering laws. | Scott Tucker was found guilty of violating federal truth in lending and racketeering laws. He was involved in a lending scheme that charged illegally high-interest rates to vulnerable borrowers. The scheme generated hundreds of millions of dollars in profits for Tucker and his associates. | He was found guilty by a jury in October of violating federal truth in lending and racketeering laws. |
| Who else was convicted along with Tucker? | Timothy Muir, 46, | Timothy Muir, a lawyer who worked with Tucker, was also convicted of the same charges in October and was sentenced to seven years in prison. | Castel also sentenced Timothy Muir, 46, a lawyer who worked with Tucker and was convicted of the same charges in October, to seven years in prison. |
| Where has Tucker competed in the past as a race car driver? | U.S. and European racing circuits, | Scott Tucker has competed on U.S. and European racing circuits in the past. | Tucker, who has competed on U.S. and European racing circuits, was previously convicted of making a false statement to obtain a bank loan and of running a fraudulent financial services business. |

For this article, both models were able to answer the questions correctly. However, GPT-3.5-Turbo was able to provide more details particularly for the first question. It is debatable whether the additional details are necessary for the user's query. However, Hedgehog's answer feels like very incomplete. Knowing what he was sentenced rather than who sentenced him felt more important to be mentioned.

For the second question, Hedgehog was able to answer although the formatting of the answer could be improved. On the other hand, GPT-3.5-Turbo was able to provide a more complete answer but also added made-up information that were not present from the results of the semantic search performed.

For the third question, Hedgehog's answer is very concise and direct to the point while GPT-3.5-Turbo's answer is more detailed and was able to capture all the details from the article.

For the last question, both models were able to answer correctly. However, Hedgehog's answer can be improved as a complete sentence.

#### Reference Article: Amazon's latest patents are focused on video, augmented reality and the connected home

| Question | Hedgehog | GPT-3.5-Turbo | Reference Article |
| --- | --- | --- | --- |
| What are the three main areas Amazon is focusing on in its recent patent activity, according to MCAM analysis? | Video, augmented reality and the connected home. | According to MCAM analysis, Amazon is ramping up patent activity in video, connected home, and AR technologies. | MCAM analysis reveals that Amazon is ramping up patent activity in video, connected home and AR technologies. |
| What is the purpose of Amazon's augmented reality mirror patent? | It allows users to try on virtual clothes in a virtual setting, using a system of cameras, mirrors, projectors and lights to project different backgrounds and clothes onto the user. | The purpose of Amazon's augmented reality mirror patent is to allow users to try on virtual clothes in a virtual setting, using a system of cameras, mirrors, projectors, and lights to project different backgrounds and clothes onto the user. | As Amazon moves deeper into the fashion world, an augmented reality mirror would bring that world into customers' homes. It allows users to try on virtual clothes in a virtual setting, using a system of cameras, mirrors, projectors and lights to project different backgrounds and clothes onto the user. |
| How did early cloud computing patents help Amazon in the market? | Amazon was getting patents issued relating to advanced cloud management and deployment. | Early cloud computing patents helped Amazon in the market by giving them a competitive advantage. Amazon was able to establish themselves as a leader in cloud management and deployment before their competitors, such as Google and Microsoft, entered the space. This allowed Amazon to have a huge lead in cloud and made it more complex and costly for competitors to stand up their cloud capability. As a result, Amazon's cloud unit, AWS, is now the company's most profitable business. | For example, long before cloud computing was a mainstream enterprise practice -- and long before Google or Microsoft entered the space -- Amazon was getting patents issued relating to advanced cloud management and deployment. \n"That strategy very much empowered their business strategy to allow them to have a huge lead in cloud," says David Pratt, Managing Director at MCAM. "In turn," he says, "it made it more complex and costly for competitors to stand up their cloud capability." |
| What privacy feature is offered in Amazon's Echo devices patent for video conferencing? | The technology allows users to use video conferencing on their Echo and mute it by blurring or pixelating the video for user privacy. | Amazon's Echo devices patent for video conferencing offers a privacy feature that allows users to mute the video by blurring or pixelating it, thus protecting the user's privacy during video communication. | Another notable patent issued in the last month could advance video technology in Amazon's Echo devices. The technology allows users to use video conferencing on their Echo and mute it by blurring or pixelating the video for user privacy. |

When asked about the three main areas Amazon is focusing on in its recent patent activity according to MCAM analysis, both Hedgehog and GPT-3.5-Turbo provided accurate answers. They mentioned video, augmented reality, and the connected home, which is consistent with the information provided in the reference article. Neither model added extra information not found in the reference article.

For the question about the purpose of Amazon's augmented reality mirror patent, both Hedgehog and GPT-3.5-Turbo provided correct and similar answers. They described the technology as allowing users to try on virtual clothes in a virtual setting using cameras, mirrors, projectors, and lights. This information is consistent with the reference article, and neither model added extra information not found in the article.

In answering how early cloud computing patents helped Amazon in the market, GPT-3.5-Turbo provided a more comprehensive response than Hedgehog. GPT-3.5-Turbo explained that the patents gave Amazon a competitive advantage, allowing them to establish themselves as a leader in cloud management and deployment before competitors like Google and Microsoft entered the space. However, it added information not found in the reference article, such as the fact that Amazon's cloud unit, AWS, is now the company's most profitable business. Hedgehog's response was shorter and didn't mention the competitive advantage or AWS's profitability which is what the question was asking about.

When asked about the privacy feature offered in Amazon's Echo devices patent for video conferencing, both Hedgehog and GPT-3.5-Turbo accurately described the feature. They mentioned that the technology allows users to mute the video by blurring or pixelating it for user privacy during video communication. Both models' answers were consistent with the reference article, and no additional information was added.

#### Changing the Topic in the Same Conversation

It has been noted that when changing the topic in the same conversation, Hedgehog might find it difficult to answer the question than when the chatbot is restarted and starts off with a clear knowledge list. Since the knowledge list is not cleared when the topic is changed, the chatbot might have a hard time finding the correct answer. This is a limitation probably due to smaller size of the model, although it may also be resolved with further fine-tuning of the model where the knowledge list consists of different topics.

Here is an example of a conversation where the topic was changed from the previous conversation about Abe and Kuroda to a topic on cybersecurity startups.

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
User: What was David Cowan's comment regarding the cyber security startups in 2018?
Hedgehog: “I have never seen such a fast-growing market with so many companies on the losing side,” said David Cowan, a partner at Bessemer Venture Partners, a venture capital firm that has invested in the cyber security sector.
User: What happened to the security products these startups have developed?
Hedgehog: CANNOTANSWER:
```

From the conversation above, it was unable to answer the question about what happened to the security products these startups have developed. Note that Hedgehog was able to answer the question when we started the conversation directly about cybersecurity startups. The following is the query that was fed to the LLM model to answer the question "What happened to the security products these startups have developed?".

```
Instruction: given a dialog context and related knowledge, you need to response safely based on the knowledge. [CONTEXT] What factors do the Japanese government consider when determining whether the economy is out of deflation? EOS The government looks at several factors, such as inflation data and the output gap, in determining whether the economy is sustainably out of deflation. EOS What was David Cowan's comment regarding the cyber security startups in 2018? EOS “I have never seen such a fast-growing market with so many companies on the losing side,” said David Cowan, a partner at Bessemer Venture Partners, a venture capital firm that has invested in the cyber security sector. EOS What happened to the security products these startups have developed? [KNOWLEDGE] Title: Under threat: Cyber security startups fall on harder times. Instead, many in the crowded market are struggling to live up to their early promise. In some cases, the security products they developed have been overtaken by advances in cyber hacking, according to industry executives and venture capitalists. In others, larger competitors have come out with similar technology and locked down customers. “I have never seen such a fast-growing market with so many companies on the losing side,” said David Cowan, a partner at Bessemer Venture Partners, a venture capital firm that has invested in the cyber security sector. Companies are also diverting money to lower-cost “bug bounty” firms that contract out researchers who help identify security weaknesses. Within a few years, its market became more competitive, as rivals such as Cylance, CrowdStrike, and SentinelOne came out with similar technologies. Some larger companies, including Symantec Corp ( SYMC.O ), also developed similar products. Carbon Black CEO Patrick Morley, in an interview, declined to comment on plans around a possible IPO, or any merger discussions. Another startup, Zscaler, which specializes in cloud security, hired investment banks to go public last year but has delayed its offering until at least March to focus on growing its revenue, according to sources. Zscaler declined to comment.
```

It could be seen from the query above that it still contains the context from the previous conversation about Abe and Kuroda. This might have caused the model to be confused when answering the question about cybersecurity startups. This is a limitation of the model probably due to its small size, and it can be resolved by clearing the knowledge list when the topic is changed. Take note that GPT-3.5-Turbo was able to answer this question despite changing the topic in the same conversation.

Another way to resolve this would be to clear the knowledge list for every question. However, the model might find it hard to answer follow-up questions that might require the knowledge from the previous question. In the future, determining the best way to clear the knowledge list from unrelated topics would be an interesting direction to explore.

# Accuracy in Similarity Search

There are cases where the semantic search results are not accurate in the sense that the text chunk that contains the correct answer is not the top result. For example, for the question "Has Abe decided whether to reappoint Kuroda for another five-year term?", the results are as follows:

```
results = [
    Document(
        page_content="Title: Japan's Abe tells Kuroda to keep up efforts on economy. Many analysts see a good chance that Kuroda will be reappointed when the government selects successors to him and his two deputies in coming months, decisions that need parliament's approval.",
        lookup_str="",
        metadata={"id": 1493, "article_id": 152, "position": 6},
        lookup_index=0,
    ),
    Document(
        page_content="Title: Japan's Abe tells Kuroda to keep up efforts on economy. Kuroda was handpicked by Abe to take the helm of the central bank in 2013 to deploy a massive stimulus program - part of the premier's \"Abenomics\" reflationary policies - that helped boost growth but failed to drive up inflation to the bank's target of 2 percent.",
        lookup_str="",
        metadata={"id": 1489, "article_id": 152, "position": 2},
        lookup_index=0,
    ),
    Document(
        page_content="Title: Japan's Abe tells Kuroda to keep up efforts on economy. \"I want him to keep up his efforts. But I haven't made up my mind,\" on who should succeed Kuroda when his term ends in April, he added.",
        lookup_str="",
        metadata={"id": 1492, "article_id": 152, "position": 5},
        lookup_index=0,
    ),
]
```

In the above result, the third chunk is the correct answer. However, it is not the top result. This suggests that further improvement can be made on the search. This limitation can be attributed to the fact that we are only performing a simple similarity search. The top result indeed is similar to the question but it does not answer the question. This kind of relationships are probably not captured by the sentence embedding model that we used. In the future, we can explore other ways to improve the search results. One specific model that can be explored is the `sentence-transformers/multi-qa-mpnet-base-dot-v1 model` [8]. This model was trained on question-answer pairs and might be able to capture the relationship between the question and the answer better than the sentence-transformers/all-MiniLM-L6-v2 model.

# Text Chunking

Previously, it was mentioned that it is required to split the text into chunks before converting it into sentence embeddings due to the size limitation of the model. By keeping the chunks smaller, the chatbot is able to pinpoint which specific parts of the article is related to the user's input.

However, the issue with chunking the text and directly converting it to sentence embeddings is that the independent chunks may lose the context that is present in the entire text. One remedy to this is by appending the title of the text to the beginning of the chunk. This is because the title of the text usually contains the main idea of the text. However, this is usually not enough. For example, consider the following text:

```
"I hit my irons as well as some of the best on tour and if I'm able to keep it in play, and give myself good positions to attack hole locations I'm going to get a lot of birdie opportunities.
```

From the above text, we wouldn't know who is the subject of the sentence. Unfortunately, the title "New year same as old for first-round leader Leishman" does not provide the correct answer. Only when previous sentences are taken into account, it can be deduced that the subject is Jhonattan Vegas. Below are the three chunks that precede the above text:

```
Vegas is also coming off a good year during which he climbed into the top 50 in the world, with he hopes more to come.
Like Leishman, he knows he can beat the best when he hits it straight.
"It's a big key for my game to drive it well," Vegas said.
```

## References

[1] https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
[2] https://github.com/facebookresearch/faiss/wiki
[3] https://python.langchain.com/en/latest/_modules/langchain/vectorstores/faiss.html
[4] https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
[5] https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances
[6] https://docs.langchain.com/docs/
[7] https://huggingface.co/microsoft/GODEL-v1_1-large-seq2seq
[8] https://huggingface.co/sentence-transformers/multi-qa-mpnet-base-dot-v1
