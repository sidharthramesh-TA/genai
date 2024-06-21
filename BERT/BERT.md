
# Gen AI training

In this section we will be exploring BERT and GPT Model architectures. Before we dive into BERT lets look at the broader classification of models that are used in Natural Language Processing

1. Autoencoders 
2. Autoregressive language models
3. Transformers encoder - decoders

These models are usually pre-trained on a large general training set and often fine-tuned for a specific task, hence are called Pre-trained language Models (PLM). When the number of parameters of these models get large and can be instructed by prompts then they are called Foundation Models [17]. 


# BERT

Bidirectional Encoder Representation from Transformer (BERT) are autoencoders (AE) models that has revolutionised NLP with thier State-of-the-art embedding model published by Google. It has successfully completed many NLP tasks such as 

1. Question Answering
2. Text Generation
3. Sentence Classification

Its success is largely due to its ability to do context-based embeddings, unlike context free embeddings like word2Vec

an oversimplified way to think of it would be 

`
Transformers - Decoders = BERT
`

### BERT Architecture

![alt text](image-25.png)

### Whats so Bidirectional about BERT ? 

Its able to read a given sentence left-to-right or right-to-left and is able to use the context learning that its got from digesting the input that its able to predict any word within that input sentence using the context.

### Why is it called Autoencorder ? 

Thats cause BERT requires only the encoder componenet of the Transformer duo. Output of a encoder layer is given as an input to another encoder layer and is propagated through the defined layers of BERT model.

## Tokens

The size of BERT vocabulary is 30K tokens. Input text is converted to Tokens by a specific kind of tokenizer like _Wordpiece_. What it does is that Common text like `dog` are assigned a token if they exist in the model vocabulary but if they are not present like `playing` then they are divided into `play` and `##ing` where the `##` gives the signal that its part of a word [17]. BERT also uses special tokens like 

`
[CLS] 
` - This is to signal the start of the Input sentence

`
[SEP]
` - This is to signal the end of Sentence

`
[MASK]
` - This is to signal the Masked word that BERT would need to predict



## Attention

Self Attention is how BERT generates Contextual Embeddings. Each token is represented by a token embedding - which is a vector of fixed length. BERT takes in input embeddings xt for each input token vt
of input sequence v1,v2,...vt . Embeddings are transformed by linear mappings to query vectors Q, key vectors K, value Vectors V

association score between the tokens are computed by taking a scalar product between query vector and key vector

![obligatory math](image-10.png)

above equation is also called scalar dot product attention and is normalized to a probability score using softmax function. We get the self attention algorithm that was proposed in the Attention paper [3] 

![self attention](image-11.png)

where ![alt text](image-12.png) is the new contextual embedding

![alt text](image-13.png) is the association score

![ VT](image-14.png) is the weighted average of the value vector 

the resulting embedding is a contextual embedding as it contains information about all words in the input text [17]

### example:

Sentence A: He got bit by Python. 

Sentence B: Python is my favorite programming language

plotting the attention relationships for the sentences we get 

![sentence A](image-15.png)   
Example 1

![sentence B](image-16.png)

Example 2

BERT is able to discern dynamic embeddings based on context as opposed to context free models like word2Vec which generate static embeddings without taking context into account. This is due to the impact of multi head attention mechanism [18]. What multi head attention here means is that each word in the sentence is related to all the words and thereby a relationship is derived.  

![Sentence A- BERT](image-17.png)

BERT generating the representation of each word in the sentence. 

Here `RHe` implies representation of the word `He` also called _embedding_  

Difference between BERT versions like base and large are the number of layers of encoders. Base contains 12 layers whereas large contains 24. 

## Visualizing Attention

In the BERTology paper [4] authors look at the various ways in which attention affects the output result. Following is a illustration of the types of matrices that are constructed by the Model with Attention values. 

![attention patterns in BERT](image-9.png)
attention patterns in BERT [4]

A interactive way of dealing with attention can be done by the following notebook. The example in the notebook can be repalced with a custom example. The notebook demonstrates the many layers and heads that are used for a simple BERT version.

[Visualizing BERT](https://colab.research.google.com/github/davidarps/2022_course_embeddings_and_transformers/blob/main/Visualizing_Attention_with_BertViz.ipynb#scrollTo=IAqLLQofc7IZ) 

## Pre-Training

BERT is pre-trained on two broad tasks

1. Masked Language Modelling (MLM)
2. Next Sentence Prediction (NSP)

### MLM

For MLM tasks a given input sentence is masked with 15% of the words and trained with the network to predict masked words. To predict the Masked word, BERT reads the sentence in both directions and predicts the masked word. Lets look at an example

BERT takes Input data as embeddings using the layers indicated below

1. Token Embedding
2. Segment Embedding
3. Position Embedding

Lets look at how tokens and embeddings affect the BERT process 

Sentence A: Paris is a beautiful city

Sentence B: I love Paris

Combining the above two sentences and breaking up them into tokens we get 

`
tokens = [Paris, is, a, beautiful, city, I, love, Paris]
`

`
tokens = [ [CLS], Paris, is, a, beautiful, city, [SEP], I, love, Paris, [SEP]]
`

![Token Embeddings](image-18.png)

Token Embeddings

In order to make it easier for the Model to understand sentence starting and ending following tokens are added to the array. [CLS] is added at the beginning of the sentence whereas [SEP] are added at the end of every sentence to indicate end of every sentence. 

![Segment Embeddings](image-19.png)

Segment Embeddings

Inorder to differentiate between sentences Segment embedding is done, EA in the above image relates to the first sentence whereas EB relates to the second sentence. 

![Position Embeddings](image-20.png)

Position Embeddings

To preserve the word order in the input sentence we have to provide a way to tag the ordr of the words. Note that E0 applies to [CLS] token whereas E10 is [SEP] , this would be expected as they correspondingly start and stop the Input text.

Combining the above embeddings we get 

![Final embeddings](image-21.png)

Final Embeddings 

BERT is an autoencoding language model which basically means that its able to read a given sentence left-to-right or right-to-left (Bidirectional) . During BERT training, a random 15% mask is applied on the words to train the network . Model reads the sentence in both directions and predicts the Masked word

![alt text](image-22.png)

Notice that the word City has been masked as part of the training. To predict Masked token, we pass that as input to a Feedforward network with a softmax activation. Feedforward + softmax takes in the input tokens and gives out a probability of the words used in the vocabulary 

![Predicting Token](image-23.png)

We observe that City is returned as the word with highest probability, which is the right answer

Two famouse BERT models are BERT base and BERT large , both can be found in hugging face 

### Next Sentence Prediction

NSP is another strategy used for training. It is a binary classification task. The input given are two sentences and BERT has to predict the relationship between the two sentences. Following example helps us understand better

Sentence A: She cooked pasta.

Sentence B: It was delicious.

In the above example Sentence B is a follow up of A. Now consider 

Sentence A: Turn the radio on.

Sentence B: She bought a new hat. 

There is no relation between Sentence B and A

To visualize the above example we have the following image that has the input text sentences going through the 12 layer BERT base and is then passed over to the Feed Forward network and Softmax to predict the likely nature of Sentence B

![alt text](image-26.png)

## Finetuning BERT

pre-training a BERT model allows it to learn syntactic and semantic properties of the language. This can be used used for training tasks for subsequent fine tuning - this is called transfer learning as it allows the pretraining knowledge to be transfered to a related application [17]. Finetuning is done by classification solved by a logistic classifier L to the output embedding of the [CLS] token at position 1 of BERT's last encoder block [17]   

![Finetuning BERT](image-7.png)
4 common finetuning tasks

1. Text Classification - classification of resturant reviews as positive or negative 
2. Text pair classification - This is used to establish relationship between consecutive sentences. 
3. Word Annotation or Named Entity Recognition (NER) annotating the tokens into clusters like location, persons, objects etc.
4. Span Prediction - this is to construct small sentences as response to a question and a pargraph of text which would contain the answer. 


![examples of FT](image-8.png)


Here we notice that for text classification the example given is `The sandwich was good and tasty` and BERT classifies the sentiment as `Positive`.

Text pair classification (NSP) example given is `It rains` and `the sun shines` and BERT is able to classify it rightly as `contradiction`

For NER example we have `Joe Biden went to New York` and BERT is able to identify the parts of sentence as Person, Object and Location

For Span prediction the input question is `who discovered America` and the answer is identified as `Columbus` 


# GPT Models

This section is about exploring GPT Models. Autoregressive language models (AR) or Generative Pre-Trained (GPT) models recieve a subsequence v1, v2, .... vn of input tokens and generate contextual embeddings for each token and use that to predict next token, thus predicting all the tokens in a sentence [17]. 

We will briefly look at the historical devleopment over the past decades and observe the technical evolution that kept adding to the innovations streak that eventually lead to the breakthroughs in LLMs 

![Development of LLMs](image.png)
Evolution process of 4 generations of LMs. [6]

In the above image we see that Statistical LMs were getting investigated from the 90s. Initially were applied only for the relatively smaller tasks like spell-check as Models were limited to looking at only a single word.

![Existing Large Language Models](image-1.png)
Timeline of LLMs. [6]

Above image is a zoomed out version of the timelines where publicly available LLMs were released. We observe the word soup of LLMs that are released from 2020 till 23 and the competitive landscape with many companies released thier own versions and making them available to the public. 

![Open AI Chat GPT timeline](image-2.png)
Chat GPT timelines. [6]

![Chat GPT models and properties](image-4.png)
Chat GPT models and properties. [15]

![Increase in the number of Parameters](image-5.png)
Increase in the number of Parameters. [15]

![extending Pre Trained LLMs](image-6.png)
extending Pre Trained LLMs





# References

1. [BERT Paper](https://arxiv.org/pdf/1810.04805)

2. [Importance of Pre-training Compact Models](https://arxiv.org/pdf/1908.08962)

3. [What we know about How BERT works](https://arxiv.org/pdf/2002.12327)

4. [What does BERT look at](https://nlp.stanford.edu/pubs/clark2019what.pdf)

5. [Attention Paper](https://arxiv.org/pdf/1706.03762)

6. [Survey of Large Language Models](https://arxiv.org/pdf/2303.18223)

7. [GenAI Handbook](https://genai-handbook.github.io/)

8. [BERT Illustrated](https://jalammar.github.io/illustrated-bert/)

9. [GPT 2 Illustrated](https://jalammar.github.io/illustrated-gpt2/)

10. [GPT 3 Illustrated](https://jalammar.github.io/how-gpt3-works-visualizations-animations/)

11. [GPT Paper Link](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

12. [GPT 2 Paper Link](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

13. [GPT 3 Paper Link](https://arxiv.org/pdf/2005.14165)

14. [GPT 4 Technical Report](https://arxiv.org/pdf/2303.08774)

15. Transforming Conversational AI, Apress 2023

16. [FT's Visual Explanation of LLMs](https://ig.ft.com/generative-ai/?xnpe_tifc=4fbXhF1jbD_7OkxX4.V7bjpJVdUZMds_Ou4.4FEN4fxdtfUN4.oA4FhshCJNbIQutI4sxD_Z4FeLbdoD4jXlxfQNhInZbdQLbDe_hfbd&utm_source=exponea&utm_campaign=B2B%20|%20One%20off%20|%20AI%20visual%20story%20promo%20|%20140923&utm_medium=email)

17. Foundation Models for Natural Language Processing, Springer 2023, Sven Gisselbach

18. Getting Started with Google BERT , Packt Publishing, Sudharshan Ravichandran 2023
