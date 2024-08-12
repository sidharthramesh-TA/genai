
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
4. Text Summarization

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

`
[PAD]
` - Spacer to increase length of tokens to keep all tokens consistent in length

## Attention

Self-attention, also known as intra-attention, is a mechanism within neural networks that allows the model to weigh the importance of different words in a sequence relative to each other. Self Attention is how BERT generates Contextual Embeddings . Each token is represented by a token embedding - which is a vector of fixed length. BERT takes in input embeddings xt for each input token vt
of input sequence v1,v2,...vt . Embeddings are transformed by linear mappings to query vectors Q, key vectors K, value Vectors V

![alt text](image-28.png)

Here, \( X \) is the input embedding, and \( W_Q \), \( W_K \), \( W_V \) are learnable weight matrices.

Multi-headed attention is an extension of the self-attention mechanism that allows the model to focus on different parts of the sequence from multiple perspectives or "heads" simultaneously. Each head operates independently, learning different aspects of the relationships between words.

![alt text](image-29.png)

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

In the `Pretrained Students learn better` paper [2] the authors explore the impact of pretrainig on compact models. state-of-the-art models are expensive both in the cost and computational complexity sense. The authors explored a way to get the same gains that a large model would by pre-training compact models. 

BERT is pre-trained on two unsupervised tasks

1. Masked Language Modelling (MLM)
2. Next Sentence Prediction (NSP)

### MLM

For MLM tasks a given input sentence is masked with 15% of the words and trained with the network to predict masked words. To predict the Masked word, BERT reads the sentence in both directions and predicts the masked word. Lets look at an example

BERT takes Input data as embeddings, Embeddings are numerical vector arrays that words or tokens have been converted to, using the layers indicated below

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

pre-training a BERT model allows it to learn syntactic and semantic properties of the language. This can be used used for training tasks for subsequent fine tuning. 


![alt text](image-27.png)

Fine tuning allows the model to regress on smaller amount of data for a specific task which leads to better model in less data , less time. The entire model, including the pre-trained layers and the new task-specific layers, is trained on the labeled dataset. The weights are adjusted to optimize performance on the new task while retaining the language understanding learned during pre-training. Fine tuning also generally requires change in architecture like adding a new layer of logistic classifiers. The output of this exercise is that the skill learned can be transfered to similar problem types. This is called Transfer Learning. This ability is the main reason behind the fame for BERT type models. 

`Pre-training + Fine tuning = Transfer learing` 

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



### Many BERTs

as the field of NLP continues to evolve, various BERT-based models have been developed to address specific limitations, enhance performance, and optimize for different use cases.

| Model                                                                                             | Functionality                                                                                                                                            |
|---------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|
| [ALBERT](https://arxiv.org/pdf/1909.11942.pdf)                                                    | A lite version of BERT that reduces model size by sharing parameters across layers and decomposing the embedding matrix.                                  |
| [RoBERTa: A Robustly Optimized BERT Pre-training Approach](https://arxiv.org/pdf/1907.11692.pdf)  | Enhances BERT by training with more data, larger batches, and longer sequences. Removes Next Sentence Prediction (NSP) task.                              |
| [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/pdf/2003.10555.pdf) | Uses a different approach by training a discriminator to distinguish between real and fake tokens created by a generator, making it more sample efficient. |
| [SpanBERT: Improving Pre-training by Representing and Predicting Spans](https://arxiv.org/pdf/1907.10529v3.pdf) | Focuses on span-level prediction tasks rather than token-level, improving performance on span-based tasks like question answering.                        |
| [Distilling the Knowledge in a Neural Network](https://arxiv.org/pdf/1503.02531.pdf)              | Introduces the concept of model distillation, where a smaller model (student) learns from a larger model (teacher).                                       |
| [DistilBERT: a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/pdf/1910.01108.pdf) | Distills BERT into a smaller, faster, and cheaper model that retains 97% of BERT's performance while being 60% faster.                                   |
| [TinyBERT: Distilling BERT for Natural Language Understanding](https://arxiv.org/pdf/1909.10351.pdf) | Further compresses BERT using a two-stage learning framework, aiming to achieve a smaller model with competitive performance.                            |
| [Distilling Task-Specific Knowledge from BERT into Simple Neural Networks](https://arxiv.org/pdf/1903.12136.pdf) | Focuses on task-specific distillation where BERT's knowledge for a specific task is distilled into a simpler neural network, improving efficiency.       |





# GPT Models

In this section we will briefly look at the historical devleopment over the past decades and observe the technical evolution that kept adding to the innovations streak that eventually lead to the breakthroughs in LLMs 

![Development of LLMs](image.png)

Evolution process of 4 generations of LMs. [6]

In the above image we see that Statistical LMs were getting investigated from the 90s. Initially were applied only for the relatively smaller tasks like spell-check as Models were limited to looking at only a single word.

Autoregressive language models (AR) or Generative Pre-Trained (GPT) were Developed by Open AI in 2018 , they are a Decoder only architecture and excel at Natural Language Generation (NLG) tasks such as summarization, creative writing etc. GPT models recieve a subsequence v1, v2, .... vn of input tokens and generate contextual embeddings for each token and use that to predict next token using `Maximum Likelihood estimation` , thus predicting all the tokens in a sentence [17].

![alt text](image-45.png)

The quality of a language model may be measured by the probability p(v1,v2...vt) of a given text collection v1,v2,....vt such that if we normalize the inverse of the number of Tokens T we get perplexity. GPT version progress correspond to lowering the perplexity on benchmark data sets. GPT 2 demonstrated the lowering from 46.5 to 35.8% [17]


![alt text](image-46.png)

`Perplexity` is a measure of probability of text ,
 
 `low perplexity = high probability`

 ![alt text](image-52.png)

 BERT vs GPT

## GPT 

GPT was first released in June 2018 , it wasnt really great in solving NLP tasks so thats where Open AI iterated over to GPT 2 in Feb 2019. ChatGPT's own architecture can be estimated as follows

`GPT + Chat interface + Appropriate content check = ChatGPT`

Since the first release of OpenAI's GPT , there have been various iterations and use cases that were tried and tested out. Initial ones being more academic in nature and later ones being more enterprise and general public oriented. Initial Models were strictly text based whereas GPT 3 onwards image and videos also got included as domains to explore gainful applications.

![Open AI Chat GPT timeline](image-2.png)

GPT timelines. [6]

Notice that even if GPT 3 was released in 2020, it wasnt until the release of ChatGPT which used GPT 3.5 in Nov 2022 that it the LLM development got to prominance. The magic ingredient that got added to the GPT3 recipe was Human Alignment. This is really just means making GPT 3 more tuned to giving responses the way we expect. Also called Human-in-the-loop, the GPT 3 model was made to undergo Reinforcement Learning from Human in the Loop (RHLF). This basically is the Thumbs up or Thumbs down that you sometimes get prompted when youre working with ChatGPT and responses obtained are weighted with a positive response meaning reward and a negative response meaning punishment to the Agent. 

![Chat GPT models and properties](image-4.png)

GPT models and properties. [15]

We notice the nonlinear increase of training data and the linear increase in output sequences that GPT is able to generate.

### GPT 1 or Generative Pre Training

In the  [GPT Paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) circa 2018, developers in OpenAI demonstrate thier model which was able to solve some Natural Language Understanding (NLU) tasks. It used generative, decoder only Transformer architecture and used hybrid approach of unsupervised pretraining and supervised fine tuning [Survey]. The breakthrough was to demonstrate that a generic model could outperform a specifically designed algo that was made for solving a specific NLU problem [18]. The innovation here was Generative Pre Training or GPT as we've come to know it was able to establish the principle that language models could sucessfully predict the next word.

![alt text](image-35.png)

Architecture of GPT model

GPT models were tested on things like

1. Word Similarity - 5.7%
2. Q & A - 4.2%
3. Commonsense Reasoning - 4%

![alt text](image-36.png)

Relative performance

modest improvements, GPT was still lacking with respect to coherent response. Training data while impressive at 117M parameters was not enough.

### GPT 2 or Unsupervised Multi Task Learner

 GPT training data needed more so the scale was increased to 1.5B parameters and was trained on webText data. Training data was the large WebText data which was labelled, add to that was the probabilistic approach for multi-task solving , which predicts the output based on the input and task information. 

```
Since the (task-specific) supervised objective is the same as the unsupervised (language modeling) objective
but only evaluated on a subset of the sequence, the global minimum of the unsupervised objective 
is also the global minimum of the supervised objective (for various tasks)
```

Some of the Tasks tested 

1. Children's Book test
2. Commonsense Reasoning
3. Q & A
4. Summarization


Now GPT 2 was a generic model that was able to perform on multiple NLU tasks but was not able to produce significantly better results. One reason was Model size.

![GPT 2 Gif](gpt-2-autoregression-2.gif)

### GPT 3 or Few Shot Learners

`GPT 2 +  Scale = GPT 3`

Developers in Open AI knew they had something that would work, it was showing promise but was hungry was data. They quickly iterated and fed it 175B parameters and tested it out Circa 2020. Along with the scale was the innovation in approach called ` in-context learning (ICL) ` which introduced prompting as a way to make the model learn. Zero shot and Few Shot learning were introduced as a way to make the model do tasks that it was not explicitly trained on. 

Learning Examples

1. Zero Shot Learning - No prior examples given
2. One Shot Learning - One example to set context
3. Few Shot Learning - Multiple Examples given, in the words of Open AI , all LLMs are few shot learning

```
Pretraining = correct word prediction

ICL = correct task solution
```

GPT 3 basically ushered in the era of LLM, what was prior called as PLMs now eveolved into LLM where the sheer scale of the model gives it power to do tasks that it was not explicitly trained for. So much was the influence of few shot learning in the development of GPT 3 that the authors dubbed the title of the paper as `few shot learning` much like the transformers paper was titled on `attention`

![GPT 3 Gif](05-gpt3-generate-output-context-window.gif)

![alt text](image-37.png)

Large models make effecient use of Few Shot learning

![alt text](image-38.png)

Examples of Few Shot learning

![alt text](image-39.png)

Comparison of Compute Training

![alt text](image-40.png)

Q & A accuracy

![GPT 3 example](image-41.png)

Example of 175B parameters model performing better with few shot learning

![alt text](image-42.png)

SAT Analogies

![alt text](image-43.png)

News text generation

As impressive GPT 3's performance are it also highlighted the need for being mindful for **Bias and Fairness**.

![alt text](image-44.png)

Example of Bias in GPT 3

### GPT 3.5

Now GPT 3 was nice and all but it was still not good enough. Before Open AI would release models that developers and data scientists who knew how to work with it would get interested with new releases. Now that GPT 3 had demonstrated conversational ability Open AI was preparing for getting bigger audience but before they could do that they wanted to make sure the vices of social media interactions could be controlled (Microsoft had already had a episode of releasing a bot on twitter and within one day the bot due to the learning it had got from the interactions from public started generating offensive responses and had to be taken down in a day) and the model be made more responsive to human input. One big leap in performance was due to Reinforcement Learning with human feedback. Open AI took the incremental gain that they got using Beta Users, added a conversational interface similar to popular chatting interfaces and made it available to the world Circa 2022 November - ChatGPT.

![Exam Performance](image-31.png)

Studies performed on understanding the robustness of 3.5 with prior models were published by [chinese research group](https://arxiv.org/pdf/2303.00293) in 2023 exploring the different attributes that go into generating responses and the quality of response. This one [paper](https://arxiv.org/pdf/2303.10420) in particular compares 3.5 with 3 and presents its findings 

### GPT 4 or Technical Report

Unlike previous models OpenAI has not released GPT 4 yet, or has talked about the architectural innovations with scientific rigor. Instead they have released a whitepaperesque content on the performance and use cases that GPT 4 has been subjected to. 

`GPT 3 - Malicious content + Intervention Alignment = GPT4`

an innovation that is getting some recognition is the Predictable Scaling mechanism that can accurately predict final performance with small proportion of compute during model training.

One other innovation that got bundled up was Multimodal capacity which allows GPT 4 to See , Hear and Speak. GPT 4o has successfully demonstrated ability to speak and respond to voice utterance. 

## GPT Example

1. Temperature - Like Simulated Annealing , makes the model more confident or less confident. Lower value makes it select highest propapble answer, higher value makes it select a lower probable random answer
2. top_k - How many tokens the model considers when generating
3. top_p - only considers token from the top X% of confidences
4. beams - How many tokens out should we consider
5. do_sample - if set to true, randomness is introduced in selection


Open AI has made available [Tokenizer viz](https://platform.openai.com/tokenizer) on its website that allows us to see the breaking up of words into Tokens

![alt text](image-34.png)

[Interactive GPT Example](https://poloclub.github.io/transformer-explainer/)



## Changing Nature of ChatGPT

GPT's like other models have been subject to intense iterations and improvements over time. Several studies have been made to understand the performance of models and to benchmark them against new heights. One such [paper that compares ChatGPT 3.5 and 4](https://ar5iv.labs.arxiv.org/html/2307.09009) gives an insight on the various problems that were subjected to it  

Models like ChatGPT 3.5 and 4 are evaluated for problems like

1. Math problems
2. Sensitive/dangerous questions
3. Opinion surveys
4. Multi-hop knowledge-intensive questions
5. Generating code
6. US Medical License tests
7. Visual reasoning

![alt text](image-33.png)

Overview of Performance and insturction drift

![alt text](image-47.png)

Math Example

![Answering Controversies](image-48.png)

Answering Controversial Questions

![Code Adherence](image-49.png)

Executable code response reduced from 52% to 10%

![alt text](image-51.png)

GPT 4 has reduced instruction following capacity




## Comparable Models

![Existing Large Language Models](image-1.png)
Timeline of LLMs. [6]

Above image is a zoomed out version of the timelines where publicly available LLMs were released. We observe the word soup of LLMs that are released from 2020 till 23 and the competitive landscape with many companies released thier own versions and making them available to the public. 

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

19. [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/pdf/1909.11942.pdf)

20. [RoBERTa: A Robustly Optimized BERT Pre-training Approach](https://arxiv.org/pdf/1907.11692.pdf)

21. [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/pdf/2003.10555.pdf)

22. [SpanBERT: Improving Pre-training by Representing and Predicting Spans](https://arxiv.org/pdf/1907.10529v3.pdf)

23. [Distilling the Knowledge in a Neural Network](https://arxiv.org/pdf/1503.02531.pdf)

24. [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/pdf/1910.01108.pdf)

25. [TinyBERT: Distilling BERT for Natural Language Understanding](https://arxiv.org/pdf/1909.10351.pdf)

26. [Distilling Task-Specific Knowledge from BERT into Simple Neural Networks](https://arxiv.org/pdf/1903.12136.pdf)

27. [Transformer Explainer](https://poloclub.github.io/transformer-explainer/)

28. [BERT compendium](https://huggingface.co/docs/transformers/model_doc/bert)

29. [GPT Playground](https://textsynth.com/completion.html)

30. [RAG Paper](https://arxiv.org/pdf/2005.11401v4)

31. [Chat GPT's Changing behaviour](https://ar5iv.labs.arxiv.org/html/2307.09009)

32. [GPT 3.5 robustness](https://arxiv.org/pdf/2303.00293)

33. [GPT 3 vs GPT 3.5](https://arxiv.org/pdf/2303.10420)

