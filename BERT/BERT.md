# BERT

Bidirectional Encoder and Representations for Transformers (BERT) are autoencoders (AE) models which recive an input text and produce contextual embedding for each token. Getting used widly these days owing to the success that it has deonstrated in language tasks.

a simple way to think of it would be 

Transformers - Decoders = BERT

BERT architecture uses only Encoders to predict the next word and next sentence

Named Entity Recognition (NER) is key to finetuning BERT to make it perform better for specific tasks

Two famouse BERT models are BERT base and BERT large , both can be found in hugging face





# GPT Models

This section is about exploring GPT Models. Autoregressive language models (AR) or Generative Pre-Trained (GPT) models recieve a subsequence v1, v2, .... vn of input tokens and generate contextual embeddings for each token and use that to predict next token, thus predicting all the tokens in a sentence. 

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




