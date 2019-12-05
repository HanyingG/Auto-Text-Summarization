# Auto-summarization


This project is try to use SIF method to do sentence embedding and generate a text auto-summarization model. 

The main steps are:
1. Use Gensim w2v to train word embedding based on Wikipedia chinese corpus and chinese news corpus, the data cleaning and w2v model trainining is in folder code/Corpus_cleaning_and_w2v_model.py
2. Generate auto-summarization model by introducing SIF sentence embedding and KNN smooth method, please see file Model/Text_summarize_model.py

Reference: please see paper < A Simple but Tough-to-Beat Baseline for Sentence Embeddings > 
https://openreview.net/forum?id=SyK00v5xx
