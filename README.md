# Auto-summarization


This project is try to use SIF method to do sentence embedding and generate a text auto-summarization model. 

The main steps are:
1. Use code/Wikipedia Extractor.py to extract chinese text, convert traditional chinese into simple chinese, split sentence, and then get rid of stopwords (please refer to reference file in folder "stopwords") and get wiki corpus.
2. Use code/Corpus_cleaning_and_w2v_model.py to clean chinese news data "sqlResult_1558435.csv" to get chinese corpus, and merge with wiki corpus, then use Gensim w2v to train word embedding based on this merged corpus.
3. Generate auto-summarization model by introducing SIF sentence embedding method, calculate the embeddings of every single sentence, the title, and the whole passage accordingly. Then calculate the similarity score between each sentence embedding with the whole doc embedding, and use KNN to smooth the similarity. Lastly, get the top_N similar sentences as the summarization of this doc. Please see file Model/Text_summarize_model.py for details.

(Corpus data files are too large, will be uploaded later...)

Reference: please see paper < A Simple but Tough-to-Beat Baseline for Sentence Embeddings > 
https://openreview.net/forum?id=SyK00v5xx
