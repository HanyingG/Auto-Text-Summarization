# -*- coding: utf-8 -*-
"""
Spyder Editor
This model is to generate the summary of a news text 
"""

import pandas as pd
import numpy as np
import jieba
import re
from gensim.models import Word2Vec
from sklearn.decomposition import TruncatedSVD
from collections import Counter,defaultdict
from scipy.spatial.distance import cosine


##tokenize function
def cut(string):
    return ' '.join(jieba.cut(string))

#load news content data
path = 'C:/Users/hanying/'
news_content = pd.read_csv(path+'sqlResult_1558435.csv',encoding = 'gb18030')
news_data = news_content[['content','title']]

news_data = news_data.fillna('')


##load w2v model 
model = Word2Vec.load(path +'word2vecModel')

## build word counts for p(w)
with open(path+'hy_corpus.txt','r+',encoding='utf-8') as f:
    lines = f.read()
    full_corpus = []
    for line in lines.split('\n\n'):
        full_corpus.append(line)

#get all tokens from corpus
all_tokens = [t for l in full_corpus for t in l.split()]
token_cnt = Counter(all_tokens)
Prob_word = {w: count/len(all_tokens) for w, count in token_cnt.items()}

##build end-to-end process to get text summarization 
class Sentence_summarize(object):
      def __init__(self,content,title,K=3):
          self.content = re.sub('[\r\n]','',content)
          self.title = title
          self.Vs = self.doc_embedding_Vc()[0]
          self.Vt = self.doc_embedding_Vc()[1]
          self.Vc = self.doc_embedding_Vc()[2]       
          self.correlation_idx = self.get_corrlations()[0]
          self.correlation_score = self.get_corrlations()[1]
          self.K = K
          self.KNN_smooth_corr = self.KNN_smooth()
        
      def cut(self,string):
          return ' '.join(jieba.cut(string))
	
	## split sentence in each doc
      def sentence_split(self,sentence):
          pattern = re.compile('[。，！？\?]')
          res = pattern.sub(' ', sentence).split() 
          res_cut = [cut(x) for x in res]
          return res_cut
      
    #compute PCA 
      def pca_compute(self,X,npc=1):
          svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
          svd.fit(X)
          components = svd.components_
          return components
	
	##remove first PCA
      def remove_pc(self,X, npc=1):
          pc = self.pca_compute(X,npc)
          if npc==1:
              XX = X - X.dot(pc.T) * pc
          else:
              XX = X - X.dot(pc.T).dot(pc)
          return XX
	
	## embedding a single sentence
      def sentence_embedding_Vs(self, sentence):
          alpha = 1e-4
          global Prob_word
        
          max_fre = max(Prob_word.values())
          words = self.cut(sentence).split()
          sentence_vec = np.zeros_like(model['小米'])
          words = [w for w in words if w in model]
          for w in words:
              p_w = Prob_word.get(w,max_fre)
              weight = alpha / (alpha + p_w)  ##in case the word doesn't exist in corpus
              sentence_vec += weight * model[w]
          
          sentence_vec /= len(words)
          ##check if there is any NAN sentence embedding
          if np.isnan(sentence_vec).any():
              sentence_vec = np.zeros(sentence_vec.shape)
         
          return sentence_vec
    
      def doc_embedding_Vc(self):
          content = self.content
          title = self.title 
          ##add title as the first sentence of the content
          S_lst = [self.sentence_embedding_Vs(title)] + [self.sentence_embedding_Vs(s) for s in self.sentence_split(content)]
          ##append the whole content embedding as the last vectore
          S_lst.append(self.sentence_embedding_Vs(title+'。'+content))
          S_matrix = np.asarray(S_lst)
		
		#do PCA
          S = self.remove_pc(S_matrix,1)
		
          Vt = S[0,:]  ##get title embedding
          Vs = S[:-1,:] ##a matrix, to get sentence embedding for every sentence+title in one doc
          Vc = S[-1,:]  ##get embedding for the whole doc
		
          self.Vt = Vt
          self.Vs = Vs
          self.Vc = Vc
		
          return Vs,Vt,Vc
	
	## get similarity score for each sentence compared with whole content embeddding
      def get_corrlations(self):
          correlations = defaultdict(float)
          Vs_matrix = self.Vs
          V_c = self.Vc
		
          for i in range(Vs_matrix.shape[0]):
              correlation = cosine(Vs_matrix[i], V_c)
              if np.isnan(correlation):
                  correlation = 0.0
              correlations[i]= correlation
		
        ##get similarity score list in ascent order by sentence idx
          correlation_idx = sorted(correlations.items(), key=lambda x: x[0]) 
        ##get similarity score list in descent order by similarity score 
          correlation_score = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        
          return correlation_idx, correlation_score
	
	##do KNN_smooth by taking the average of similarity scores from K neighbors for sentence V_sj, where the neighbors means from a sequence perspective (later on will try K neighbors from a distance perspective)
	##the param 'correlation_lst' should be the first result (correlation_idx) from get_correlations function
      def KNN_smooth(self):
          cor_res = []
          k = self.K
          correlation_lst = self.correlation_idx
		
		##add a number of (k-1)//2 paddings for edge vectors in order to do KNN smooth in a same K sliding window
          cor_new = [correlation_lst[0] for i in range((k-1)//2)] + correlation_lst + [correlation_lst[-1] for i in range((k-1)//2)]  
          cor_new = list(map(lambda x:x[1],cor_new))
          for i in range((k-1)//2,len(correlation_lst)+1):
              new_score = sum(cor_new[i-(k-1)//2 : i+(k-1)//2 +1]) / k
              cor_res.append((i-(k-1)//2,new_score))
		
          KNN_smooth_corr = sorted(cor_res, key=lambda x: x[1], reverse=True)
        
          return KNN_smooth_corr
	
	## get top_N most similar sentence  and return summarize
      def get_summarize_sentence(self, N):
	
          content = self.content
          title = self.title          
          sorted_similarity = self.KNN_smooth_corr
		
          S_lst = [title] + [s for s in self.sentence_split(content)]
          top_n_idx = sorted_similarity[:N]
          selected_lst = []
          sort_summarize_idx = sorted(top_n_idx,key=lambda x:x[0])
          for idx,score in sort_summarize_idx:
              selected_lst.append(S_lst[idx])
        
          return ','.join(selected_lst)
    
    
    
# =============================================================================
# Test cases    
# =============================================================================
##test case 1
test_doc = news_data['content'][:1].astype(str).values[0]
test_title = news_content['title'][:1].astype(str).values[0]
a = Sentence_summarize(test_doc,test_title,5)
print(a.get_summarize_sentence(10))
