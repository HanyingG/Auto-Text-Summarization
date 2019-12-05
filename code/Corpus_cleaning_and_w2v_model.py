# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 10:57:42 2019
"""

import pandas as pd
import numpy as np
import re
import jieba
from gensim.models import FastText
import matplotlib.pyplot as plt

'''
Chinese News Corpus
'''
FILE_PATH1 = 'export_sql_1558435.zip'
news_df = pd.read_csv(FILE_PATH1,compression='zip',encoding='gb18030')

def clean_sentence(val):
    "remove chars that are not letters or numbers, downcase, then remove stop words"
    regex = re.compile('([^\s\w]|_)+')
    sentence = regex.sub('', val).lower()
    sentence = sentence.split(" ")
            
    sentence = " ".join(sentence)
    return sentence

def clean_dataframe(data):
    "drop nans, then apply 'clean_sentence' function to question1 and 2"
    data = data.dropna(how="any")
    
    for col in ['title', 'content']:
        data[col] = data[col].apply(clean_sentence)
    
    return data

news_df = clean_dataframe(news_df)
#tokenize 
def cut(text): return ' '.join(jieba.cut(text)) 
main_content = pd.DataFrame()
main_content['tokenized_content'] = news_df['content'].fillna('').apply(cut)

with open('hy_corpus.txt','w',encoding='utf-8') as f:
    f.write(' '.join(main_content['tokenized_content'].tolist()))

'''
wiki Chinese corpus
'''

#wikiExactor
with open('wiki_00', 'r', encoding="utf-8") as f:
    text = f.read()

from opencc import OpenCC
cc = OpenCC('t2s')
converted = cc.convert(text)
with open("jian_wiki.txt", "w", encoding="utf-8") as f:
    f.write(converted)

#
import re
def split_sentences(text,p='[。？！；]',filter_p='\s+'):
    f_p = re.compile(filter_p)
    text = re.sub(f_p,'',text)
    pattern = re.compile(p)
    split = re.split(pattern,text)
    return split
file = open('jian_wiki.txt',"r",encoding="utf-8")
text = '\n'.join(split_sentences(file.read()))
with open("wiki_final.txt", "w", encoding="utf-8") as f:
    f.write(text)
def delblankline(infile, outfile):
    infopen = open(infile, 'r',encoding="utf-8")
    outfopen = open(outfile, 'w',encoding="utf-8")
    db = infopen.read()
    outfopen.write(db.replace('>','>\n'))
    infopen.close()
    outfopen.close()

delblankline('wiki_final.txt', 'wiki_00.txt')

#get stop words
def get_stopwords():
    #
    stopword_set = []
    with open("CNstopwords.txt",'r',encoding="utf-8") as stopwords:
        for stopword in stopwords:
            stopword_set.append(stopword.strip("\n"))
    return stopword_set

#parse text
def parse_zhwiki(read_file_path,save_file_path):
    #filter out <doc>
    regex_str = "[^<doc.*>$]|[^</doc>$]"
    file = open(read_file_path,"r",encoding="utf-8")
    
    output = open(save_file_path,"w",encoding="utf-8")
    content_line = file.readline()
    #get stopwords 
    stopwords = get_stopwords()
    
    article_contents = ""
    i = 1
    while content_line:
        match_obj = re.match(regex_str,content_line)
        content_line = content_line.strip("\n")
        if len(content_line) > 0:
            if match_obj:
                #use jieba to tokenize words
                words = jieba.cut(content_line,cut_all=False)
                for word in words:
                    if word not in stopwords:
                        article_contents += word+" "
                article_contents = article_contents+"\n"
            else:
                if len(article_contents) > 0:
                    i += 1
                    print(i)
                    output.write(article_contents)
                    article_contents = ""
        content_line = file.readline()
    output.close()

parse_zhwiki('wiki_00.txt','wiki_corpus.txt')

#merge corpus together
def merge_corpus(input1,input2,output):
    o = open(output,"w",encoding="utf-8")
    f1 = open(input1, "r",encoding="utf-8")
    f2 = open(input2, "r",encoding="utf-8")
    corpus1 = f1.readlines()
    corpus2 = f2.readlines()
    print(len(corpus1),len(corpus2))
    match_list = [] 
    for token in corpus1:
        match_list.append(token) 
    match_list.append('\n')
    for token in corpus2:
        match_list.append(token)
    print(len(match_list))
    for i in match_list:
        o.write(i)
    o.close()

merge_corpus('hy_corpus.txt','wiki_corpus.txt','all_corpus.txt')

'''
train word embedding 
'''
from gensim.models import word2vec  
import logging
import os.path
import sys

#get logger info
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))
sentences = word2vec.LineSentence('all_corpus.txt')
model  = word2vec.Word2Vec(sentences,size = 400,window = 5, min_count=200, workers=4)  
model.save('word2vecModel')
model.save_word2vec_format('word2vecVector', binary=False)

#call model 
model = word2vec.Word2Vec.load('word2vecModel')

'''
test word embedding similarity 
'''
print(model.wv.most_similar('勇敢'))
print(model.wv.most_similar('美女'))

from wordcloud import WordCloud

#get a circle mask 
def get_mask():
    x,y = np.ogrid[:300,:300]
    mask = (x-150) ** 2 + (y-150)**2 > 130 ** 2
    mask = 255 * mask.astype(int)
    return mask
 
#draw word cloud 
def draw_word_cloud(word_cloud):
    wc = WordCloud(background_color="white",mask=get_mask(),font_path='C:/Windows/Fonts/msyh.ttc')
    wc.generate_from_frequencies(word_cloud)

    plt.axis("off")
    plt.imshow(wc,interpolation="bilinear")
    plt.savefig('cloud.jpg', dpi=500)
    plt.show()
 
def test(word):
    logging.basicConfig(format="%(asctime)s:%(levelname)s:%(message)s",level=logging.INFO)
    model = word2vec.Word2Vec.load('word2vecModel')
    #find the top_10 most similar words 
    one_corpus = [word]
    result = model.wv.most_similar(one_corpus[0],topn=100)
    
    word_cloud = dict()
    for sim in result:
        # print(sim[0],":",sim[1])
        word_cloud[sim[0]] = sim[1]
    
    draw_word_cloud(word_cloud)
test('中国')

'''
word embedding semantic similarity 
'''
def analogy (x1,x2,y1):
    result = model.most_similar(positive = [y1,x2],negative = [x1])
    return result[0][0]
print(analogy('中国', '汉语', '美国'))
print(analogy('美国', '奥巴马', '美国'))
print(analogy('美国', '奥巴马', '朝鲜'))

'''
visualization 
'''
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random
#import chinese font 
from matplotlib.font_manager import FontProperties
font = FontProperties(fname='C:/Windows/Fonts/msyh.ttc')

def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []
    vocab = list(model.wv.vocab)
    random.shuffle(vocab)

    for word in vocab[:600]:
        tokens.append(model[word])
        labels.append(word)
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
    
    plt.figure(figsize=(40, 30)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom',
                     fontproperties=font)
    plt.savefig('first.jpg', dpi=800)
    plt.show()

tsne_plot(model)