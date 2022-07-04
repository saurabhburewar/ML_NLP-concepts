#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
import logging
import spacy
import gensim
from pyLDAvis import gensim
import pyLDAvis
import pprint
from gensim.models import CoherenceModel
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
from operator import index
from nltk import word_tokenize, ngrams, pos_tag
from nltk import RegexpParser, Tree, ne_chunk
from nltk.corpus import stopwords
import pandas as pd
import re
import math
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm
# import nltk

# nltk.download('stopwords')
# nltk.download('popular')


class Vocabulary:

    def __init__(self):
        self.wordCount = {}
        self.numWords = 0
        self.tagCounts = {}
        self.npCounts = {}
        self.vpCounts = {}
        self.nerList = {}
        self.wordimp = {}
        self.Ngrams = []
        self.stopwords = set(stopwords.words('english'))

    def addWord(self, word):
        if word != '':
            if word not in self.stopwords:
                if word not in self.wordCount:
                    word = re.sub(r'[^\w\s]', '', word)
                    self.wordCount[word] = 1
                    self.numWords += 1
                else:
                    self.wordCount[word] += 1

    def addPara(self, para):
        para = word_tokenize(para)
        for word in para:
            self.addWord(word)

        return self

    def getWords(self):
        return list(self.wordCount.keys())

    def getWordsWithCounts(self):
        return dict(sorted(self.wordCount.items(), key=lambda x: x[1], reverse=True))

    def getVocabSize(self):
        return self.numWords

    def calNgrams(self, para, n):
        nGrams = ngrams(word_tokenize(para), n)
        for gram in nGrams:
            gramcomp = ' '.join(gram)
            if gramcomp not in self.Ngrams:
                self.Ngrams.append(gramcomp)

        return self

    def getNgrams(self):
        return self.Ngrams

    def posTagging(self, para):
        tags = pos_tag(word_tokenize(para))
        for word, tag in tags:
            if tag not in self.tagCounts:
                self.tagCounts[tag] = 1
            else:
                self.tagCounts[tag] += 1

        return self

    def getTagCounts(self):
        return dict(sorted(self.tagCounts.items(), key=lambda x: x[1], reverse=True))

    def npChunking(self, para):
        grammar = "NP: {<DT>? <JJ>* <NN.*>+}"

        tags = pos_tag(word_tokenize(para))
        parser = RegexpParser(grammar)
        result = parser.parse(tags)
        for elem in result:
            if isinstance(elem, Tree):
                NPstr = ""
                for (text, tag) in elem:
                    if NPstr != "":
                        NPstr = NPstr + " " + text
                    else:
                        NPstr = NPstr + text
                if NPstr not in self.npCounts:
                    self.npCounts[NPstr] = 1
                else:
                    self.npCounts[NPstr] += 1

    def getNpCounts(self, n):
        return dict(sorted(self.npCounts.items(), key=lambda x: x[1], reverse=True)[:n])

    def vpChunking(self, para):
        grammar = "VP: {<VB.*> <DT>? <JJ>* <NN.*>+ <RB.?>?}"
        # grammar = "VP: {<DT>? <JJ>* <NN.*>+ <VB.*> <RB.?>?}"

        tags = pos_tag(word_tokenize(para))
        parser = RegexpParser(grammar)
        result = parser.parse(tags)
        for elem in result:
            if isinstance(elem, Tree):
                VPstr = ""
                for (text, tag) in elem:
                    if VPstr != "":
                        VPstr = VPstr + " " + text
                    else:
                        VPstr = VPstr + text
                if VPstr not in self.vpCounts:
                    self.vpCounts[VPstr] = 1
                else:
                    self.vpCounts[VPstr] += 1

    def getVpCounts(self, n):
        return dict(sorted(self.vpCounts.items(), key=lambda x: x[1], reverse=True)[:n])

    def ner(self, para):
        ne_tree = ne_chunk(pos_tag(word_tokenize(para)))
        for elem in ne_tree:
            if isinstance(elem, Tree):
                self.nerList[elem[0]] = elem.label()

        return self

    def getNer(self):
        return self.nerList

    def getImp(self):
        for key in self.wordCount:
            idf = math.log(24991/self.wordCount[key])
            tf = self.wordCount[key]/self.numWords
            self.wordimp[key] = tf * idf

        return self

    def topN(self, n=0):
        if n == 0:
            pass
#             n = len(self.getImp())
        self.getImp()
        return dict(sorted(self.wordimp.items(), key=lambda x: x[1], reverse=True)[:n])


file_path = "data.json"
output_file = "Output.txt"
f = open(output_file, "a")

df = pd.read_json(file_path, lines=True)
vocab = Vocabulary()
df.to_csv('data.csv', index=False)

df = df[df['reviewText'].notna()]

for para in tqdm(df['reviewText']):
    vocab.addPara(para)
    vocab.calNgrams(para, 3)
    vocab.posTagging(para)
    vocab.npChunking(para)
    vocab.vpChunking(para)
    vocab.ner(para)


# Module 1: Question 2

wordFreq = vocab.getWordsWithCounts()
f.write("Vocabulary -\n " + str(wordFreq) + " \n")

ngramsList = vocab.getNgrams()
f.write("Tri-grams -\n " + str(ngramsList) + " \n")

tagcounts = vocab.getTagCounts()
f.write("POS Tagging, Frequencies -\n " + str(tagcounts) + " \n")

npList = vocab.getNpCounts(5)
f.write("Noun phrases -\n " + str(npList) + " \n")

vpList = vocab.getVpCounts(5)
f.write("Verb phrases -\n " + str(vpList) + " \n")

nerList = vocab.getNer()
f.write("NERs -\n " + str(nerList) + " \n")


# Module 1: Question 3

top10 = vocab.topN(10)
topwords = vocab.topN()
f.write("Top words to describe corpus -\n" + str(top10) + "\n")


# Module 1: Question 4

# plt.figure(figsize=(10, 7))
# plt.ylabel("TF-IDF score")
# plt.xlabel("Words")
# x, y = zip(*topwords.items())
# plt.plot(x, y)
# plt.savefig("ImpWords.png")


# Module 1: Question 5

# plt.figure(figsize=(10, 7))
# plt.ylabel("Frequency")
# plt.xlabel("Words")
# x, y = zip(*wordFreq.items())
# plt.plot(x, y)
# plt.savefig("FreqVsRank.png")

rxf = {}
rank = 1
for key in wordFreq:
    rxf[key] = wordFreq[key] * rank
    rank += 1

# plt.figure(figsize=(10, 7))
# plt.ylabel("Frequency*Rank")
# plt.xlabel("Words")
# x, y = zip(*rxf.items())
# plt.plot(x, y)
# plt.savefig("FreqxRank.png")

f.close()


# In[3]:


# In[10]:


data_word = list(vocab.getWords())
data_words = []
data_words.append(data_word)


# In[71]:


# In[72]:


def remove_stopwords(texts):
    stop_words = set(stopwords.words("english"))
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


data_words_nostop = remove_stopwords(data_words)


# In[73]:


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append(
            [token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


nlp = spacy.load('en', disable=['parser', 'ner'])
data_lemmatized = lemmatization(data_words_nostop, allowed_postags=[
                                'NOUN', 'ADJ', 'VERB', 'ADV'])
print(data_lemmatized[:1])


# In[74]:


id2word = corpora.Dictionary(data_lemmatized)
texts = data_lemmatized
corpus = [id2word.doc2bow(text) for text in texts]


# In[85]:


lda_model2 = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, minimum_probability=0.01, minimum_phi_value=0.01,
                                             random_state=100, update_every=1, chunksize=100, passes=100, alpha='auto', per_word_topics=True)


# In[86]:


li = lda_model2.print_topics(200)
pprint.pprint(lda_model2.print_topics(100))


# In[64]:


pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
vis
