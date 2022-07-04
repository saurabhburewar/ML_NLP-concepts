import json
from operator import index
from nltk import ngrams, pos_tag, PorterStemmer, WordNetLemmatizer
from nltk import RegexpParser, Tree, ne_chunk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import pandas as pd
import re
import math
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm

nltk.download('popular')


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

    def lemmatize_stemming(self, para):
        return PorterStemmer().stem(WordNetLemmatizer().lemmatize(para))

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
        para = self.lemmatize_stemming(para)
        tokenizer = RegexpTokenizer(r'\w+')
        para = tokenizer.tokenize(para)
        for word in para:
            if word in self.stopwords:
                para.remove(word)
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
        tokenizer = RegexpTokenizer(r'\w+')
        para = self.lemmatize_stemming(para)
        para = tokenizer.tokenize(para)
        for word in para:
            if word in self.stopwords:
                para.remove(word)
        nGrams = ngrams(para, n)
        for gram in nGrams:
            gramcomp = ' '.join(gram)
            if gramcomp not in self.Ngrams:
                self.Ngrams.append(gramcomp)

        return self

    def getNgrams(self):
        return self.Ngrams

    def posTagging(self, para):
        tokenizer = RegexpTokenizer(r'\w+')
        para = self.lemmatize_stemming(para)
        para = tokenizer.tokenize(para)
        for word in para:
            if word in self.stopwords:
                para.remove(word)
        tags = pos_tag(para)
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

        tokenizer = RegexpTokenizer(r'\w+')
        para = self.lemmatize_stemming(para)
        para = tokenizer.tokenize(para)
        for word in para:
            if word in self.stopwords:
                para.remove(word)
        tags = pos_tag(para)
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

        tokenizer = RegexpTokenizer(r'\w+')
        para = self.lemmatize_stemming(para)
        para = tokenizer.tokenize(para)
        for word in para:
            if word in self.stopwords:
                para.remove(word)
        tags = pos_tag(para)
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
        tokenizer = RegexpTokenizer(r'\w+')
        para = self.lemmatize_stemming(para)
        para = tokenizer.tokenize(para)
        for word in para:
            if word in self.stopwords:
                para.remove(word)
        ne_tree = ne_chunk(pos_tag(para))
        for elem in ne_tree:
            if isinstance(elem, Tree):
                self.nerList[elem[0][0]] = elem.label()

        return self

    def getNer(self):
        return dict(self.nerList)

    def getImp(self):
        for key in self.wordCount:
            idf = math.log(24991/self.wordCount[key])
            tf = self.wordCount[key]/self.numWords
            self.wordimp[key] = tf * idf

        return self

    def topN(self, n=0):
        if n == 0:
            n = len(self.wordimp)
        self.getImp()
        return dict(sorted(self.wordimp.items(), key=lambda x: x[1], reverse=True)[:n])


file_path = "data.json"
output_file = "Output1.txt"
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

print(vocab.numWords)

# Module 1: Question 2

wordFreq = vocab.getWordsWithCounts()
wordFreqstr = json.dumps(wordFreq)
f.write("Vocabulary -\n")
f.write(wordFreqstr)
f.write("\n\n")

ngramsList = vocab.getNgrams()
ngramsListstr = json.dumps(ngramsList)
f.write("Tri-grams -\n")
f.write(ngramsListstr)
f.write("\n\n")

tagcounts = vocab.getTagCounts()
tagcountsstr = json.dumps(tagcounts)
f.write("POS Tagging, Frequencies -\n")
f.write(tagcountsstr)
f.write("\n\n")

npList = vocab.getNpCounts(5)
npListstr = json.dumps(npList)
f.write("Noun phrases -\n")
f.write(npListstr)
f.write("\n\n")

vpList = vocab.getVpCounts(5)
vpListstr = json.dumps(vpList)
f.write("Verb phrases -\n")
f.write(vpListstr)
f.write("\n\n")

nerList = vocab.getNer()
nerListstr = json.dumps(nerList)
f.write("NERs -\n")
f.write(nerListstr)
f.write("\n\n")


# Module 1: Question 3

top500 = vocab.topN(500)
topwords = vocab.topN()
top500str = json.dumps(top500)
f.write("Top words to describe corpus -\n")
f.write(top500str)
f.write("\n\n")


# Module 1: Question 4

plt.figure(figsize=(10, 7))
plt.ylabel("TF-IDF score")
plt.xlabel("Words")
x, y = zip(*top500.items())
plt.plot(x, y)
plt.savefig("ImpWords.png")


# Module 1: Question 5

wordFreqPlot = {}

rxf = {}
rank = 1
for key in wordFreq:
    if rank < 5000:
        wordFreqPlot[key] = wordFreq[key]
    rxf[key] = wordFreq[key] * rank
    rank += 1

plt.figure(figsize=(10, 7))
plt.ylabel("Frequency")
plt.xlabel("Words")
x, y = zip(*wordFreqPlot.items())
plt.plot(x, y)
plt.savefig("FreqVsRank.png")

rxfstr = json.dumps(rxf)
f.write("RankxFreq -\n")
f.write(rxfstr)
f.write("\n\n")

# plt.figure(figsize=(10, 7))
# plt.ylabel("Frequency*Rank")
# plt.xlabel("Words")
# x, y = zip(*rxf.items())
# plt.plot(x, y)
# plt.savefig("FreqxRank.png")

f.close()
