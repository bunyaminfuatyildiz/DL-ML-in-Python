# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 14:21:56 2020

@author: bunya
"""
import os
os.chdir("C:\\Users\\bunya\\python")
from egeaML import DataIngestion, nlp
import egeaML
import csv
import re
import string


import nltk
import gensim

#gensim
from gensim.parsing.preprocessing import strip_non_alphanum
from gensim.parsing.preprocessing import strip_punctuation
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import strip_multiple_whitespaces
from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import simple_preprocess
from gensim.corpora.dictionary import Dictionary
from gensim import models, utils, matutils
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models.doc2vec import TaggedDocument

#nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import regexp_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
nltk.download('wordnet')














import nltk
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
from nltk.tokenize import sent_tokenize





mystr = "I haven't been to Rome (last year)-that's amazing!"
tok_egeaML = nlp.simple_tokenization(mystr)
tok_nltk = word_tokenize(mystr)
tok_gensim = simple_preprocess(mystr)
print('Original document: ', mystr)
print('Tokenized list using the egeaML library: ', tok_egeaML)
print('Tokenized list using the nltk library: ', tok_nltk)
print('Tokenized list using the gensim library: ', tok_gensim)


'''import a simple .txt document containing two small articles concerning a deep learning technique called Convolutional Neural Network.'''
with open('article.txt') as f: # enter them one by one
    reader = csv.reader(f)
    csv_rows = list(reader)
text = ""
for i in range(len(csv_rows[0])):
    text += csv_rows[0][i]
text = text.split('\\n')
#show what sent_tokenization performs, we use the ﬁrst article as an example:
sentences = sent_tokenize(text[0])
print(sentences)
    
''' book-speciﬁc class nlp, which has a method called parsing_text that basically removes the most common english stopwords, punctuactions and leading and trailing spaces.'''
print(nlp.parsing_text(sentences[2]))  


'''Alternatively, one might deﬁne a a set of words that has to be removed before lem-
mization: this is typically made by all stopwords and punctuations.  
'''

punct = set(string.punctuation)
stop = set(stopwords.words('english'))
stop.add('to')
   
   
'''we use the book-speciﬁc method
clean_text from the egeaML library, which performs the following steps:
1. It ﬁrstly performs simple tokenization;
2. If the token is not a stopwords or its length is smaller than three, then we
perform lemmatization and stemmization on that token;
3. else, it is removed.'''

doc_sample = text[0]
print('Original document: \n')
words = []
for word in doc_sample.split(' '):
    words.append(word)
print(words)
print('\n Tokenized and lemmatized document: \n')
print(nlp().clean_text(doc_sample))



doc = [nlp().clean_text(x) for x in text]  

 #show how the TweetTokenizer works, let’s take, for instance, a simple tweet:
tweet = 'I used a kernelized SVM to classify text. I love learning new #NLP techniques using #python! @someone #NLP is real fun! :-) #ml #NLP #python'   
tknzr = TweetTokenizer()
tokens = tknzr.tokenize(tweet)


regex = r"#\w+"
list(set(regexp_tokenize(tweet, regex)))



## Numerical Representation of Documents: the Bag-of-Words      
print(doc[0])

# create a numerical representation of that list. This is easily implemented in gensim, as follows      
dictionary = Dictionary(doc)
print(dictionary.doc2idx(doc[0]))    

mylist = list()
for k,v in dictionary.token2id.items():
    mylist.append(k)

doc2freq = pd.DataFrame(matutils.corpus2dense(corpus,
num_terms=len(dictionary.token2id)),
index = mylist,
columns=['Doc1', 'Doc2'])

doc2freq.T.iloc[:,10:20]



tf_sparse_array = matutils.corpus2csc(corpus)
tf_sparse_array





vectorizer = CountVectorizer(analyzer='word',
min_df=2,
stop_words='english',
lowercase=True,
token_pattern='[a-zA-Z0-9]{2,}',
)
data_vectorized = vectorizer.fit_transform(text)
data_dense = data_vectorized.todense()

    
    

