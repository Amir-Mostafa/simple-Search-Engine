import nltk
#tokenizer
from nltk.tokenize import sent_tokenize, word_tokenize
#steming
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer

#cosien similarity
from scipy import spatial
##crawling
import requests
from bs4 import BeautifulSoup

#TFIDF
from sklearn.feature_extraction.text import TfidfVectorizer
# PANDA
import pandas as pd

#numpy
import numpy as np

#regular expression
import re
import math 

porter = PorterStemmer()
lancaster=LancasterStemmer()
stopWords=stopwords=nltk.corpus.stopwords.words('Indonesian')

def stemSentence(sentence):
    token_words=word_tokenize(sentence)
    ##token_words
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(porter.stem(word))
        #stem_sentence.append(" ")
    return stem_sentence


def removeStoWords(txt):
    
    result=[]
    for word in txt:
        if word not in stopWords:
            result.append(word)
            
    res=" ";
    return (res.join([str(elem) for elem in result]))
    

def get_similar_documents(q, df):
  
  # query to vector
  q = [q]
  q_vec = vectorizer.transform(q).toarray().reshape(df.shape[0],)
  sim = {}
  # Calculate the similarity
  for i in range(10):
    #sim[i] = np.dot(df.loc[:, i].values, q_vec) / np.linalg.norm(df.loc[:, i]) * np.linalg.norm(q_vec)
    sim[i]=1 - spatial.distance.cosine(df.loc[:, i].values, q_vec)
  
  # Sort the values 
  sim_sorted = sorted(sim.items(), key=lambda x: x[1], reverse=True)
  # Print the articles and their similarity values
  for k, v in sim_sorted:
    if v != 0.0:
      print(" Similaritas:", v)
      print(documents[k])
      print(link[k])
      print()
      


#request to the website
r = requests.get('https://bola.kompas.com/')

#parse the HTML format
soup = BeautifulSoup(r.content, 'html.parser')

# Retrieve all links
link = []
for i in soup.find('div', {'class':'most__wrap'}).find_all('a'):
    link.append(i['href'])
    
documents = []
for i in link:
    #request to the link
    r = requests.get(i)
    soup = BeautifulSoup(r.content, 'html.parser')
  
    #all paragraphs
    sen = []
    for i in soup.find('div', {'class':'read__content'}).find_all('p'):
        sen.append(i.text)
  
    # Add the combined paragraphs to documents
    documents.append(' '.join(sen))


documents_clean = []
for d in documents:
    # Remove Unicode
    document_test = re.sub(r'[^\x00-\x7F]+', ' ', d)

    # Lowercase the document
    document_test = document_test.lower()
  
    # Lowercase the numbers
    document_test = re.sub(r'[0-9]', '', document_test)
    
    # Remove the doubled space
    document_test = re.sub(r'\s{2,}', ' ', document_test)
    
    #steming
    document_test=stemSentence(document_test)
    
    #remove stop word
    document_test=removeStoWords(document_test)

    documents_clean.append(document_test)

# Instantiate
vectorizer = TfidfVectorizer()
# as a vector
X = vectorizer.fit_transform(documents_clean)

names=vectorizer.get_feature_names()
##convert matrix to array
X=X.T.toarray()

# Create a DataFrame 
df = pd.DataFrame(X, index=names)

print(df)

# Add The Query
q1 = 'pelatih diego simeone buka suara setel'

# Call the function

get_similar_documents(q1, df)












# def computeTF(wordDict, doc):
#     tfDict = {}
#     corpusCount = len(doc)
#     for word, count in wordDict.items():
#         tfDict[word] = count/float(corpusCount)
#     return(tfDict)

# def computeIDF(docList):
#     idfDict = {}
#     N = len(docList)
    
#     idfDict = dict.fromkeys(docList[0].keys(), 0)
#     for word, val in idfDict.items():
#         idfDict[word] = math.log10(N / (float(val) + 1))
        
#     return(idfDict)


# def computeTFIDF(tfBow, idfs):
#     tfidf = {}
#     for word, val in tfBow.items():
#         tfidf[word] = val*idfs[word]
#     return(tfidf)






 


