#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on Fri Oct 20 13:33:49 2017

@author: harshat,nanthini b
"""
import pandas as pd
import string
from nltk.probability import FreqDist
import numpy as np
from math import sqrt
from numpy import linalg as LA
import random
import time
def testerror(w):
    M = len(test_taggedFeatures)
    c = 0
    for i in range(M):
        y,line = test_taggedFeatures[i]
        count = FreqDist(line)
        y = (int)(y)
        count = dict(count)
        nzi = []
        nzv = []
               #Get the non zero indices and values
        for k in count:
            try:
                nzi.append(indexLookUp[k])
                nzv.append(count[k])
            except KeyError,e:
                continue
        score  = 0
        for j in range(len(nzi)):
            score+= (w[0,nzi[j]] * nzv[j])
        if score>=0:
            predY = 1
        else:
            predY = -1
        #print i," : ",y,predY    
        if predY!=y:
            c+=1
    error = float(c)/M*100
    return error
def trainerror(w):
    M=100000
    c = 0
    for i in range(M):
        instance = random.sample(taggedFeatures,1)
        y = None
        line = None
        for yDash, l in instance:
            y = int(yDash)
            line = l
        count = FreqDist(line)
        count = dict(count)
        nzi = []
        nzv = []
        #Get the non zero indices and values
        for k in count:
            try:
                nzi.append(indexLookUp[k])
                nzv.append(count[k])
            except KeyError,e:
                continue

        score  = 0
        for j in range(len(nzi)):
            score+= (w[0,nzi[j]] * nzv[j])
        if score>=0:
            predY = 1
        else:
            predY = -1
        if predY!=y:
            c+=1
    error = float(c)/M*100
    return error

def getLookUpTable():
   indexLookUp = dict()
   corpus = list(bag_of_words)
   for index, elem in enumerate(corpus):
       indexLookUp[elem] = index
   return indexLookUp

def getTaggedFeatures():
    '''

   Objective: Generates the Bag of wrods for a given document.

   Tasks Performed:
       (A) Data cleaning:
           -1- Removing "http" and "www" words from the tweets
           -2- Removing all @'s
           -3- Removes redundant spaces
           -4- Removes the stopWords
       (B) Generate Bag of Words Model

   Returns bag_of_words for the given document corpus

    '''
    processed_data = []


    for (senti,line) in data:
        processed_data.append((senti,line.lower()))
    taggedData = np.array(processed_data)
    taggedFeatures = []
    for (senti,test) in taggedData:
        #test = taggedData[k,1]
        temp = list(set(test.split()))
        test = ' '.join(temp)
        Bad = []
        Bad2 = []
        Bad = [word for word in temp if (word.startswith('http') or word.startswith('www.'))] #removing all links
        Bad2 = [word for word in temp if (word.startswith('@'))] #removing at users
        for i in range(np.shape(Bad)[0]):
            test = test.replace(Bad[i], 'URL')
        for i in range(np.shape(Bad2)[0]):
            test = test.replace(Bad2[i],'AT-USER')
        test = test.translate(None, string.punctuation) #Removing punctuation
        test = ' '.join(test.split())
        testS = set(test.split()) # Removing repetitions
        testS = testS.difference(StopwordsSet)
        taggedFeatures.append((int(senti),testS))
    return taggedFeatures


def pegasos(T=100):
   testErrorPeg = []
   trainErrorPeg = [] 
   iterationsPeg = []    
   regLambda = 0.01
   start_t = time.time()
   w = np.zeros([1,len(bag_of_words)])
   tmp = np.random.rand(1,len(bag_of_words))
   w[0] = tmp[0]
   norm = np.linalg.norm(w[0])
   w[0] = w[0]/norm
   N = 100
   w[0] = w[0]/sqrt(regLambda)
   for t in range(1,T):
       
       #finalSenti = []

       eta = 1/(regLambda * t)
       summation = np.zeros([1,len(bag_of_words)])
       #Subset of samples
       for i  in range(N):
           instance = random.sample(taggedFeatures,1)
           y = None
           line = None
           for yDash, l in instance:
               y = yDash
               line = l
           count = FreqDist(line)
           count = dict(count)
           nzi = []
           nzv = []
           #Get the non zero indices and values
           '''
           Here we are interested only in the non zero values stored in the
           feature vector. Since it is a sparse vector we do not want to perform a 
           dot product with the np.dot since it was too time consuming.
           
           What we use here is the property of dot product which is basically
           the fact that each element is multiply with the corresponding element 
           in the second vector and a sum of all these products is taken.
           
           Thus, it prevents us from having to store really sparse np array which were 
           extremely inefficent.
           '''
           for k in count:
               try:
                   nzi.append(indexLookUp[k])
                   nzv.append(count[k])
               except KeyError,e:
                   continue
           score  = 0
           for j in range(len(nzi)):
               score+= (w[0,nzi[j]] * nzv[j])
           score*=y
           if score<1:
               for j in range(len(nzi)):
                   summation[0,nzi[j]]+= y * nzv[j]
       grad = regLambda * w - (1./N) * (summation)
       wDash = w - eta*grad
       w = min(1,((1/sqrt(regLambda))/LA.norm(wDash))) * wDash
       if t!=0 and t%1000==0:
           iterationsPeg.append(t)
           trainErrorPeg.append(trainerror(w))
           testErrorPeg.append(testerror(w))
           print t," : \n(pegasos)\ntrain: ",trainErrorPeg,"\ntest: ",testErrorPeg
   print "Pegasos ",T," : ",time.time()-start_t   
   return w
def adagrad(T=100):
   trainErrorAda = []
   testErrorAda = []
   iterationsAda = []
   regLambda = 0.01
   start_t = time.time()
   w = np.zeros([1,len(bag_of_words)])
   G_total = np.ones([1,len(bag_of_words)])
   N = 100

   for t in range(1,T):       
       eta = 1/(regLambda * t)
       #Subset of samples
       summation = np.zeros([1,len(bag_of_words)])
       for i  in range(N):
           
           instance = random.sample(taggedFeatures,1)
           y = None
           line = None
           for yDash, l in instance:
               y = yDash
               line = l
           count = FreqDist(line)
           count = dict(count)
           nzi = []
           nzv = []
           #Get the non zero indices and values
           for k in count:
               try:
                   nzi.append(indexLookUp[k])
                   nzv.append(count[k])
               except KeyError,e:
                   continue
           score  = 0
           for j in range(len(nzi)):
               score+= (w[0,nzi[j]] * nzv[j])
           score*=y
           if score<1:
               for j in range(len(nzi)):
                   summation[0,nzi[j]]+= y * nzv[j]
       grad = regLambda*w - (1./N) * (summation)
       G_total = G_total + np.square(grad)
       G = np.sqrt(G_total)
       Ginv = (1. / ((G)))
       wDash = w - eta*np.multiply(Ginv,grad)
       w = min(1,((1/sqrt(regLambda))/LA.norm(np.multiply(G,wDash)))) * wDash
       if t!=0 and t%1000==0:
           iterationsAda.append(t)
           trainErrorAda.append(trainerror(w))
           testErrorAda.append(testerror(w))
           print t," : \n(adagrad)\ntrain: ",trainErrorAda
           print "test: ",testErrorAda
   print "Adagrad for T = ",T," : ",time.time()-start_t  
   return w

#Change name back before submitting
df = pd.read_csv("data/training.1600000.processed.noemoticon.csv",usecols=[0,5],header = None)

#Preprocessing
data = np.array(df)
data[data==0] = -1
data[data==4] = 1

taggedFeatures = []
with open('data/stopwords.txt') as f:
   customStopwords = f.read().splitlines()
StopwordsSet = set(customStopwords)
StopwordsSet.add('ATUSER') # We are adding this because after removing punctuations '-' gets removed
#Question 1.2 and 1.3 Generating Bag of Words
print "Getting tagged features..."
taggedFeatures = getTaggedFeatures()
bag_of_words = set()


for (senti,line) in taggedFeatures:
    for word in line:
        bag_of_words.add(word)
print "Generating bag of words..."

'''
We are using a set representaion for the bag of words in order to avoid repetitions of words

We then use a index look up dictionary to account for the indices in the feature vector that need
to be filled with the frequency count.

'''
bag_of_words = list(bag_of_words)
#Create an index lookup for the words
indexLookUp = getLookUpTable()


print "Loading test file..."
'''

The test file is loaded and the tweets in the file are cleaned in the same way that training data was
and the w obtained from the pegasos and adagrad are used to get the error

'''

testdf = pd.read_csv("data/testdata.manual.2009.06.14.csv",usecols=[0,5],header = None)
testData = np.array(testdf)
testData[testData==0] = -1
testData[testData==4] = 1
testData[testData==2] = 1
test_taggedFeatures = []
test_processed_data = []

for (senti,line) in testData:
    test_processed_data.append((senti,line.lower()))
test_taggedData = np.array(test_processed_data)
for (senti,test) in (test_taggedData):
    temp = list(set(test.split()))
    test = ' '.join(temp)
    Bad = []
    Bad2 = []
    Bad = [word for word in temp if (word.startswith('http') or word.startswith('www.'))]
    Bad2 = [word for word in temp if (word.startswith('@'))]
    for i in range(np.shape(Bad)[0]):
        test = test.replace(Bad[i], 'URL')
    for i in range(np.shape(Bad2)[0]):
        test = test.replace(Bad2[i],'AT-USER')
    test = test.translate(None, string.punctuation)
    test = ' '.join(test.split())
    testS = set(test.split())
    testS = list(testS.difference(StopwordsSet))
    test_taggedFeatures.append(((int)(senti),testS))

pegasos(10000)
adagrad(10000)