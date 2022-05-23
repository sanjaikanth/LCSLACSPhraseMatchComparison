#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from nltk.stem import PorterStemmer
from collections import Counter
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from stop_words import get_stop_words
from collections import Counter
from collections import OrderedDict


# In[2]:


def removeStopeWords(text):
    if text is None:
        return ""
     
    ps = PorterStemmer()
    counts = Counter()
    words = re.compile(r'\w+')
    counts.update(words.findall(text.lower()))

    stop_words1 = list(get_stop_words('en'))         #About 900 stopwords
    nltk_words = list(stopwords.words('english')) #About 150 stopwords
    stop_words1.extend(nltk_words)
    sw_list = ['The',',','.']
    stop_words1.extend(sw_list)
    #text=' '.join(first100)
    text = re.sub(r'[^\w\s]', '', text).lower()
    text_tokens = word_tokenize(text)
    tokens_without_sw = [word for word in text_tokens if not word in stop_words1]
    tokens_without_sw = [ps.stem(word) for word in tokens_without_sw]
    return " ".join(tokens_without_sw)


# In[3]:


def lcs(str_a, str_b):#, m, n):
    X=str_a.split(' ')
    Y=str_b.split(' ')
    m=len(str_a.split(' '))
    n=len(str_b.split(' '))
    L = [[0 for x in range(n+1)] for x in range(m+1)]
 
    # Following steps build L[m+1][n+1] in bottom up fashion. Note
    # that L[i][j] contains length of LCS of X[0..i-1] and Y[0..j-1]
    for i in range(m+1):
        for j in range(n+1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i-1] == Y[j-1]:
                L[i][j] = L[i-1][j-1] + 1
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1])
 
    # Following code is used to print LCS
    index = L[m][n]
 
    # Create a character array to store the lcs string
    lcs = [""] * (index+1)
    lcs[index] = ""
 
    # Start from the right-most-bottom-most corner and
    # one by one store characters in lcs[]
    i = m
    j = n
    while i > 0 and j > 0:
 
        # If current character in X[] and Y are same, then
        # current character is part of LCS
        if X[i-1] == Y[j-1]:
            lcs[index-1] = X[i-1]
            i-=1
            j-=1
            index-=1
 
        # If not same, then find the larger of two and
        # go in the direction of larger value
        elif L[i-1][j] > L[i][j-1]:
            i-=1
        else:
            j-=1
 
    #print (  " ".join(lcs))
    if lcs is None:
        return ""
    else:
        return " ".join(lcs).strip()
    #print ((lcs))
 
# Driver program
 

#str_a ="mango pet The address mentioned is for vaiga and last name is sanjaikanth so"#"A B C B D A B"# "xBCDxFGxxxKLMx"
#str_b ="my last  vaiga sanjaikanth mango and i petam 8"#"B D C A B A"# "aBCDeFGhijKLMn"
str_a ="hi mango pet The address mentioned is for vaiga and  name is  so"
str_b ="my vaiga mango and i petam 8"
m = len(str_a)
n = len(str_b)
#lcs(str_a.split(' '), str_b.split(' '))#, len(str_a.split(' ')), len(str_b.split(' ')))
lcs(str_a, str_b)#, len(str_a.split(' ')), len(str_b.split(' ')))


# In[14]:


def findPhraces(str_a, str_b):
    lstraces=[]

    strLCS=lcs(str_a, str_b)
    while (strLCS!=''):
        #print('LCS Values :',strLCS )  
        X=strLCS.split(' ')
        if(len(X)>=1):
            lstraces.append(X)
            for i in range(len(X)):
                strVal=X[i].strip()
                #lstraces.append(strVal)
                str_a=str_a.replace(strVal,'',1).replace('  ',' ').strip()
                str_b=str_b.replace(strVal,'',1).replace('  ',' ').strip()
                #print('New str_a: ',str_a) 
                #print('New str_b: ',str_b)
            strLCS=lcs(str_a, str_b)
        else :
            strLCS=''
    #print(lstraces)  
    return lstraces

#str_a ="mango pet The address mentioned is for vaiga and last name is sanjaikanth so"#"A B C B D A B"# "xBCDxFGxxxKLMx"
#str_b ="my last  vaiga sanjaikanth mango and i petam 8"#"B D C A B A"# "aBCDeFGhijKLMn"
#str_a ="get coronaviru longer need contact someon"
#str_b ="longer need contact someon get coronaviru"
#m = len(str_a)
#n = len(str_b)
#arrResponse=findPhraces(str_a, str_b)
#Count=0
#for line in arrResponse:
#    Count=Count+len(line)
#Count


# In[5]:


def findLACS(X,Y):
    X=[X.lower() for X in X]
    Y=[Y.lower() for Y in Y]
    X_as_set = set(X)
    intersection = X_as_set.intersection(Y)
    intersection_as_list = list(intersection)
    #print(intersection_as_list)
    return intersection_as_list


# In[6]:


def FindSimilarityPhrase(strX,strY):
    arrX= strX.split(' ')
    arrY= strY.split(' ')
    arrLACS=findLACS(arrX,arrY)
    countarrLACS=len(arrLACS)
    if countarrLACS==0:
        return 0
    lstPhraces=findPhraces(strX, strY)
    Count=0
    for line in lstPhraces:
        Count=Count+len(line)
    
    similarity = countarrLACS + 2*(len(lstPhraces)) + Count
    return similarity*100


# In[7]:


def getResult():
    #Loading Data
    df = pd.read_excel('SourceData2022.xlsx')
    #Loading Data
    dfTest = pd.read_excel('test.xlsx')
    dfTest["PredictedClass"]=""
    dfTest["MatchedIDInTraining"]=""
    dfTest["MatchedMessageInTraining"]=""
    dfTest["SimilarityScore"]=0
    for index, row in df.iterrows():
        df.at[index, "Message"]=removeStopeWords(df.at[index, "Message"])

    for index, row in dfTest.iterrows():
        dfTest.at[index, "Message"]=removeStopeWords(dfTest.at[index, "Message"])

    for indexTest, rowTest in dfTest.iterrows():
        SimilarityScore=0
        Class=""
        for index, row in df.iterrows():
            tweetTest=dfTest.at[indexTest, "Message"]
            tweetData=df.at[index, "Message"]
            if tweetData!=tweetTest:
                tmpSimilarityScore=FindSimilarityPhrase(tweetTest,tweetData)
                if SimilarityScore<tmpSimilarityScore:
                    SimilarityScore=tmpSimilarityScore
                    Class=df.at[index, "Class"]
                    dfTest.at[indexTest, "PredictedClass"]=Class
                    dfTest.at[indexTest, "MatchedIDInTraining"]=df.at[index, "ID"]
                    dfTest.at[indexTest, "MatchedMessageInTraining"]=df.at[index, "Message"]
                    dfTest.at[indexTest, "SimilarityScore"]=SimilarityScore

    for indexTest, rowTest in dfTest.iterrows():
        actualval=dfTest.at[indexTest, "Class"]
        predictedval=dfTest.at[indexTest, "PredictedClass"]
        isSame=actualval==predictedval
        dfTest.at[indexTest, "Result"]=isSame


    totalRealCount=dfTest[dfTest["Class"]=='Real'].shape[0] 
    totalXDisinformativeCount=dfTest[dfTest["Class"]=='Disinformation'].shape[0] 
    totalFakeCount=dfTest[dfTest["Class"]=='Fake'].shape[0] 
    totalMisInformativeCount=dfTest[dfTest["Class"]=='MisInformative'].shape[0] 
    totalUnverifiedCount=dfTest[dfTest["Class"]=='Unverified'].shape[0] 
    #print(totalRealCount,totalXDisinformativeCount,totalFakeCount,totalMisInformativeCount,totalUnverifiedCount,dfTest.shape[0])

    totalCorrectCount=dfTest[  (dfTest["Result"]==True)].shape[0]
    totalRealCorrectCount=dfTest[(dfTest["Class"]=='Real') & (dfTest["Result"]==True)].shape[0]
    totalDisinformativeCorrectCount=dfTest[(dfTest["Class"]=='Disinformation') & (dfTest["Result"]==True)].shape[0]
    totalFakeCorrectCount=dfTest[(dfTest["Class"]=='Fake') & (dfTest["Result"]==True)].shape[0]
    totalMisInformativeCorrectCount=dfTest[(dfTest["Class"]=='MisInformative') & (dfTest["Result"]==True)].shape[0]
    totalUnverifiedCorrectCount=dfTest[(dfTest["Class"]=='Unverified') & (dfTest["Result"]==True)].shape[0]
    #print(totalRealCorrectCount,totalDisinformativeCorrectCount,totalFakeCorrectCount,totalMisInformativeCorrectCount,totalUnverifiedCorrectCount,totalCorrectCount)

    dfSUmmary= pd.DataFrame( columns = ['Class','Total','PhraseMatchedCorrectPrediction','PhraseMatchedAccuracy' ])
    dr={'Class':'Disinformative','Total':totalXDisinformativeCount,'PhraseMatchedCorrectPrediction':totalDisinformativeCorrectCount,'PhraseMatchedAccuracy':round(100*totalDisinformativeCorrectCount/totalXDisinformativeCount) }
    dfSUmmary=dfSUmmary.append(dr,ignore_index = True ) 
    dr={'Class':'Real','Total':totalRealCount,'PhraseMatchedCorrectPrediction':totalRealCorrectCount,'PhraseMatchedAccuracy':round(100*totalRealCorrectCount/totalRealCount) }
    dfSUmmary=dfSUmmary.append(dr,ignore_index = True ) 
    dr={'Class':'Fake','Total':totalFakeCount,'PhraseMatchedCorrectPrediction':totalFakeCorrectCount,'PhraseMatchedAccuracy':round(100*totalFakeCorrectCount/totalFakeCount) }
    dfSUmmary=dfSUmmary.append(dr,ignore_index = True ) 
    dr={'Class':'MisInformative','Total':totalMisInformativeCount,'PhraseMatchedCorrectPrediction':totalMisInformativeCorrectCount,'PhraseMatchedAccuracy':round(100*totalMisInformativeCorrectCount/totalMisInformativeCount) }
    dfSUmmary=dfSUmmary.append(dr,ignore_index = True ) 
    dr={'Class':'Unverified','Total':totalUnverifiedCount,'PhraseMatchedCorrectPrediction':totalUnverifiedCorrectCount,'PhraseMatchedAccuracy':round(100*totalUnverifiedCorrectCount/totalUnverifiedCount) }
    dfSUmmary=dfSUmmary.append(dr,ignore_index = True )
    dr={'Class':'Total','Total':dfTest.shape[0],'PhraseMatchedCorrectPrediction':totalCorrectCount,'PhraseMatchedAccuracy':round(100*totalCorrectCount/dfTest.shape[0]) }
    dfSUmmary=dfSUmmary.append(dr,ignore_index = True )
    dfSUmmary.to_csv('PhraseResult.csv',index=False)

