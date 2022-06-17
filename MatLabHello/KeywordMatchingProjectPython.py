#!/usr/bin/env python
# coding: utf-8

# In[7]:


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


# In[8]:


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


# In[9]:


def findKeywordMatch(X,Y):
    X=[X.lower() for X in X]
    Y=[Y.lower() for Y in Y]
    X_as_set = set(X)
    intersection = X_as_set.intersection(Y)
    intersection_as_list = list(intersection)
    #print(intersection_as_list)
    return intersection_as_list


# In[10]:


def FindSimilarity(strX,strY):
    arrX= strX.split(' ')
    arrY= strY.split(' ')
    arrKeyword=findKeywordMatch(arrX,arrY)
    arrKeyword=len(arrKeyword)    
    return arrKeyword*100


# In[11]:


def PlotChart(dfTest):
    countTrue=0
    countFalse=0
    CountTotal=0
    arrPercentValues=[]
    arrCount=[]
    for indexTest, rowTest in dfTest.iterrows():
        isTrue= 1 if rowTest.Result else 0
        isFalse= 0 if rowTest.Result else 1
        countTrue=countTrue+isTrue
        countFalse=countFalse+isFalse
        intPercentage=countTrue*100/(indexTest+1)
        arrPercentValues.append(intPercentage)
        arrCount.append(indexTest+1)
    #    print(PercentValues)
    #plt.plot(arrCount, arrPercentValues)  
    #plt.title('Percentage True')
    #plt.show()
    #dfSUmmary.to_csv('LCSResult.csv',index=False)
    df = pd.DataFrame({"Index" : arrCount, "Percent" : arrPercentValues})
    df.to_csv("PlotKeywordMatching.csv", index=False)


# In[12]:


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
                tmpSimilarityScore=FindSimilarity(tweetTest,tweetData)
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

    dfSUmmary= pd.DataFrame( columns = ['Class','Total','KeywordMatching Correct Prediction','KeywordMatching Accuracy' ])
    dr={'Class':'Disinformative','Total':totalXDisinformativeCount,'KeywordMatching Correct Prediction':totalDisinformativeCorrectCount,'KeywordMatching Accuracy':round(100*totalDisinformativeCorrectCount/totalXDisinformativeCount) }
    dfSUmmary=dfSUmmary.append(dr,ignore_index = True ) 
    dr={'Class':'Real','Total':totalRealCount,'KeywordMatching Correct Prediction':totalRealCorrectCount,'KeywordMatching Accuracy':round(100*totalRealCorrectCount/totalRealCount) }
    dfSUmmary=dfSUmmary.append(dr,ignore_index = True ) 
    dr={'Class':'Fake','Total':totalFakeCount,'KeywordMatching Correct Prediction':totalFakeCorrectCount,'KeywordMatching Accuracy':round(100*totalFakeCorrectCount/totalFakeCount) }
    dfSUmmary=dfSUmmary.append(dr,ignore_index = True ) 
    dr={'Class':'MisInformative','Total':totalMisInformativeCount,'KeywordMatching Correct Prediction':totalMisInformativeCorrectCount,'KeywordMatching Accuracy':round(100*totalMisInformativeCorrectCount/totalMisInformativeCount) }
    dfSUmmary=dfSUmmary.append(dr,ignore_index = True ) 
    dr={'Class':'Unverified','Total':totalUnverifiedCount,'KeywordMatching Correct Prediction':totalUnverifiedCorrectCount,'KeywordMatching Accuracy':round(100*totalUnverifiedCorrectCount/totalUnverifiedCount) }
    dfSUmmary=dfSUmmary.append(dr,ignore_index = True )
    dr={'Class':'Total','Total':dfTest.shape[0],'KeywordMatching Correct Prediction':totalCorrectCount,'KeywordMatching Accuracy':round(100*totalCorrectCount/dfTest.shape[0]) }
    dfSUmmary=dfSUmmary.append(dr,ignore_index = True )
    dfSUmmary.to_csv('KeywordMatchingResult.csv',index=False)
    PlotChart(dfTest)
 

 

