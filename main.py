# Comp-6321 Project
# Analysing Suspicious URLs using Machine Learning Algorithms
# Name: Mandeep Kaur
# Student ID: 40059801

import ipaddress as IP
import pandas as pd
import numpy as np
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
import tldextract
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
import whois
import requests
from bs4 import BeautifulSoup

from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import re
import pygeoip
nf = -1


# url dataset
dataset = pd.read_csv("input_dataset.csv")

#HOST BASED CHANGES START
def getRequestedUrlInfo(url, allow_redirects=False):
    try:
        requestedPage = requests.get(url,allow_redirects)
        return BeautifulSoup(requestedPage.content, 'html.parser')
    except requests.exceptions.ConnectionError:
        return None
    except:
        return nf

def getNoOfIframeCount(soup):
    zeroSizeiFrameCount = 0
    hiddeniFrameCount = 0
    try:
        iframeList = soup.find_all('iframe')
        for iFrame in iframeList:
            if str(iFrame.get('height')) == '0' or str(iFrame.get('width')) == '0':
                zeroSizeiFrameCount = zeroSizeiFrameCount + 1
            if 'visibility:hidden' in str(iFrame.get('style')).replace(" ", ""):
                hiddeniFrameCount = hiddeniFrameCount + 1
        return [zeroSizeiFrameCount, hiddeniFrameCount]
    except:
        return [nf, nf]

def getNoOfSuspectedJavaScriptMethodCount(soup):
    suspectedMethodFoundCount=0
    suspectedMethodList=[ 'unescape(', 'search(', 'link(','escape(', 'eval(', 'exec(']
    try:
        suspectedScriptList = soup.find_all('script')
        for suspectedScript in suspectedScriptList:
            for suspectedMethod in suspectedMethodList:
                if suspectedMethod in str(suspectedScript):
                    suspectedMethodFoundCount = suspectedMethodFoundCount + str(suspectedScript).count(suspectedMethod)
        return suspectedMethodFoundCount
    except:
        return nf

def getInvalidHtmlTagsCount(soup):
    return len(soup.find_all('html'))

def getNoOfAnchorTags(soup):
    return len(soup.find_all('a'))

#HOST BASED ENDS

# Method to calculate average no. of tokens
def get_Tokens_Count(url):
    token_list = []
    num_of_tokens =0
    if url=='':
        return 0
    token_list=re.split('\W+',url)
    for token in token_list:
        length = len(token)
        if length >0:
            num_of_tokens = num_of_tokens+1
    return num_of_tokens

# Method to calculate average no. of tokens
def get_AverageTokens(url):
    token_list = []
    num_of_tokens=0
    sum =0
    if url=='':
        return 0
    token_list=re.split('\W+',url)
    for token in token_list:
        length = len(token)
        sum +=1
        if length >0:
            num_of_tokens = num_of_tokens+1
    return float(sum)/num_of_tokens

# Method to check presence of '.'
def Dot_count(url):
    num = 0
    match = '.'
    for key in url:
        if key == match:
            num += 1
    return num

# Method to check presence of '-'
def Hyphen(url):
    num = 0
    match = '-'
    for key in url:
        if key == match:
            num += 1
    return num

# Method to check presence of '?'
def occurence_of_Qmark(url):
    num = 0
    match = '?'
    for key in url:
        if key == match:
            num += 1
    return num

# Method to check presence of '@'
def occurance_Of_At(url):
    num = 0
    match = '@'
    for key in url:
        if key == match:
            num += 1
    return num

# This method is for validating whether IP address is present as a hostname
def presence_of_IPaddress(url):
    try:
        if IP.ip_address(url):
            return 1
    except:
        return 0
# Method to get Autonomous System Number from host
def get_ASN(host):
    try:
        g = pygeoip.GeoIP('GeoIPASNum.dat')
        asn=int(g.org_by_name(host).split()[0][2:])
        return asn
    except:
        return  nf

# Method to get all the features from url dataset
def get_AllFeatures(url,label):
    url = str(url)
    Feature_Set = []
    # Parsing url for extracting url information
    obj = urlparse(url)
    host = obj.netloc
    ext = tldextract.extract(url)
    # adding url to feature set
    Feature_Set.append(url)
    # adding length of url to feature set
    Feature_Set.append(len(url))
    # adding count of dots
    Feature_Set.append(Dot_count(url))
    # adding number of tokens present in url
    Feature_Set.append(get_Tokens_Count(url))
    # adding average number of tokens present in url
    Feature_Set.append(get_AverageTokens(url))
    # adding number of tokens present in host
    Feature_Set.append(get_Tokens_Count(host))
    # adding average number of tokens present in host
    Feature_Set.append(get_AverageTokens(host))
    # adding number of hyphens present in url
    Feature_Set.append(Hyphen(url))
    # adding number of Question mark present in url
    Feature_Set.append(occurence_of_Qmark(url))
    # adding number of '@' At symbol present in url
    Feature_Set.append(occurance_Of_At(url))
    # Check whecther IP address is present in url
    Feature_Set.append(presence_of_IPaddress(ext.domain))
    # adding length of host
    Feature_Set.append(len(host))
    # adding ASN
    Feature_Set.append(get_ASN(host))
    soup=getRequestedUrlInfo(url)
    if soup is None:
        Feature_Set.append(nf)
        Feature_Set.append(nf)
        Feature_Set.append(nf)
        Feature_Set.append(nf)
        Feature_Set.append(nf)
    else:
        [heightWidthNullCount, iframeInvisibleCount]=getNoOfIframeCount(soup)
        Feature_Set.append(heightWidthNullCount)
        Feature_Set.append(iframeInvisibleCount)
        Feature_Set.append(getNoOfSuspectedJavaScriptMethodCount(soup))
        Feature_Set.append(getInvalidHtmlTagsCount(soup))
        Feature_Set.append(getNoOfAnchorTags(soup))

    # adding label
    Feature_Set.append(str(label))
    return Feature_Set

# Creating DataFrame - 'feature_data' to store Url features
feature_data = pd.DataFrame(columns=('url','URL_length','Num_ofDots','Token_Count','Avg_TokenCount','Domain_TokenCnt','Avg_Domain TokenLength','No_of Hyphen','No_of Question Mark','No_of At','Presence_IP','Host_len','ASN','iframe_dimen','iframe_visibility','Suspected_func_usage','Html_tagCnt','Link_cnt','label'))

# Adding features to Dataframe
for i in range(len(dataset)):
    output = get_AllFeatures(dataset["url"].loc[i],dataset["label"].loc[i])
    feature_data.loc[i] = output
    feature_data.to_csv('url_features.csv')

#uncomment below line to execute only the classification algorithm and comment the above code
# feature_data = pd.read_csv("url_features.csv")

# Input X - collected from featured data
X = feature_data.drop(['url','label'],axis=1).values
# Input Y - contains 0 and 1 values
Y = feature_data['label'].values

# Getting training and testing data
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2)

# Using Logistic Regression Algorithm
classifier_lgs = LogisticRegression()
classifier_lgs.fit(X_train, y_train)
print("Accuracy for Logistic Regression: ", end="")
# Accuracy of the classifier
accuracy_lgs = classifier_lgs.score(X_test, y_test) * 100
print(accuracy_lgs)

# Using Linear Support vector machine Alogrithm
classifier_svm = SVC()
classifier_svm.fit(X_train,y_train)
print("Accuracy for SVM: ",end="")
# Accuracy of the classifier
accuracy_svm = classifier_svm.score(X_test,y_test)*100
print(accuracy_svm)

# Using Perceptron
classifier_per = Perceptron()
classifier_per.fit(X_train,y_train)
print("Accuracy for perceptron: ",end="")
# Accuracy of the classifier
accuracy_per = classifier_per.score(X_test,y_test)*100
print(accuracy_per)


# Graph Plot - Showing comparison of accuracy of both classifiers
x_axis = ('Logistic','SVM','Perceptron')
y_axis = [accuracy_lgs,accuracy_svm,accuracy_per]
graph = plt.bar( x_axis,y_axis,width= 0.3,color= ('deepskyblue', 'springgreen','sandybrown'))
for rect in graph:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2.0, height, '%.2f' % float(height), ha='center', va='bottom')


# Add title and axis names
plt.title('Comparison between Accuracy of Algorithms')
plt.ylabel('Accuracy')
plt.figure(1)
plt.show()