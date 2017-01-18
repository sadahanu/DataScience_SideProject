# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 23:10:40 2016

@author: zhouyu
"""
#%%
import pandas as pd
import numpy as np
import os
import re
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
os.chdir('/Users/zhouyu/Documents/Zhou_Yu/DS/kaggle_challenge/text processing')
#%% step1: import data 
import glob
alltrainfiles = glob.glob("*.csv")
raw_text =pd.concat((pd.read_csv(f,index_col = None, header =0) for f in alltrainfiles),ignore_index = True)
#raw_text = pd.read_csv("crypto.csv",index_col = None)
#%% step2: clean data, remove HTML, symbols and stopwords
def text_to_words(rawtext):
    #split into individual words, remove HTML, only keep letters and number
    # convert letters to lower case
    reg_c = re.compile('[^a-zA-Z0-9_\\+\\-]')
    words = [word for word in reg_c.split(rawtext.lower()) if word!='']
    stops = set(stopwords.words("english"))
    #take out stop words
    meaningful_words = [w for w in words if not w in stops]
    return(" ".join(meaningful_words))
def target_to_words(rawtext):
    #only return the first target word
    reg_c = re.compile('[^a-zA-Z0-9_\\+\\-]')
    words = [word for word in reg_c.split(rawtext.lower()) if word!='']
    stops = set(stopwords.words("english"))
    #take out stop words
    meaningful_words = [w for w in words if not w in stops]
    return(meaningful_words[0])
#%%    
cleaned_post = []
cleaned_target = []
sz = raw_text.shape[0]
for i in range(0,sz):
    raw_post = raw_text['title'][i]+' '+raw_text['content'][i]
    raw_post = BeautifulSoup(raw_post).get_text() 
    cleaned_post.append(text_to_words(raw_post))
    cleaned_target.append(target_to_words(raw_text['tags'][i]))
    if((i+1)%1000==0):
        print "Cleanning %d of %d\n" % (i+1,sz)
#print cleaned_post[1]
#%% step3: creating features from a bag of words
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000) 

X_train_counts = count_vect.fit_transform(cleaned_post) 
#X_target_counts = count_vect.fit_transform(cleaned_target)

from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf = False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
#%% training a linear model
# METHOD 1: BUILD randomforestclassifier...
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 10)
forest = rf.fit(X_train_tf, cleaned_target)
#%% examine the result produced by METHOD 1: 
pred = rf.predict(X_train_tf)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from collections import OrderedDict
import matplotlib.pyplot as plt
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
cnf_matrix = confusion_matrix(cleaned_target,pred)
#target_names = set(cleaned_target)
#np.set_printoptions(precision = 2)
#plt.figure()
#plot_confusion_matrix(cnf_matrix,classes = target_names,normalize = True,title='Normalized confusion matrix')
#plt.show()
target_names = list(OrderedDict.fromkeys(cleaned_target))
print(classification_report(cleaned_target,pred,target_names = target_names))
#######
#%% Method 2: directly predicted as the highest frequency element
# find the highest tf-idf
#step1: select a random sample 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from collections import OrderedDict
sample = np.random.choice(87000,1000,replace = False)
tf_pred = []
tf_target = []
for i in range(0,1000):
    r = sample[i];
    tf_target.append(cleaned_target[r])
    tf_post = X_train_tf.getrow(r).toarray()
    tf_post_max = tf_post.argmax()
    tf_pred.append(count_vect.get_feature_names()[tf_post_max])
tf_cnf_matrix = confusion_matrix(tf_target,tf_pred)
target_names = list(OrderedDict.fromkeys(tf_pred+tf_target))
print(classification_report(tf_target, tf_pred,target_names =target_names))
#%% evaluate test set
test = pd.read_csv('test/test.csv')
cleaned_test = []
test_sz = test.shape[0]
for i in range(0,test_sz):
    test_post = test['title'][i]+' '+test['content'][i]
    test_post = BeautifulSoup(test_post).get_text() 
    cleaned_test.append(text_to_words(test_post))
    if((i+1)%1000==0):
        print "Cleanning %d of %d\n" % (i+1,test_sz)
#%% use random forest        
X_test_counts = count_vect.fit_transform(cleaned_test) 
X_test_tf = tf_transformer.transform(X_test_counts)
result = forest.predict(X_test_counts)
# use max tf-idf
#%%
test_pred = []
for i in range(0,test_sz):
    tf_test = X_test_tf.getrow(i).toarray()
    # just return one tag
    #tf_test_max = tf_test.argmax()
    #test_pred.append(count_vect.get_feature_names()[tf_test_max])
    ind = np.argpartition(tf_test,-4)[:,-4:]
    pred_tags = [count_vect.get_feature_names()[j] for j in ind[0,:].tolist()]
    test_pred.append( " ".join(pred_tags))
    if((i+1)%1000==0):
        print "Predicting %d of %d\n" % (i+1,test_sz)
result = test_pred
#%% prepare submission
submission = pd.read_csv('test/sample_submission.csv')
submission.iloc[:,1] = result
submission.to_csv('test/submission.csv',index = None) 
#%% try to use NMF model can not be mapped to specific question...
n_features = 5000
n_topics = 10
n_samples = test_sz
n_top_words = 4
def get_top_words(model, feature_names, n_top_words):
    res = []
    for topic_idx, topic in enumerate(model.components_):
        tags = " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        res.append(tags)
    return res

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF
from time import time
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                   max_features=n_features,
                                   stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(cleaned_test)
# Fit the NMF model
print("Fitting the NMF model (Frobenius norm) with tf-idf features, "
      "n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
t0 = time()
nmf = NMF(n_components=n_topics, random_state=1,
          alpha=.1, l1_ratio=.5).fit(tfidf)
print("done in %0.3fs." % (time() - t0))

print("\nTopics in NMF model (Frobenius norm):")
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
#print_top_words(nmf, tfidf_feature_names, n_top_words)
result = get_top_words(nmf,tfidf_feature_names,n_top_words)