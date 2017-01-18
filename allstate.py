# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 22:43:13 2016

@author: zhouyu
for kaggle challenge  - allstate
"""
import pandas as pd
import numpy as np
import seaborn as sns
dataset = pd.read_csv('/Users/zhouyu/Documents/Zhou_Yu/DS/kaggle_challenge/train.csv')
testset = pd.read_csv('/Users/zhouyu/Documents/Zhou_Yu/DS/kaggle_challenge/test.csv')
dataset.info();
dataset.columns[dataset.isnull().sum()>0];
dataset.describe();
dataset.describe(include = ['object'])
cont_columns = []
cat_columns = []
for i in dataset.iloc[:,1:].columns:
    if dataset[i].dtype =='float' and i!='loss':
        cont_columns.append(i)
    elif dataset[i].dtype =='object':
        cat_columns.append(i)
#%%    plot-pairwise 
#sel_col = cont_columns[0:3]
#sel_col.append('loss')       
#g = sns.pairplot(dataset[cont_columns], vars = sel_col,kind = 'scatter',diag_kind = 'kde')

#%% Analyize correlation among continuous variables
import matplotlib.pylab as plt
corr = dataset[cont_columns].corr()
sns.set(style = "white")
f,ax = plt.subplots(figsize=(11,9))
cmap = sns.diverging_palette(220,10,as_cmap=True)
mask = np.zeros_like(corr,dtype = np.bool)
sns.heatmap(corr,mask = mask,cmap = cmap, vmax=.3,square = True, linewidths = .5,
            cbar_kws = {"shrink": .5},ax = ax)
#%% a simple XGregression model
import matplotlib.mlab as mlab
from scipy.stats import norm, lognorm
dataset['log_loss'] = np.log(dataset['loss'])
# fit normal distribution on ln(loss)
(mu,sigma) = norm.fit(dataset['log_loss'])
n,bins,pathes = plt.hist(dataset['log_loss'],60,normed=1,facecolor = 'green',alpha =0.75)
#add the fitted line
y = mlab.normpdf(bins,mu,sigma)
l = plt.plot(bins,y,'r--',linewidth =2)
plt.xlabel('Ln(loss)')
plt.ylabel('Probability')
plt.grid(True)
plt.show()
#%% XGboost algorithm
import xgboost as xgb
ntrain  = dataset.shape[0];
ntest =  testset.shape[0];
train_test = pd.concat((dataset[cont_columns+cat_columns],testset[cont_columns+cat_columns])).reset_index(drop = True)  
for c in cat_columns:
    train_test[c] = train_test[c].astype('category').cat.codes
    
train_x = train_test.iloc[:ntrain,:]
test_x = train_test.iloc[ntrain:,:]
xgdmat = xgb.DMatrix(train_x, dataset['log_loss'])
#%%
params = {'eta':1, 'seed':0, 'subsample':0.5,'colsample_bytree': 0.5, 
             'objective': 'reg:linear', 'max_depth':6, 'min_child_weight':3}
num_rounds = 10
bst = xgb.train(params, xgdmat, num_boost_round = num_rounds)
#%% submission
test_xgb = xgb.DMatrix(test_x)
submission = pd.read_csv('/Users/zhouyu/Documents/Zhou_Yu/DS/kaggle_challenge/sample_submission.csv')
submission.iloc[:,1] = np.exp(bst.predict(test_xgb)) 
submission.to_csv('/Users/zhouyu/Documents/Zhou_Yu/DS/kaggle_challenge/submission.csv',index = None)       
#%% compare predicted 
train_xgb = xgb.DMatrix(train_x)
pred = np.exp(bst.predict(train_xgb))
plt.scatter(dataset['loss'],pred)

axes = plt.gca()
axes.set_xlim([0, 80000])
axes.set_ylim([0, 80000])

plt.show()