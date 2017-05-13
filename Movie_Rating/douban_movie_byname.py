# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 15:54:17 2017

@author: zhouyu
"""
import requests
import re
from bs4 import BeautifulSoup
import os
os.chdir("/Users/zhouyu/Documents/Zhou_Yu/DS/Data_Incubator/Incubator_2017/Project")
import time
import pandas as pd
import string
#%% 1. Scraping data from https://movie.douban.com
# return the douban movie rating that matches the movie name and year
# read in the movie name
def doubanRating(name):
    movie_name = name.decode('gbk').encode('utf-8')
    url_head = 'http://movie.douban.com/subject_search'
    pageload = {'search_text': movie_name}
    r = requests.get(url_head,params = pageload)
    soup = BeautifulSoup(r.text,'html.parser')
    first_hit = soup.find_all(class_= 'nbg')
    try:
        r2_link = first_hit[0].get('href')
        # sometime douban returns items like celebrity instead of movies    
        if 'subject' not in r2_link:
            r2_link = first_hit[1].get('href')
        r2 = requests.get(r2_link)
        soup2 = BeautifulSoup(r2.text,'html.parser')
        title = soup2.find(property = "v:itemreviewed")
        title = title.get_text() # in unicode
        # remove Chinese characters
        title = ' '.join((title.split(' '))[1:])
        title = filter(lambda x:x in set(string.printable),title)
        flag = True
        if title != name:
            print "Warning: name may not match"
            flag = False
        year = (soup2.find(class_='year')).get_text()# in unicode
        rating = (soup2.find(class_="ll rating_num")).get_text() # in unicode
        num_review = (soup2.find(property="v:votes")).get_text()
        return [title, year, rating,num_review,flag]
    except:
        print "Record not found for: "+name
        return [name, None, None, None, None]
     
#%%2. Store scrapped data    
dataset = pd.read_csv("movie_metadata.csv")
total_length = 5043
#first_query = 2500
res = pd.DataFrame(columns = ('movie_title','year','rating','num_review','flag'))
for i in xrange(1,total_length):
    name = dataset['movie_title'][i].strip().strip('\xc2\xa0')
    res.loc[i] = doubanRating(name)
    print "slowly and finally done %d query"%i
    time.sleep(10)
    if (i%50==0):
        res.to_csv("douban_movie_review.csv")
        print "saved until record: %d"%i
res.to_csv("douban_movie_review.csv")
#%% 2. start preliminary data analysis
imdb_dat = pd.read_csv("movie_metadata.csv")
douban_dat = pd.read_csv("douban_movie_review.csv")
douban_dat.info()
douban_dat.rename(columns = {'movie_title':'d_movie_title','year':'d_year','rating':'douban_score','num_review':'dnum_review','flag':'dflag'},inplace = True)
res_dat = pd.concat([imdb_dat,douban_dat],axis = 1)
res_dat.info()
# 2.1. visulize the gross distribution of ratings from imdb(x-axis) and douban(y-axis)
import seaborn as sns
g = sns.jointplot(x = 'imdb_score',y = 'douban_score',data = res_dat)
g.ax_joint.set(xlim=(1, 10), ylim=(1, 10))
#%%
# 2.2. Predict differences in ratings
res_dat['diff_rating'] = res_dat['douban_score']-res_dat['imdb_score'] 
# 2.2.1. covert categorical variable Genre to Dummy variables
# only extract the first genre out of the list to simplify the problem
res_dat['genre1'] = res_dat.apply(lambda row:(row['genres'].split('|'))[0],axis = 1)
#res_dat['genre1'].value_counts()
# Because there are 21 genres, here we only choose the top 7 to convert to index
top_genre = ['Comedy','Action','Drama','Adventure','Crime','Biography','Horror']
# The rest of genre types we just consider them as others
res_dat['top_genre'] = res_dat.apply(lambda row:row['genre1'] if row['genre1'] in top_genre else 'Other',axis =1)
#select num_user_for_reviews ,director_facebook_likes ,actor_1_facebook_likes  ,gross , genres,
#budget,# dnum_review # for EDA
res_subdat = res_dat[['top_genre','num_user_for_reviews','director_facebook_likes','actor_1_facebook_likes','gross','budget','dnum_review','diff_rating']]
res_subdat = pd.get_dummies(res_subdat,prefix =['top_genre'])
# create a subset for visualization and preliminary analysis
col2 = [u'num_user_for_reviews', u'director_facebook_likes',
       u'actor_1_facebook_likes', u'gross', u'budget', u'dnum_review', u'top_genre_Action', u'top_genre_Adventure',
       u'top_genre_Biography', u'top_genre_Comedy', u'top_genre_Crime',
       u'top_genre_Drama', u'top_genre_Horror', u'top_genre_Other',u'diff_rating']
res_subdat = res_subdat[col2]
# a subset for plotting correlation
col_cat = [u'gross', u'budget', u'dnum_review',u'num_user_for_reviews',u'top_genre_Action', u'top_genre_Adventure',
       u'top_genre_Biography', u'top_genre_Comedy', u'top_genre_Crime',
       u'top_genre_Drama', u'top_genre_Horror', u'diff_rating']
res_subdat_genre = res_subdat[col_cat]
# show pair-wise correlation between differences in ratings and estimators
import matplotlib.pylab as plt
import numpy as np
corr = res_subdat_genre.corr()
sns.set(style = "white")
f,ax = plt.subplots(figsize=(11,9))
cmap = sns.diverging_palette(220,10,as_cmap=True)
mask = np.zeros_like(corr,dtype = np.bool)
sns.heatmap(corr,mask = mask,cmap = cmap, vmax=.3,square = True, linewidths = .5,
            cbar_kws = {"shrink": .5},ax = ax)
# prepare trainning set and target set
col_train = col2[:len(col2)-1]
col_target = col2[len(col2)-1]
cl_res_subdat = res_subdat.dropna(axis =0)
# 2.2.2 Use Random Forest Regressor for prediction
train_set = cl_res_subdat[col_train]
target_set = cl_res_subdat[col_target]
# METHOD 1: BUILD randomforestregressor
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 100)
forest = rf.fit(train_set, target_set)
pre_res = rf.score(train_set,target_set)
# print: R-sqr
print pre_res
