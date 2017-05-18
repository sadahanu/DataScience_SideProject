# -*- coding: utf-8 -*-
"""
Created on Wed May 17 19:45:36 2017

@author: Zhou
"""
import pandas as pd
def getExperimentalReviews(inFile, id_pool):
    # inFile is the input json raw file
    # here is the 3+G Json
    # id_pool is the Business id list from the business info json
    count = 0
    record = []
    with open(inFile,'rb') as f:
        count +=1
        print "now searching record",count
        for line in f:
            if line.split('"')[11] in id_pool:
                record.append(line.rstrip())
            if count == 50:
                break
        data_str = "["+','.join(record)+"]" 
    data_df = pd.read_json(data_str) 
    return data_df    

def getExperimentalTips(inFile, id_pool):
    record = []
    with open(inFile,'rb') as f:
        for line in f:
            if line.split('"')[13] in id_pool:
                record.append(line.rstrip())
        data_str = "["+','.join(record)+"]"    
    data_df = pd.read_json(data_str)
    return data_df  


