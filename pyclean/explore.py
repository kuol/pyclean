# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 15:54:46 2016

@author: kliu
"""
import numpy as np
import pandas as pd

def main():
    path = "C:/Users/kliu/Documents/Work/201611_Predictive_Adviser/Data/"
    all_2007_2011 = path + "LoanStats3a.csv"
    rejected_2007_2012 = path + "RejectStats.csv"
    df_all = pd.read_csv(all_2007_2011, low_memory=False)
    df = variability_analysis(df_all)


def get_num_str_cols(df):
    num_cols = [x for x in df.columns if df[x].dtype == np.float64 or df[x].dtype == np.int64]
    cat_cols = [x for x in df if x not in num_cols]
    return num_cols, cat_cols

def without_many_nans(df, threshold = 0.1):
    """ Return column names where the percent of not_nans is greater than threshold
    """
    result = df.count() / float(df.shape[0]) > threshold
    return result.index[result]

def without_many_zeros(df, threshold = 0.1):
    """ Return column names where the percent of none_zeros is greater than threshold
    """ 
    result = (df.astype(bool).sum() / float(df.shape[0])) > threshold
    return result.index[result]  

def variability_analysis(df, nan_thres = 0.1, zero_thres = 0.1):
    cols = without_many_nans(df, nan_thres)
    print len(cols)
    df = df[cols]
    cols = without_many_zeros(df, zero_thres)
    print len(cols)
    return df[cols]
    
def check_levels(df, cat_cols):
    levels = df[cat_cols].apply(lambda x: len(set(x))) 
    return levels
 
def possible_nums(df, cat_cols, threshold = 5):
    """ Return column names if column item are in the following format:
        *) ' xx.x%   '
        *) '  $xxx '
    """
    cols = []
    for x in cat_cols:
        n_percent_sign = sum(df[x].apply(lambda x: (str(x).strip()[-1] == '%')))
        n_dollar_sign = sum(df[x].apply(lambda x: (str(x).strip()[0] == '$')))
        if n_percent_sign > threshold or n_dollar_sign > threshold:
            cols.append(x)
    return cols

   
if __name__ == '__main__':
    main()