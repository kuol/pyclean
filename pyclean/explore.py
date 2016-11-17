# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 15:54:46 2016

@author: kliu
"""
import numpy as np
import pandas as pd
import re

def main():
    path = "C:/Users/kliu/Documents/Work/201611_Predictive_Adviser/Data/"
    all_2007_2011 = path + "LoanStats3a.csv"
    rejected_2007_2012 = path + "RejectStats.csv"
    df_all = pd.read_csv(all_2007_2011, low_memory=False)
    df = variability_analysis(df_all)

#==================================================
# Which columns are: 
#       - Numeric
#       - String
#==================================================
def get_num_str_cols(df):
    float_cols = [x for x in df.columns if df[x].dtype == np.float64]
    int_cols = [x for x in df.columns if df[x].dtype == np.int64]
    num_cols = float_cols + int_cols
    cat_cols = [x for x in df if x not in num_cols]
    return float_cols, int_cols, cat_cols

#============================================================
# Column type analysis: 
#       - Is numeric really numeric?
#             1) Float may be integer : 1.0, 2.0, 3.0 ...
#             2) Integer may be categorical: 1,1,1,0,0,0,1
#       - Is string really string?
#             1) Percentage: 23%
#             2) Currency:  $2323.23
#             3) Comma seperated number: 1,234
#             4) A combination of 2) and 3): $1,235.29
#=============================================================
def float_is_int(x):
    if np.isnan(x):
        return True
    else:
        return int(x) - x == 0

def change_float_is_int(df, float_cols):
    """ Change float column to int, if it essentially is integer type. """
    for x in float_cols:
        if all(df[x].apply(float_is_int)):
            df[x] = df[x].apply(lambda x: int(x))


def cols_int_is_categorical(df, int_cols, threshold = 5):
    """ Return columns in integer type, but are possibilty categorical"""
    threshold = min(threshold, df.shape[0]/10)
    df_count = df[int_cols].apply(lambda x: len(set(x)))
    return list(df_count[df_count < threshold].index)
    

def cols_possible_nums(df, cat_cols, threshold = 5):
    """ Return column names if column item are in the following format:
        *) ' xx.x%   '
        *) '  $xxx '
        *) ' 12,347' -- TODO
    """
    threshold = min(threshold, df.shape[0]/10)
    cols = []
    for x in cat_cols:
        n_percent_sign = sum(df[x].apply(lambda x: (str(x).strip()[-1] == '%')))
        n_dollar_sign = sum(df[x].apply(lambda x: (str(x).strip()[0] == '$')))
        n_comma_number = sum(df[x].apply(lambda x: bool(re.match("^[\d\,]*$", x))))
        if n_percent_sign > threshold or n_dollar_sign > threshold or n_comma_number > threshold:
            cols.append(x)
    return cols 

#==============================================================================
# Column type fix: 
#       - Is numeric really numeric?
#             1) Float may be integer :  change float to integer (detect & fix)
#             2) Integer may be categorical: change integer to string
#       - Is string really string?
#             1) Percentage: 23%
#             2) Currency:  $2323.23
#             3) Comma seperated number: 1,234
#             4) A combination of 2) and 3): $1,235.29
#==============================================================================
def change_int_to_string(df, cols):
    df[cols] = df[cols].astype(str)

def strip_helper(x):
    xx = x.strip()
    if not len(xx):
        return np.nan 

    xx = re.sub('[\,\$]', '', xx)
    if xx[-1] == '%':
        xx = float(xx[:-1])/100
    elif re.search('\.', xx):
        xx = float(xx)
    else:
        xx = int(xx)        
               
def fix_string_to_number(df, cols):
    df[cols] = df[cols].applymap(strip_helper)


#======================================================================
# Missing value detection:
#       - Return columns with missing values, the program will convert
#         white space in string type column to numpy.nan
#=======================================================================
def cols_with_nulls(df):
    """ Convert whitespace entries to NaN, Return columns with NaN
    """
    # Note: Empty string will be converted to NaN automatically,
    df = df.replace(r'\s+', np.nan, regex=True)
    return list(df.isnull().any().index)
    
def cols_with_many_nulls(df, threshold = 0.5):
    """ Return column names, where null values are over percentage of 
    threshold (50% by default)"""
    null_stats = dict(df.astype(bool).sum())
    cols = [k for k, v in null_stats.items() if v > threshold * df.shape[0]]
    return cols   

#======================================================================
# Missing value handling:
#       1. If the percentage of missing value is too high, drop the column
#       2. Impute missing values with 4 strategies.
#=======================================================================
der drop_cols(df, cols):
    """Delete column cols from df in place"""
    df.drop(cols, inplace=True, axis=1)


def impute(df, col, strategy = 'value', val = 0):
    """ Impute missing values (NaN) based on the following strategy:
            - Fill all missing value with a designate value: val
            - Fill all missing value with "mean"
            - Fill all missing value with "median"
            - Fill all missing value with "most_frequent"
    """
    if strategy == 'mean':
        val = df[col].mean()
    elif strategy == 'median':
        val = df[col].median()
    elif strategy == 'most_frequent':
        val = df[col].mode()[0]
    df[col].fillna(val, inplace = True)    
    


#======================================================================
# Extreme value / Outlier analysis:
#   - Numeric columns:
#       1. Single column: Quantile
#       2. Single column: 3-sigma
#       3. Multiple column: Mahalanobis distance TODO
#       4. Multiple column: K-nearest neighbor TODO
#   - String columns:
#       1. Outlier -- umbalanced TODO
#=======================================================================
def detect_outliers_single(df, col, strategy = 'quantile'):
    if df[col].dtype.name == 'object':
        print "[Error]: Outlier detection is only for numeric columns at the moment"
        return
    if s.isnull().any():
        print "[Error]: There are missing values in the column. ", \
            "Please drop the column or impute the missing values first."

    s = df[col]
    if strategy == '3sigma': 
        outliers = s[(s-s.mean()).abs() > 3 * s.std()]
    else:
        outliers = s[(s > s.quantile(.25)) & (s < s.quantile(.75))]
    return outliers
    
    

#============================================================================
# Variability analysis:
#       - Numeric columns: check if standard deviation < threshold
#       - Categorical columns: check if minority class percentage < threshold
#============================================================================
def check_std(df, num_cols, threshold = 0.01):
    """ Return column names, whose standard deviation is less than the threshold"""
    stds = dict(df.std())
    cols = [k for k, v in stds.items() if v < threshold]
    return cols

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

#======================================================================
# Target analysis:
#       1. Missing values 
#           - Records with target missing are used for testing set.
#           - Otherwise, discard these records.
#       2. Integer, but actually is categorical
#       3. Categorical target: very imbalanced
#======================================================================= 
def check_levels(df, cat_cols):
    levels = df[cat_cols].apply(lambda x: len(set(x))) 
    return levels

def level_distribution(df, cat_cols, threshold = 0.1):
    cols = []
    d = {}
    for x in cat_cols:
        level_counts = dict(df[x].value_counts())
        vals = level_counts.values()
        lowest = min([float(x)/sum(vals) for x in vals])
        d[x] = level_counts 
        if lowest < threshold:
            cols.append(x)
    return cols, d
       
       
def analyze_target(df, target, thres_level = 5):
    num_missing = df[target].isnull().sum()
    if num_missing > 0:
        print "[Warning:] Target column has missing values!!"
    if df[target].dtype == np.float64:
        if all(df[target].apply(float_is_int)):
            print "[Message:] change the target column to integer type."
            df[target] = df[target].apply(lambda x: int(x))
    if len(set(df[target])) <= thres_level:
        print "[Warning:] Target column may be categorical!!"
            
#======================================================================
# Column analysis:
#       1. Are there too many columns?
#       2. Do some of these columns linearly dependent?
#=======================================================================       
        
        
if __name__ == '__main__':
    main()