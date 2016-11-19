import unittest
import numpy as np
import pandas as pd
from pyclean import diagnose_suggest
import pyclean.diagnose_suggest as ds

class TestColumnTypeAnalysis(unittest.TestCase):
    def setUp(self):
        d = {'a': ['we2', '%3a8', '23']*10,
             'b': range(30),
             'c': [1.2, 12, 34.0]*10,
             'd': [1.0, 2.0]*15}
        self.df = pd.DataFrame.from_dict(d)

        dd = {'a': [' $123.45', '$ 234', '$12,343'],
              'b': [' 234%', '12.34%', '12%'],
              'c': ['1,234','12,342', '123,342'],
              'd': ['1,2312','1,1231', '1234' ]}
        self.ddf = pd.DataFrame.from_dict(dd)

    def test_get_num_str_cols(self):
        float_cols, int_cols, cat_cols = ds.get_num_str_cols(self.df)
        self.assertEqual(float_cols, ['c', 'd'])
        self.assertEqual(int_cols, ['b'])
        self.assertEqual(cat_cols, ['a'])
    
    def test_change_float_is_int(self):
        ds.change_float_is_int(self.df, ['c', 'd'])
        self.assertEqual(self.df['d'].dtypes, 'int64')

    def test_cols_int_is_categorical(self):
        temp = ds.cols_int_is_categorical(self.df, ['b', 'd'])
        self.assertEqual(temp, ['d'])

    def test_cols_possible_nums(self):
        temp = ds.cols_possible_nums(self.ddf,list(self.ddf.columns))
        self.assertEqual(temp, ['a','b','c'])
    
    def test_fix_string_to_number(self):
        ds.fix_string_to_number(self.ddf, ['a', 'b', 'c'])
        self.assertItemsEqual(self.ddf['a'], [123.45, 234, 12343])
        self.assertItemsEqual(self.ddf['b'], [2.34, 0.1234, 0.12])
        self.assertItemsEqual(self.ddf['c'], [1234, 12342, 123342])

class TestMissingValueAnalysis(unittest.TestCase):
    def setUp(self):
        d = {'a': ['', '   \n', 'ha']*10,
             'b': [' ', 'John', 'Jane']*10,
             'c': [1, 5, np.nan]*10}
        self.df = pd.DataFrame.from_dict(d)

    def test_cols_with_nulls(self):
        temp = ds.cols_with_nulls(self.df)
        self.assertEqual(temp, ['a','b','c'])

    def test_cols_with_many_nulls(self):
        ds.cols_with_nulls(self.df)
        temp = ds.cols_with_many_nulls(self.df)
        self.assertEqual(temp, ['a'])

    def test_drop_cols(self):
        ds.drop_cols(self.df, ['a'])
        self.assertEqual(list(self.df.columns), ['b','c'])

    def test_impute_value(self):
        ds.cols_with_nulls(self.df)
        ds.impute(self.df, 'a', strategy = 'value', val = 'ha')
        self.assertItemsEqual(self.df['a'], ['ha']*30)

    def test_impute_mean(self):
        ds.cols_with_nulls(self.df)
        ds.impute(self.df, 'c', strategy = 'mean')
        self.assertItemsEqual(self.df['c'], [1, 5, 3]*10)

    def test_impute_median(self):
        ds.cols_with_nulls(self.df)
        ds.impute(self.df, 'c', strategy = 'median')
        self.assertItemsEqual(self.df['c'], [1, 5, 3]*10)
        

    def test_impute_value(self):
        ds.cols_with_nulls(self.df)
        ds.impute(self.df, 'a', strategy = 'most_frequent')
        self.assertItemsEqual(self.df['a'], ['ha']*30)

if __name__ == '__main__':
    unittest.main()
