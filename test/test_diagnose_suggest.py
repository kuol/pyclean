import unittest
import pandas as pd
from pyclean import diagnose_suggest
import pyclean.diagnose_suggest as ds

class TestDiagnose(unittest.TestCase):
    def setUp(self):
        d = {'a': ['we2', '%3a8', '23']*10,
             'b': range(30),
             'c': [1.2, 12, 34.0]*10,
             'd': [1.0, 2.0]*15,}
        self.df = pd.DataFrame.from_dict(d)

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




if __name__ == '__main__':
    unittest.main()
