import unittest
import pandas as pd
from pyclean import diagnose_suggest
import pyclean.diagnose_suggest as ds

class TestDiagnose(unittest.TestCase):
    def setUp(self):
        d = {'a': ['we2', '%3a8', '23'],
             'b': range(2,5),
             'c': [1.2, 12, 34.0]}
        self.df = pd.DataFrame.from_dict(d)
    def test_something(self):
        float_cols, int_cols, cat_cols = ds.get_num_str_cols(self.df)
        #print float_cols, int_cols, cat_cols
        self.assertEqual(float_cols, ['c'])
        self.assertEqual(int_cols, ['b'])
        self.assertEqual(cat_cols, ['a'])





if __name__ == '__main__':
    unittest.main()
