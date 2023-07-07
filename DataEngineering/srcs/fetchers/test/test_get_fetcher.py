import unittest
from fetchers.data_fetcher import DataFetcher

class TestDataFetcher(unittest.TestCase):
    def setUp(self):
        self.fetcher = DataFetcher()
        
    def test_get_historical_data(self):
        print("---- Historical Data ----")
        data = self.fetcher.get_historical_data("AAPL")
        print(data)
        self.assertIsNotNone(data)
        
    def test_get_actions(self):
        print("---- Actions ----")
        data = self.fetcher.get_actions("AAPL")
        print(data)
        self.assertIsNotNone(data)
    
    def test_get_dividends(self):
        data = self.fetcher.get_dividends("AAPL")
        print("---- Dividends ----")
        print(data)
        self.assertIsNotNone(data)
    
    def test_get_splits(self):
        data = self.fetcher.get_splits("AAPL")
        print("---- Splits ----")
        print(data)
        self.assertIsNotNone(data)
        
    def test_get_capital_gains(self):
        data = self.fetcher.get_capital_gains("AAPL")
        print("---- Capital Gain ----")
        print(data)
        self.assertIsNotNone(data)
        
    def test_get_shares_full(self):
        data = self.fetcher.get_shares_full("AAPL")
        print("---- Shares Full ----")
        print(data)
        self.assertIsNotNone(data)
    
if __name__ == '__main__':
    unittest.main()