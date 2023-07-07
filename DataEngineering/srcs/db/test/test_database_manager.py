import unittest
from unittest.mock import patch, Mock
from db.database_manager import DatabaseManager
from pandas import DataFrame

class TestDatabaseManager(unittest.TestCase):
    @patch('fetchers.data_fetcher.DataFetcher.get_historical_data')
    @patch('db.database_manager.DatabaseManager.save_data')
    def test_save_historical_data(self, mock_save_data, mock_get_historical_data):
        # Define the return value for get_historical_data
        mock_get_historical_data.return_value = DataFrame({
            'Open': [1.1, 1.2],
            'High': [1.3, 1.4],
            'Low': [1.0, 0.9],
            'Close': [1.2, 1.1],
            'Adj Close': [1.2, 1.1],
            'Volume': [1000, 2000],
            'Dividends': [0.0, 0.0],
            'Stock Splits': [0.0, 0.0],
            'Index': ['2022-01-01', '2022-01-02'],
        })
        db_password = '1234'  # Replace this with your actual password
        db_manager = DatabaseManager(f'postgresql://postgres:{db_password}@localhost:5432/postgres')

        # Call the method you want to test
        result = db_manager.save_historical_data('AAPL', period='1y', interval='1d')

        # Check that the method behaves as expected
        mock_get_historical_data.assert_called_once_with('AAPL', '1y', '1d', None, None)
        mock_save_data.assert_called_once_with('some_data')
        self.assertEqual(result, 'historical data saved successfully')

if __name__ == '__main__':
    unittest.main()