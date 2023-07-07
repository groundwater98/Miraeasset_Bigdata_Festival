from fetchers.data_fetcher import DataFetcher
from db.database_manager import DatabaseManager
from managers.stock_manager import StockManager

db_manager = DatabaseManager('sqlite:///stocks.db')
stock_manager = StockManager(DataFetcher(), db_manager)

stock_manager.update_stock_data('AAPL')