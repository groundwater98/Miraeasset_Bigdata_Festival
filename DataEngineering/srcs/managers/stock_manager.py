class StockManager:
    def __init__(self, fetcher, db_manager):
        self.fetcher = fetcher
        self.db_manager = db_manager

    def update_stock_data(self, ticker):
        data = self.fetcher.get_data(ticker)
        # process the data if necessary
        self.db_manager.save_data(ticker, data)