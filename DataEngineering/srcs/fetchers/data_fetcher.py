import yfinance as yf

class DataFetcher:
    def get_data(self, ticker):
        stock  = yf.Ticker(ticker)
        return stock.history()