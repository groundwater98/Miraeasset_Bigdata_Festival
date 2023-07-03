import yfinance as yf

class DataFetcher:
    def get_historical_data(self, ticker, period='max', interval='1d', start=None, end=None):
        """Get historical data from Yahoo Finance

        Args:
            ticker (_type_): 주식 종목 코드
            period (str, optional): 추출 기간. Defaults to 'max'.
            interval (str, optional): 추출 간격. Defaults to '1d'.
            start (_type_, optional): 시작 날짜. Defaults to None.
            end (_type_, optional): 종료 날짜. Defaults to None.

        Returns:
            hist: (pandas.DataFrame): historical data
        """
        stock  = yf.Ticker(ticker)
        if start and end:
            hist = stock.history(period=period, interval=interval, start=start, end=end, auto_adjust=False)
        hist = stock.history(period=period, interval=interval, auto_adjust=False)
        print(stock.history_metadata)
        return hist
    
    def get_actions(self, ticker):
        """ Get stock actions (dividends, splits, etc.) from Yahoo Finance

        Args:
            ticker (_type_): 주식 종목 코드

        Returns:
            actions: (pandas.DataFrame): stock actions 
        """
        stock  = yf.Ticker(ticker)
        actions = stock.actions
        print(actions)
        return actions
    
    def get_dividends(self, ticker):
        """Get stock dividends from Yahoo Finance

        Args:
            ticker (_type_): 주식 종목 코드

        Returns:
            actions: (pandas.DataFrame): stock actions 
        """
        stock  = yf.Ticker(ticker)
        dividends = stock.dividends
        print(dividends)
        return dividends
    
    # 
    def get_splits(self, ticker):
        """Get stock splits from Yahoo Finance

        Args:
            ticker (_type_): 주식 종목 코드

        Returns:
            actions: (pandas.DataFrame): stock actions 
        """
        stock  = yf.Ticker(ticker)
        splits = stock.splits
        print(splits)
        return splits

    def get_capital_gains(self, ticker):
        """Get stock capital gains from Yahoo Finance (only for mutual funds & etfs)

        Args:
            ticker (_type_): 주식 종목 코드

        Returns:
            actions: (pandas.DataFrame): stock actions 
        """
        stock  = yf.Ticker(ticker)
        capital_gains = stock.capital_gains
        print(capital_gains)
        return capital_gains
    
    def get_shares_full(self, ticker):
        """Get stock shares from Yahoo Finance

        Args:
            ticker (_type_): 주식 종목 코드

        Returns:
            _type_: _description_
        """
        stock = yf.Ticker(ticker)
        shares = stock.get_shares_full()
        print(shares)
        return shares
    
    
    
        

    