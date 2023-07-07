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
            shares: (pandas.DataFrame): stock shares
        """
        stock = yf.Ticker(ticker)
        shares = stock.get_shares_full()
        print(shares)
        return shares
    
    def get_income_statement(self, ticker):
        """Get stock income statement from Yahoo Finance

        Args:
            ticker (_type_): _description_
            
        Returns:
            income_stmt: (pandas.DataFrame): stock income statement
        """
        stock = yf.Ticker(ticker)
        income_stmt = stock.income_stmt
        print(income_stmt)
        return income_stmt
        
    
    def get_quarterly_income_statement(self, ticker):
        """Get stock quarterly income statement from Yahoo Finance

        Args:
            ticker (_type_): 주식 종목 코드
        
        Returns:
            income_stmt: (pandas.DataFrame): stock income statement
        """
        stock = yf.Ticker(ticker)
        income_stmt = stock.quarterly_income_stmt
        print(income_stmt)
        return income_stmt
        
    
    def get_balance_sheet(self, ticker):
        """Get stock balance sheet from Yahoo Finance

        Args:
            ticker (_type_): 주식 종목 코드

        Returns:
            balance_sheet: (pandas.DataFrame): stock balance sheet
        """
        stock = yf.Ticker(ticker)
        balance_sheet = stock.balance_sheet
        print(balance_sheet)
        return balance_sheet
    
    def get_quarterly_balance_sheet(self, ticker):
        """Get stock quarterly balance sheet from Yahoo Finance

        Args:
            ticker (_type_): 주식 종목 코드

        Returns:
            balance_sheet: (pandas.DataFrame): stock balance sheet
        """
        stock = yf.Ticker(ticker)
        balance_sheet = stock.quarterly_balance_sheet
        print(balance_sheet)
        return balance_sheet
    
    def get_cashflow(self, ticker):
        """Get stock cashflow from Yahoo Finance

        Args:
            ticker (_type_): 주식 종목 코드

        Returns:
            cashflow: (pandas.DataFrame): stock cashflow
        """
        stock = yf.Ticker(ticker)
        cashflow = stock.cashflow
        print(cashflow)
        return cashflow
    
    def get_quarterly_cashflow(self, ticker):
        """Get stock quarterly cashflow from Yahoo Finance

        Args:
            ticker (_type_): 주식 종목 코드

        Returns:
            cashflow: (pandas.DataFrame): stock cashflow
        """
        stock = yf.Ticker(ticker)
        cashflow = stock.quarterly_cashflow
        print(cashflow)
        return cashflow
    
    def get_major_holders(self, ticker):
        """Get stock major holders from Yahoo Finance

        Args:
            ticker (_type_): 주식 종목 코드

        Returns:
            major_holders: (pandas.DataFrame): stock major holders
        """
        stock = yf.Ticker(ticker)
        major_holders = stock.major_holders
        print(major_holders)
        return major_holders
    
    def get_institutional_holders(self, ticker):
        """Get stock institutional holders from Yahoo Finance

        Args:
            ticker (_type_): 주식 종목 코드

        Returns:
            institutional_holders: (pandas.DataFrame): stock institutional holders
        """
        stock = yf.Ticker(ticker)
        institutional_holders = stock.institutional_holders
        print(institutional_holders)
        return institutional_holders
    
    def get_mutualfund_holders(self, ticker):
        """Get stock mutualfund holders from Yahoo Finance

        Args:
            ticker (_type_): 주식 종목 코드

        Returns:
            mutualfund_holders: (pandas.DataFrame): stock mutualfund holders
        """
        stock = yf.Ticker(ticker)
        mutualfund_holders = stock.mutualfund_holders
        print(mutualfund_holders)
        return mutualfund_holders
    
    def get_earning_dates(self, ticker, limit):
        """Get stock earning dates from Yahoo Finance

        Args:
            ticker (string): 주식 종목 코드
            limit (int): 제한

        Returns:
            earning_dates: (pandas.DataFrame): stock earning dates
        """
        stock = yf.Ticker(ticker)
        earning_dates = stock.get_earnings_dates(limit=limit)
        print(earning_dates)
        return earning_dates
    
    def get_isin(self, ticker):
        """Get stock ISIN number from Yahoo Finance

        Args:
            ticker (_type_): 주식 종목 코드

        Returns:
            isin: (string): stock ISIN number
        """
        stock = yf.Ticker(ticker)
        isin = stock.isin
        print(isin)
        return isin
    
    def get_options(self, ticker):
        """Get stock options from Yahoo Finance

        Args:
            ticker (_type_): 주식 종목 코드

        Returns:
            options: (pandas.DataFrame): stock options
        """
        stock = yf.Ticker(ticker)
        options = stock.options
        print(options)
        return options
    
    def get_news(self, ticker):
        """Get stock news from Yahoo Finance

        Args:
            ticker (_type_): 주식 종목 코드

        Returns:
            news: (pandas.DataFrame): stock news
        """
        stock = yf.Ticker(ticker)
        news = stock.news
        print(news)
        return news

    def get_option_chain(self, ticker, date=None):
        """Get stock option chain from Yahoo Finance

        Args:
            ticker (_type_): 주식 종목 코드
            date (string, 'YYYY-MM-DD'): 날짜

        Returns:
            option_chain: (pandas.DataFrame): stock option chain
        """
        if date is None:
            raise ValueError("date is required")
        stock = yf.Ticker(ticker)
        option_chain = stock.option_chain(date)
        print(option_chain)
        return option_chain
