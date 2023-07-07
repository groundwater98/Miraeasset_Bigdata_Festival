from sqlalchemy import create_engine, Column, Integer, Float, String, BigInteger
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class StockData(Base):
    __tablename__ = 'stock_data'
    
    id = Column(Integer, primary_key=True)
    date = Column(String)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    adj_close = Column(Float)
    volume = Column(BigInteger)
    dividends = Column(Float)
    stock_splits = Column(Float)

class StockMetadata(Base):
    __tablename__ = 'stock_metadata'
    
    symbol = Column(String, primary_key=True)
    currency = Column(String)
    exchange_name = Column(String)
    instrument_type = Column(String)
    first_trade_date = Column(Integer)
    regular_market_time = Column(Integer)
    gmtoffset = Column(Integer)
    timezone = Column(String)
    exchange_timezone_name = Column(String)
    regular_market_price = Column(Float)
    chart_previous_close = Column(Float)
    price_hint = Column(Integer)
    data_granularity = Column(String)
    valid_ranges = Column(String)
    current_trading_period = Column(String)