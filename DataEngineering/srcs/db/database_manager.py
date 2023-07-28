from sqlalchemy import create_engine, Table, MetaData
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.ext.declarative import declarative_base
from schema import Base, StockData, StockMetadata
from functools import wraps
from tools.logger import Logger
from db.schema import StockData
from fetchers.data_fetcher import DataFetcher

logger = Logger(__name__).get_logger()

def handle_insertion(data_type):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                logger.info(f"{data_type} saved successfully")
                return f"{data_type} saved successfully"
            except Exception as e:
                logger.error(f"Failure in saving {data_type}: {str(e)}")
                return f"Failure in saving {data_type}"
        return wrapper
    return decorator
            
class Singleton(type):
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class DatabaseManager(metaclass=Singleton):
    def __init__(self, db_url):
        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)
        self.fetcher = DataFetcher()
        logger.info("Database manager initialized")

    def create_all_table(self):
        try:
            metadata = Base.metadata
            metadata.create_all(self.engine, checkfirst=True)
            logger.info("Table created successfully")
            logger.info("If table already exists, no effect.")
        except OperationalError as e:
            logger.Fatal("OperationalError occurred while creating tables: %s", str(e))
        except Exception as e:
            logger.Fatal("Error occurred while creating tables: %s", str(e))

    def save_data(self, data):
        session = self.Session()
        try:
            session.add_all(data)
            session.commit()
        except IntegrityError as e:
            session.rollback()
            logger.error("Integrity error occurred: %s", e)
            raise e
        except OperationalError as e:
            session.rollback()
            logger.error("Operational error occurred: %s", e)
            raise e
        except Exception as e:
            session.rollback()
            logger.error("Unknown error occurred: %s", e)
            raise e
        finally:
            session.close()
    
    @handle_insertion('historical data')
    def save_historical_data(self, ticker, period='max', interval='1d', start=None, end=None):
        data = self.fetcher.get_historical_data(ticker, period, interval, start, end)
        data = [
            StockData(
                date=str(row.Index),
                open=row.Open,
                high=row.High,
                low=row.Low,
                close=row.Close,
                volume=row.Volume,
                # dividends=row.Dividends,
                # stock_splits=row.Stock_Splits,
            )
            for row in data.itertuples()
        ]
        
        self.save_data(data)

        