from sqlalchemy import create_engine, Table, MetaData
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import IntegrityError, OperationalError
from helpers import Logger

logger = Logger(__name__).get_logger()

class DatabaseManager:
    def __init__(self, db_url):
        self.engine = create_engine(db_url)
        self.metadata = MetaData()
        self.Session = sessionmaker(bind=self.engine)
            
    def save_data(self, table_name, data):
        session = self.Session()
        try:
            table = Table(table_name, self.metadata, autoload=True, autoload_with=self.engine)
            session.execute(table.insert(), data)
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
        