import os
import toml
import pymysql
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, Table, Column, MetaData, String, inspect, text
import pandas as pd
from tools.logger import Logger

config_path = os.getenv('DB_CONFIG_PATH')
logger = Logger(__name__).get_logger()

def clear_table_contents(table_name):
    engine = create_engine_from_config()
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        delete_stmt = text(f"DELETE FROM {table_name}")
        session.execute(delete_stmt)
        session.commit()
        logger.info(f"Deleted records from {table_name}")
    except Exception as e:
        session.rollback()
        logger.error(f"Error: problem occurred while deleting records from {table_name}, {e}")
    finally:
        session.close()
    
# MySQL에 테이블 생성 함수
def create_table_from_schema(engine, table_name, table):
    logger.info(f"Creating table {table_name}...")
    inspector = inspect(engine)
    if table_name in inspector.get_table_names():
        logger.info(f"Table {table_name} already exists")
        return
    else:
        table.create(bind=engine)
        logger.info(f"Table {table_name} created")
        
# 스키마 추출 함수
def infer_schema_from_data(data):
    schema = {}
    for key, value in data.items():
        if isinstance(value, int):
            schema[key] = "INT"
        elif isinstance(value, float):
            schema[key] = "FLOAT"
        else:
            schema[key] = f"VARCHAR({max(255, len(str(value)))})"
    
    print(schema)
    return schema

def create_engine_from_config(config_path=None):
    host = "db-iflf8-kr.vpc-pub-cdb.ntruss.com"
    user = "youngmuk"
    password = "youngmuk2!"
    db = "inhive"

    logger.info("Connecting to MySQL...")
    # 데이터베이스 연결
    engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}/{db}")
    return engine

# 데이터 저장 함수
def save_to_mysql_auto(data, table_name):
    engine = create_engine_from_config(config_path)
    logger.info("Connected to MySQL")

    # 트랜젝션 세션 생성
    Session = sessionmaker(bind=engine)
    session = Session()

    # 데이터 저장
    try:
        logger.info("Saving data...")
        df = pd.DataFrame([data] if isinstance(data, dict) else data)
        logger.info(f"Dataframe shape: {df.shape}")
        
        df.to_sql(table_name, engine, if_exists='append', index=False)
        logger.info(f"Data saved to MySQL table {table_name}")
        
    except Exception as e:
        
        logger.error(f"Error occurred: {e}")
        
        # 롤백
        session.rollback()

    finally:
        # 연결 종료
        session.close()
        