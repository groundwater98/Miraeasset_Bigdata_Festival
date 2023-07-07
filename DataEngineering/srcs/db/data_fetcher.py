import os
from db.database_manager import DatabaseManager

db_password = os.environ.get("DB_PASSWORD")
db_manager = DatabaseManager(f'postgresql://postgres:{db_password}@localhost:5432/postgres')

db_manager.save_historical_data('AAPL', period='1y', interval='1d')