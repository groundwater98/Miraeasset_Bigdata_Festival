import os
import toml
from tools.logger import Logger
from db.database_manager import DatabaseManager

config_path = os.getenv('DB_CONFIG_PATH')
logger = Logger(__name__).get_logger()

def initialize_database():
    try:
        with open(config_path, "r") as f:
            data = toml.load(f)
        db_name = data["database"]["db_name"]
        db_password = data["database"]["db_password"]
        db_username = data["database"]["db_username"]
        docker_host = data["database"]["docker_host"]
        db_port = data["database"]["db_port"]
        db_url = f"postgresql://{db_username}:{db_password}@{docker_host}:{db_port}/{db_name}"
        db_manager = DatabaseManager(db_url)
        return db_manager
    except (FileNotFoundError, toml.TomlDecodeError) as e:
        logger.error("Error occurred while loading the configuration file: %s", str(e))
    
    logger.fatal("Terminate program with the unexpected action(s)")
    raise SystemExit

def app():
    db_manager = initialize_database()
    db_manager.create_all_table()
    
if __name__ == '__main__':
    app()

# db_manager.save_historical_data('AAPL', period='1y', interval='1d')