import os
import toml
from tools import logger
from api.dart.handler import DartAPIHandler

logger = logger.Logger(__name__).get_logger()

def load_config(file_path):
    with open(file_path, 'r') as f:
        return toml.load(f)

def main():
    config = load_config(os.getenv('DART_CONFIG_PATH'))
    
    configs = {
        "url": config["args"]["url"],
        "corp_code": config["args"]["corp_code"],
        "bsns_year": config["args"]["bsns_year"],
        "reprt_code": config["args"]["reprt_code"],
    }
    
    credentials = {
        "api_key": os.getenv('DART_APIKEY')
    }

    error_msgs = config["error_messages"]

    dart_handler = DartAPIHandler(configs, credentials, error_msgs)
    res = dart_handler.get_response()
    logger.debug(res)
        
if __name__ == "__main__":
    main()