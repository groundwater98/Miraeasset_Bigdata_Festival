import os
import toml
from tools import logger
from api_handler.naver.handler import NaverAPIHandler

logger = logger.Logger(__name__).get_logger()

def load_config(file_path):
    with open(file_path, 'r') as f:
        return toml.load(f)

def main():
    config = load_config(os.getenv('NAVER_CONFIG_PATH'))
    headers = {
        'X-Naver-Client-Id': os.getenv('NAVER_CLIENT_ID'),
        'X-Naver-Client-Secret': os.getenv('NAVER_CLIENT_SECRET'),
    }
    logger.debug("headers: %s", headers)
    naver_api_handler = NaverAPIHandler()
    naver_api_handler.init_config({
        "url": config["url"]["news"],
        "query": config["search"]["query"],
        "start": config["search"]["start"],
        "sort": config["search"]["sort"],
    })
    naver_api_handler.init_credential(headers)
    naver_api_handler.init_error_codes(config["news_error_code"])
    response = naver_api_handler.get_response()
    print(response)

if __name__ == "__main__":
    main()