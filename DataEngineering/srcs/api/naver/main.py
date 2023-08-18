import os
import toml
from tools import logger, csv
from api.naver.handler import NaverAPIHandler

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
    queries = config["search"]["query"]
    logger.debug("headers: %s", headers)
    for query in queries:
        logger.debug("query: %s", query)
        naver_api_handler = NaverAPIHandler()
        naver_api_handler.init_configs({
            "url": config["url"]["news"],
            "start": config["search"]["start"],
            "display": config["search"]["display"],
            "sort": config["search"]["sort"],
        })
        naver_api_handler.init_credential(headers)
        naver_api_handler.init_error_codes(config["error_messages"])
        response = naver_api_handler.get_response(query)
        logger.debug("response: %s", response)
        output_file_dir = config["output"]["path"]
        output = output_file_dir + f"{query}.csv"
        csv.json_to_csv(response, output)
        
if __name__ == "__main__":
    main()