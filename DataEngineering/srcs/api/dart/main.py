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
    # queries = config["search"]["query"]
    # logger.debug("headers: %s", headers)
    # for query in queries:
    #     logger.debug("query: %s", query)
    #     naver_api_handler = NaverAPIHandler()
    #     naver_api_handler.init_configs({
    #         "url": config["url"]["news"],
    #         "start": config["search"]["start"],
    #         "display": config["search"]["display"],
    #         "sort": config["search"]["sort"],
    #     })
    #     naver_api_handler.init_credential(headers)
    #     naver_api_handler.init_error_codes(config["error_messages"])
    #     response = naver_api_handler.get_response(query)
    #     logger.debug("response: %s", response)
    #     output_file_dir = config["output"]["path"]
    #     output = output_file_dir + f"{query}.csv"
    #     csv.json_to_csv(response, output)
        
if __name__ == "__main__":
    main()