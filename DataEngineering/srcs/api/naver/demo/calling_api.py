import os
import requests
import toml
from tools import logger

logger = logger.Logger(__name__).get_logger()
config = toml.load(os.getenv('NAVERAPI_CONFIG_PATH'))

url = config['url']['news']
query = "aapl"

params = {
    "query": query,
    "display": 10,
    "start": 10,
    "sort": "date"
}

headers = {
    "X-Naver-Client-Id": os.environ.get("NAVER_CLIENT_ID"),
    "X-Naver-Client-Secret": os.environ.get("NAVER_CLIENT_SECRET")
}

response = requests.get(url, params=params, headers=headers, timeout=30)

error_codes = {
    "SE01": "Incorrect query request. Please check the API request URL protocol, parameters, etc.",
    "SE02": "Invalid display value. Please make sure display parameter value is within the allowed range (1~100).",
    "SE03": "Invalid start value. Please make sure start parameter value is within the allowed range (1~1000).",
    "SE04": "Invalid sort value. Please check if there is a typo in the sort parameter value.",
    "SE06": "Malformed encoding. Please ensure the search term is in UTF-8 encoding.",
    "SE05": "Invalid search API. Please check if there is a typo in the API request URL.",
    "SE99": "System Error. Please report the error to the 'Developer Forum'.",
}

if response.status_code == 200:
    logger.info(response.text)
else:
    error_message = response.json().get('errorMessage')
    error_code = response.json().get('errorCode')
    logger.fatal("Error: %s", error_codes.get(error_code, 'Unknown error. Message: %s' % error_message))
