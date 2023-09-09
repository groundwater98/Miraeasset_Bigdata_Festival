import requests, json, html
from datetime import datetime
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from api.handler import APIHandler
from tools import logger

logger = logger.Logger(__name__).get_logger()

class DartRequestArgValidator:
    pass

class DartAPIHandler(APIHandler):
    def __init__(self, configs: dict = None, credentials: dict = None, error_codes: dict = None):
        """_summary_
        constructor for the Dart API Handler
        
        Args:
            configs (dict, optional): dictionary used to initialize the Dart API Handler. Defaults to None.
            - url: str (url of the Dart API)
            - corp_code: list (list of the corporation codes)
            - bsns_year: str (business year)
            - reprt_code: str (report code)
            
            credentials (dict, optional): dictionary used to initialize the Dart API Handler. Defaults to None.
            - api_key: str (api key for the Dart API)
            
            error_codes (dict, optional): dictionary used to initialize the Dart API Handler. Defaults to None.
        """
        if any([configs is None, credentials is None, error_codes is None]):
            logger.fatal("Error: %s", "Please set the configs, credentials, and error_codes for the Dart API.")

        super().__init__()
        
        self.configs = {
            "url": "",              # 공시검색 API URL
            "crtfc_key": "",        # API 인증키    
            "corp_code": [],        # 공시대상회사의 종목코드 (다중 회사 검색 가능)
            "bsns_year": "",        # 사업연도
            "reprt_code": "",       # 보고서 코드
        }
        
        self.init_configs(configs)
        
        self.init_credential(credentials)


    def init_configs(self, configs: dict):
        """_summary_
        init the configs for the Dart API
        
        Args:
            configs (dict): dictionary with fields "url", "corp_code", "bsns_year", "reprt_code"
            - url: str (url of the Dart API)
            - corp_code: list (list of the corporation codes)
            - bsns_year: str (business year)
            - reprt_code: str (report code)
        """
        if configs["url"] == "":
            logger.fatal("Error: %s", "Please set the configs for the Dart API url.")
            
        self.configs = {
            "url": configs["url"],
            "corp_code": configs["corp_code"],
            "bsns_year": configs["bsns_year"],
            "reprt_code": configs["reprt_code"],
        }
        logger.debug("successfully initialized the configs for the Dart API")


    def init_credential(self, credentials: dict):
        """_summary_
        init the credentials for the Dart API

        Args:
            credentials (dict): dictionary with single field "api_key"
        """
        if credentials["api_key"] == "":
            logger.fatal("Error: %s", "Please set the headers for the Naver API.")  
                      
        self.configs["crtfc_key"] = credentials["api_key"]
        logger.debug("successfully initialized the credentials for the Dart API")


    def init_error_codes(self, error_codes: dict):
        """_summary_
        init the error codes for the Dart API
        
        Args:
            error_codes (dict): dictionary with fields "000", "010", "011", "013", "020", "100", "800", etc ...
        """
        self.error_codes = error_codes
        logger.debug("successfully initialized the error codes for the Dart API")


    def get_response(self):
        """_summary_
        get the response from the Dart API
        
        Returns:
            json: if the response is successful, otherwise None
        """
        logger.debug("headers: %s", self.headers)
        logger.debug("configs: %s", self.configs)
        
        response = requests.get(
            self.configs['url'],
            params={
                "corp_code": self.configs['corp_code'],
                "bsns_year": self.configs['bsns_year'],
                "reprt_code": self.configs['reprt_code'],
                "crtfc_key": self.configs['crtfc_key'],
            },
            
            timeout=30
        )

        logger.debug("response: %s", response)
           
        res_json = response.json()
        
        if res_json['status'] == '000':
            logger.debug("successfully got the response from the Dart API")
            return response.json()
        
        logger.error("Error code: %s", res_json['status'])
        return None