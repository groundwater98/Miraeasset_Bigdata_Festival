import requests, json, html
import xml.etree.ElementTree as ET
from api_handler.handler import APIHandler
from bs4 import BeautifulSoup
from tools import logger

logger = logger.Logger(__name__).get_logger()

class NaverAPIHandler(APIHandler):
    def __init__(self):
        self.config = {
            "url": "",
            "query": "",
            "start": "",
            "sort": "",
        }
        
        self.headers = {
            "X-Naver-Client-Id": "",
            "X-Naver-Client-Secret": "",
        }
        
        self.error_codes = None
    
    
    def init_config(self, config):
        if config["url"] == "" or config["query"] == "":
            logger.fatal("Error: %s", "Please set the config for the Naver API.")
            
        self.config = {
            "url": config["url"],
            "query": config["query"],
            "start": config["start"],
            "sort": config["sort"],
        }
    
    
    def init_credential(self, headers):
        if headers["X-Naver-Client-Id"] == "" or headers["X-Naver-Client-Secret"] == "":
            logger.fatal("Error: %s", "Please set the headers for the Naver API.")            
        self.headers['X-Naver-Client-Id'] = headers["X-Naver-Client-Id"]
        self.headers['X-Naver-Client-Secret'] = headers["X-Naver-Client-Secret"]
    
    
    def init_error_codes(self, error_codes):
        self.error_codes = error_codes
    
    
    def get_response(self):
        response = requests.get(
            self.config['url'], 
            params={
                "query": self.config['query'], 
                "start": self.config['start'], 
                "sort": self.config['sort']
            }, 
            headers=self.headers,
            timeout=30
        )        
        
        items = []
        if response.status_code == 200:
            root = ET.fromstring(response.text)
            for item in root.iter('item'):
                title = BeautifulSoup(html.unescape(item.find('title').text), 'html.parser').get_text()
                link = item.find('link').text
                description = BeautifulSoup(item.find('description').text, 'html.parser').get_text()
                item_dict = {
                    "title": title,
                    'link': link,
                    'description': description,
                }
                print(item_dict)
                items.append(item_dict)
            return json.dumps(items, indent=4, ensure_ascii=False)
        
        else:
            error_message = response.json().get('errorMessage')
            error_code = response.json().get('errorCode')
            logger.fatal("Error: %s", self.error_codes.get(error_code, 'Unknown error. Message: %s' % error_message))
        
        
         
        
    