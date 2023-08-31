import requests, json, html
from datetime import datetime
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from api.handler import APIHandler
from tools import logger

logger = logger.Logger(__name__).get_logger()

class NaverAPIHandler(APIHandler):
    def __init__(self):
        
        super().__init__()
        
        self.configs = {
            "url": "",
            "query": "",
            "display": "",
            "start": "",
            "sort": "",
        }
        
        self.headers = {
            "X-Naver-Client-Id": "",
            "X-Naver-Client-Secret": "",
        }
        
        self.error_codes = None
    
    
    def init_configs(self, configs):
        if configs["url"] == "":
            logger.fatal("Error: %s", "Please set the configs for the Naver API url.")
            
        self.configs = {
            "url": configs["url"],
            "display": configs["display"],
            "start": configs["start"],
            "sort": configs["sort"],
        }
    
    
    def init_credential(self, credentials):
        if credentials["X-Naver-Client-Id"] == "" or credentials["X-Naver-Client-Secret"] == "":
            logger.fatal("Error: %s", "Please set the headers for the Naver API.")            
        self.headers['X-Naver-Client-Id'] = credentials["X-Naver-Client-Id"]
        self.headers['X-Naver-Client-Secret'] = credentials["X-Naver-Client-Secret"]
    
    
    def init_error_codes(self, error_codes):
        self.error_codes = error_codes
        
    
    def get_response(self, query: str = None):
        """
        return: json string
        title: str (title of the news)
        link: str (url of news)
        description: str (description of the news)
        pubdate: isoformat (publication date of the news)
        """        
        logger.debug("headers: %s", self.headers)
        logger.debug("configs: %s", self.configs)
        response = requests.get(
            self.configs['url'], 
            params={
                "query": query,
                "display": self.configs['display'],
                "start": self.configs['start'], 
                "sort": self.configs['sort']
            }, 
            headers=self.headers,
            timeout=30
        )        
        items = []
        if response.status_code == 200:
            root = ET.fromstring(response.content)
            for item in root.iter('item'):
                
                title = None
                link = None
                description = None
                date_object = None
                
                if item is None:
                    continue
                # item_xml = ET.tostring(item, encoding='unicode')
                
                title_element = item.find('title')
                if title_element is not None:
                    title = BeautifulSoup(html.unescape(item.find('title').text), 'html.parser').get_text()
                
                link = item.find('link').text
                
                try:
                    description_element = item.find('description')
                
                    if description_element is not None:
                        description = BeautifulSoup(item.find('description').text, 'html.parser').get_text()
                except TypeError:
                    description = ""

                pubdate_element = item.find('pubDate')
                if pubdate_element is not None:
                    pubdate = pubdate_element.text
                    date_format = "%a, %d %b %Y %H:%M:%S %z"
                    date_object = datetime.strptime(pubdate, date_format).isoformat()
                    
                item_dict = {
                    "query": query,
                    "title": title if title is not None else '',
                    'link': link if link is not None else '',
                    'description': description if description is not None else '',
                    'pubdate': date_object if date_object is not None else '',
                }
                items.append(item_dict)
            return json.dumps(items, indent=4, ensure_ascii=False)
        else:
            root = ET.fromstring(response.text)
            error_code = root.find('errorCode').text
            error_message = root.find('errorMessage').text
            logger.fatal("Error: %s", self.error_codes.get(error_code, 'Message: %s' % error_message))
            return None