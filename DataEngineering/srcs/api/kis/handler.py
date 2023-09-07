import requests
import json
import os
import pandas as pd
from datetime import datetime, timedelta
from tools.logger import Logger
from db.data_fetcher import save_to_mysql_auto, clear_table_contents

logger = Logger(__name__).get_logger()

KIS_APPKEY = os.environ.get('KIS_APPKEY')
KIS_SECRET = os.environ.get('KIS_SECRET')
URL_BASE = "https://openapivts.koreainvestment.com:29443"

def get_access_token():                                 # POST 방식
    headers = {"content-type":"application/json"}       # 기본정보(Content-Type)
    body = {
        "grant_type":"client_credentials",          # R : 권한부여 타입
        "appkey":KIS_APPKEY,                                  
        "appsecret":KIS_SECRET
    }                                   
    
    PATH = "oauth2/tokenP"                                              # 기본정보(URL) : URL
    
    URL = f"{URL_BASE}/{PATH}"                                          # 기본정보(도메인) : https://openapivts.koreainvestment.com:29443/oauth2/tokenP
    
    res = requests.post(URL, headers=headers, data=json.dumps(body))	# Response 데이터 호출
    
    ACCESS_TOKEN = res.json()["access_token"]
    return ACCESS_TOKEN


def inqure_daily_itemprice(ACCESS_TOKEN:str, fid_input_iscd:str="", save=None):
    
    def date_range(start, end, interval=timedelta(days=100)):
        current = start
        while current < end:
            next_date = current + interval
            if next_date > end:
                next_date = end
            yield (current, next_date)
            current = next_date + timedelta(days=1)
    
    if fid_input_iscd == "":
        raise ValueError("fid_input_iscd is empty")
    
    tr_id = "FHKST03010100"
    custtype = "P"
    
    headers = {
        "content-type":"application/json",
        "authorization": f"Bearer {ACCESS_TOKEN}",
        "appkey": KIS_APPKEY,
        "appsecret": KIS_SECRET,
        "tr_id": tr_id,
        "custtype": custtype
    }
    FID_INPUT_DATE_1 = "20220101"
    FID_INPUT_DATE_2 = "20230907"
    start_date = datetime.strptime(FID_INPUT_DATE_1, '%Y%m%d')
    end_date = datetime.strptime(FID_INPUT_DATE_2, '%Y%m%d')
    
    FID_COND_MRKT_DIV_CODE = "J"
    FID_PERIOD_DIV_CODE = "D"
    FID_ORG_ADJ_PRC = "1"
    
    for start, end in date_range(start_date, end_date):
        params = {
            "FID_COND_MRKT_DIV_CODE": FID_COND_MRKT_DIV_CODE,
            "FID_INPUT_ISCD": fid_input_iscd,
            "FID_INPUT_DATE_1": start.strftime('%Y%m%d'),
            "FID_INPUT_DATE_2": end.strftime('%Y%m%d'),
            "FID_PERIOD_DIV_CODE": FID_PERIOD_DIV_CODE,
            "FID_ORG_ADJ_PRC": FID_ORG_ADJ_PRC
        }
    
        PATH = '/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice'
        URL = f"{URL_BASE}{PATH}"
        res = requests.get(URL, headers=headers, params=params, timeout=600)
        data = res.json()
        
        save = True
        # dump json into flle
        if save:
            with open(f'./data/{fid_input_iscd}.json', 'w', encoding='utf-8') as f:
                json.dump(res.json(), f)
        
        if 'itewhol_loan_rmnd_ratem name' in data['output1']:
            data['output1']['itewhol_loan_rmnd_ratem_name'] = data['output1'].pop('itewhol_loan_rmnd_ratem name')
        
        # Full Refresh, output1 데이터를 데이터프레임으로 변환 후 'StockBasicInfo' 테이블에 저장
        save_to_mysql_auto(data['output1'], 'StockBasicInfo')
        logger.info(f"{fid_input_iscd} output1 data saved to StockBasicInfo")

        # Full Refresh, output2 데이터를 데이터프레임으로 변환 후 'StockDailyData' 테이블에 저장
        df_output2 = pd.DataFrame(data['output2'])
        df_output2['stock_code'] = fid_input_iscd
        save_to_mysql_auto(df_output2, 'StockDailyData')
        logger.info(f"{fid_input_iscd} output2 data saved to StockDailyData")

    return True


def extract_fs():
    access_token = get_access_token()
    stock_codes = {
        "JYP": "035900",
        "SK하이닉스": "000660",
        "삼성SDI": "006400",
        "삼성바이오로직스": "207940",
        "삼성전자우": "005935",
        "셀트리온제약": "068760",
        "셀트리온헬스케어": "091990",
        "에코프로": "086520",
        "에코프로비엠": "247540",
        "엘앤에프": "066970",
        "펄어비스": "263750",
        "현대차": "005380",
        "삼성전자": "005930",
        "카카오": "035720",
        "네이버": "035420",
    }
    
    
    clear_table_contents('StockBasicInfo')
    clear_table_contents('StockDailyData')
    
    for stock_code in stock_codes.values():
        logger.info(f"Extracting {stock_code}...")
        inqure_daily_itemprice(access_token, stock_code)

extract_fs()
