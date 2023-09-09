import os
import re
import pandas as pd
import dart_fss as dart
import dart_fss.errors
from tools import logger
from db.data_fetcher import save_to_mysql_auto, clear_table_contents

API_KEY = os.getenv('DART_APIKEY')
dart.set_api_key(api_key=API_KEY)

Logger = logger.Logger(__name__).get_logger()

def reshape_dataframe(df: pd.DataFrame, _corp_code) -> pd.DataFrame:
    Logger.info(f"Reshaping dataframe for corp_code: {_corp_code}")
    
    # MultiIndex를 단일 인덱스로 변환
    df.columns = ['_'.join(map(str, col)).strip() for col in df.columns.values]
    
    # melted_data 설정
    melted_data = [col for col in df.columns if col.startswith("2022") \
            or col.startswith("2021") \
            or col.startswith("2020") \
            or col.startswith("2019") \
            or col.startswith("2018") \
            or col.startswith("2017") \
            or col.startswith("2016") \
            or col.startswith("2015")]
    
    # unmelted_data 설정
    unmelted_data = [col for col in df.columns if col not in melted_data]

    # pd.melt()를 사용하여 데이터프레임의 구조를 변경
    Logger.info(f"Reshaping dataframe for corp_code: {_corp_code}")
    reshaped_df = pd.melt(df, id_vars=unmelted_data,
                          value_vars=melted_data, 
                          var_name="Date", 
                          value_name="finance_statement")
    
    # 컬럼명에서 불필요한 문자열을 제거
    Logger.info(f"Removing unnecessary strings from column names for corp_code: {_corp_code}")
    prefix = "[D210000] Statement of financial position, current/non-current - Consolidated financial statements (Unit: KRW)_"
    reshaped_df.columns = [col.replace(prefix, '') for col in reshaped_df.columns]
    
    # 추가 데이터 정제
    Logger.info(f"Refining additional data for corp_code: {_corp_code}")
    reshaped_df['Date'] = reshaped_df['Date'].str.extract(r'(\d{8})')
    reshaped_df = reshaped_df.sort_values(by=['Date'])
    reshaped_df.drop('class0', axis=1, inplace=True)
    reshaped_df["class1"] = reshaped_df["class1"].str.split(' ').str[0]
    
    return reshaped_df

def extract_fs(_corp_code: str, bgn_de: str, save=False):
    try:
        fs = dart.fs.extract(
                corp_code=_corp_code,
                bgn_de=bgn_de,
            )
        fs['bs'].to_csv(f'{_corp_code}.csv', index=False)
        # 데이터프레임의 구조를 변경합니다.
        reshaped_bs = reshape_dataframe(fs['bs'], _corp_code)
        reshaped_bs['corp_code'] = str(_corp_code)
        
        # 변경된 데이터프레임을 CSV 파일로 저장합니다.
        if save:
            reshaped_bs.to_csv(f'./data/{_corp_code}.csv', index=False)
            Logger.info(f"Saved {_corp_code}.csv")
        
        # 데이터프레임을 MySQL 테이블에 저장합니다.
        save_to_mysql_auto(reshaped_bs, 'AnnualFinanceStatement')
        return fs
    except dart_fss.errors.NoDataReceived as err:
        Logger.error(f"{err}: No data received for corp_code: {_corp_code}")
        pass
    except dart_fss.errors.NotFoundConsolidated as err:
        Logger.error(f"{err}: No consolidated data for corp_code: {_corp_code}")
        pass

def main():
    directory_path = '../../../data/dart/corp_codes'
    file_names = os.listdir(directory_path)
    filtered_files = [f for f in file_names \
                    if f.startswith('corp_codes') and \
                        f.endswith('.txt')]

    stock_codes = {
        "JYP": "035900",
        "SK하이닉스": "000660",
        "삼성SDI": "006400",
        "삼성바이오로직스": "207940",
        "셀트리온제약": "068760",
        "셀트리온헬스케어": "091990",
        "에코프로": "086520",
        "에코프로비엠": "247540",
        "엘앤에프": "066970",
        "펄어비스": "263750",
        "현대차": "005380",
        "삼성전자": "005930",
        "네이버": "035420",
        "카카오": "035720"
    }

    # 테이블 초기화
    Logger.info("Clearing table contents...")
    clear_table_contents('AnnualFinanceStatement')
    
    # 재무제표 추출 및 적재
    for code in stock_codes.values():
        Logger.info(f"Extracting financial statements for {code}...")
        extract_fs(_corp_code=code, bgn_de='20220101')
        
    
main()


# # Open DART API KEY 설정
# API_KEY=os.getenv('DART_APIKEY')
# dart.set_api_key(api_key=API_KEY)
# corp_list = dart.get_corp_list()

# samsung = corp_list.find_by_corp_name('삼성전자', exactly=True)[0]

