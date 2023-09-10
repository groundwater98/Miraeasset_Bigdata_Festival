
'''
Stocks
'''

import random
import csv
import pandas as pd

# 패턴 예제
sentence_patterns = [
    "{}의 부채는 총 얼마야?",
    "{}의 부채를 알려줘",
    "{}의 부채는 얼마나 돼?",
    "{}의 부채총계를 알려줘",

    "{}의 자산는 총 얼마야?",
    "{}의 자산 보유랑을 알려줘",
    "{}의 총 자산은 얼마냐?",
    "{}의 자산총계를 알려줘",


    "{}년도 기준으로 {}의 부채는 얼마야?",
    "{}년도 {}의 부채를 알려줘",
    "{}년도부터 {}의 부채를 보여줘?",

    "{}년도 기준으로 {}의 자산은 얼마야?",
    "{}년도 {}의 자산을 알려줘",
    "{}년도부터 {}의 자산를 보여줘?",

    "{}의 최근 주식 시가는 얼마야?",
    "{}의 주식 시가를 알려줘",

    "{}의 주식 최고가는 얼마야?",
    "{}의 주식 최저가는 얼마야?",

    "{} 주식의 변화량을 알려줘"

]

sql_patterns = [
    "Select finance_statement, class2, Date, company FROM AnnualFinanceStatement WHERE (company = '{}') and (class2 = '부채총계')",
    "Select finance_statement, class2, Date, company FROM AnnualFinanceStatement WHERE (company = '{}') and (class2 = '부채총계')",
    "Select finance_statement, class2, Date, company FROM AnnualFinanceStatement WHERE (company = '{}') and (class2 = '부채총계')",
    "Select finance_statement, class2, Date, company FROM AnnualFinanceStatement WHERE (company = '{}') and (class2 = '부채총계')",

    "Select finance_statement, class2, Date, company FROM AnnualFinanceStatement WHERE (company = '{}') and (class2 = '자산총계')",
    "Select finance_statement, class2, Date, company FROM AnnualFinanceStatement WHERE (company = '{}') and (class2 = '자산총계')",
    "Select finance_statement, class2, Date, company FROM AnnualFinanceStatement WHERE (company = '{}') and (class2 = '자산총계')",
    "Select finance_statement, class2, Date, company FROM AnnualFinanceStatement WHERE (company = '{}') and (class2 = '자산총계')",

    "Select finance_statement, class2, Date, company FROM AnnualFinanceStatement WHERE DATE = '{}1231' and company = '{}' and class2 = '부채총계'",
    "Select finance_statement, class2, Date, company FROM AnnualFinanceStatement WHERE DATE = '{}1231' and company = '{}' and class2 = '부채총계'",
    "Select finance_statement, class2, Date, company FROM AnnualFinanceStatement WHERE DATE = '{}1231' and company = '{}' and class2 = '부채총계'",

    "Select finance_statement, class2, Date, company FROM AnnualFinanceStatement WHERE DATE = '{}1231' and company = '{}' and class2 = '자산총계'",
    "Select finance_statement, class2, Date, company FROM AnnualFinanceStatement WHERE DATE = '{}1231' and company = '{}' and class2 = '자산총계'",
    "Select finance_statement, class2, Date, company FROM AnnualFinanceStatement WHERE DATE = '{}1231' and company = '{}' and class2 = '자산총계'",

    "Select company, stck_bsop_date, stck_oprc from StockDailyData where company = '{}'",
    "Select company, stck_bsop_date, stck_oprc from StockDailyData where company = '{}'",

    "Select company, stck_bsop_date, stck_hgpr from StockDailyData where company = '{}'",
    "Select company, stck_bsop_date, stck_lwpr from StockDailyData where company = '{}'",

    "Select company, stck_bsop_date, stck_oprc, stcl_hgpr, stck_lwpr from StockDailyData where company = '{}'"

]

companies =["JYP", "SK하이닉스", "삼성SDI", "삼성바이오로직스", "에코프로", "에코프로비엠", "엘앤에프", "펄어비스", "네이버", "카카오"]
# 2022년 1월 ~ 2023년 9월 5일
def generate_sql_sentences():
    sentences = []
    queries = []
    
    for _ in range(1000):
        company = random.choice(companies)
        idx = random.randint(0, len(sentence_patterns) - 1)
        
        # 문장 패턴에 따라 필요한 값들을 추출
        num_args = sentence_patterns[idx].count("{}")
        
        # {}의 개수 확인
        if num_args == 1:
            values = [company]
        elif num_args == 2:
            values = [company, random.randint(2016, 2022)]

        sentence = sentence_patterns[idx].format(*values)
        query = sql_patterns[idx].format(*values)

        sentences.append(sentence)
        queries.append(query)

    return sentences, queries

sentences, queries = generate_sql_sentences()

# 예제 출력
for i in range(5):
    print(f"Sentence: {sentences[i]}")
    print(f"SQL: {queries[i]}")
    print('-' * 50)
# 데이터를 섞는다.

combined = list(zip(sentences, queries))
random.shuffle(combined)

# 데이터를 훈련용, 검증용, 테스트용으로 분할 (예: 80%, 10%, 10%)
train_size = int(0.8 * len(combined))
val_size = int(0.1 * len(combined))

train_data = combined[:train_size]
val_data = combined[train_size:train_size+val_size]
test_data = combined[train_size+val_size:]

# CSV 파일로 저장

# train
with open('/mnt/c/Users/starp/future_asset/Seq2seqLSTM_2/train.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["source", "target"])  # header
    writer.writerows(train_data)

# val
with open('/mnt/c/Users/starp/future_asset/Seq2seqLSTM_2/val.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["source", "target"])  # header
    writer.writerows(val_data)

# test
with open('/mnt/c/Users/starp/future_asset/Seq2seqLSTM_2/test.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["source", "target"])  # header
    writer.writerows(test_data)

print("Dataset is ready")

