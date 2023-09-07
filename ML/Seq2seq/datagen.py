import random
import csv
import pandas as pd

# 패턴 예제
sentence_patterns = [
    "{}의 최신 소식을 알려줘",
    "{} 관련 최근 뉴스를 알려줘",
    "{}의 최신 뉴스 링크를 보여줘",
    "{}의 최근 소식을 알려줘",
    
    "{}의 지난 {}일 동안의 뉴스를 요약해줘",
    "{}의 지난 {}일 간의 뉴스를 요약해줘",
    "{}의 최근 {}일 간의 뉴스를 요약해줘",

    "{}의 최신 뉴스 {}개를 요약해줘",
    "{} 뉴스의 최근 제목 {}개를 알려줘",

    "{}의 {}일 전 뉴스 제목을 알려줘",
    
    "{}의 {}월 {}일자 뉴스를 알려줘",
    "{}에 대한 {}월 {}일의 뉴스 제목을 알려줘",
    "{} {}월 {}일의 뉴스 링크를 알려줘",
    "{}에서 {}월 {}일에 발표한 뉴스가 뭐야?",
    "{}의 {}월 {}일에 대한 뉴스 내용을 보여줘",
    "{}에서 {}월 {}일에 발표한 뉴스가 뭐야?",
]

sql_patterns = [
    "SELECT title, link, description, pubdate FROM {} ORDER BY pubdate DESC LIMIT 1",
    "SELECT title, link, description, pubdate FROM {} ORDER BY pubdate DESC LIMIT 1",
    "SELECT title, link, description, pubdate FROM {} ORDER BY pubdate DESC LIMIT 1",
    "SELECT title, link, description, pubdate FROM {} ORDER BY pubdate DESC LIMIT 1",

    "SELECT title, link ,description FROM {} ORDER BY pubdate DESC LIMIT {}",
    "SELECT title, link ,description FROM {} ORDER BY pubdate DESC LIMIT {}",
    "SELECT title, link ,description FROM {} ORDER BY pubdate DESC LIMIT {}",


    "SELECT title, description FROM {} ORDER BY pubdate DESC LIMIT {}",
    "SELECT title, description FROM {} ORDER BY pubdate DESC LIMIT {}",

    "SELECT title, description FROM {} WHERE pubdate = NOW() - INTERVAL {} DAY",

    "SELECT title, link, description FROM {} WHERE MONTH(pubdate) = {} AND DAY(pubdate) = {}",
    "SELECT title, link, description FROM {} WHERE MONTH(pubdate) = {} AND DAY(pubdate) = {}",
    "SELECT title, link, description FROM {} WHERE MONTH(pubdate) = {} AND DAY(pubdate) = {}",
    "SELECT title, link, description FROM {} WHERE MONTH(pubdate) = {} AND DAY(pubdate) = {}",
    "SELECT title, link, description FROM {} WHERE MONTH(pubdate) = {} AND DAY(pubdate) = {}",
    "SELECT title, link, description FROM {} WHERE MONTH(pubdate) = {} AND DAY(pubdate) = {}",
]

companies = ["HLB", "JYP", "LG에너지솔루션", "SK하이닉스", "삼성SDI", "삼성바이오로직스", "삼성전자우", "셀트리온제약", "셀트리온헬스케어", "에코프로", "에코프로비엠", "엘앤에프","POSCO홀딩스", "펄어비스", "포스코DX", "포스코퓨처엠", "현대차", "삼성전자", "애플", "네이버", "카카오"]

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
            values = [company, random.randint(1, 30)]
        elif num_args == 3:
            values = [company, random.randint(1, 12), random.randint(1, 28)]
        
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
with open('train.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["source", "target"])  # header
    writer.writerows(train_data)

# val
with open('val.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["source", "target"])  # header
    writer.writerows(val_data)

# test
with open('test.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["source", "target"])  # header
    writer.writerows(test_data)

print("Dataset is ready")

