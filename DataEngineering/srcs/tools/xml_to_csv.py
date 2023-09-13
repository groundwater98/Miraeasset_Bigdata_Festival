import pandas as pd
import xml.etree.ElementTree as ET

def xml_to_csv(xml_filepath, csv_filepath):
    """_summary_
    파일 경로를 기반으로 XML 파일을 CSV 파일로 변환합니다.
    """
    tree = ET.parse(xml_filepath)
    root = tree.getroot()

    data = []

    # XML 구조에 따라 데이터를 추출합니다.
    for item in root:
        row_data = {}
        for child in item:
            row_data[child.tag] = child.text
        data.append(row_data)

    # 데이터 프레임으로 변환
    df = pd.DataFrame(data)

    # CSV 파일로 저장
    df.to_csv(csv_filepath, index=False)

# 사용 예시
xml_to_csv('/Users/user/repo/miraeasset-festa/DataEngineering/data/dart/corpcode.xml', "output.csv")