import json
import csv
from tools import logger

logger = logger.Logger(__name__).get_logger()

def json_to_csv(json_data: str, csv_file_path: str):
    
    logger.debug("loads json data")
    data = json.loads(json_data)

    logger.info("write data to csv file")
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(data[0].keys()))
        writer.writeheader()

        for row in data:
            writer.writerow(row)
    logger.debug("complete writing data to csv file")