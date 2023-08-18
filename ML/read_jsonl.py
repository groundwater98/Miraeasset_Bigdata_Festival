# jsonl 파일 잘못 만들어졌는지 확인할려고 작성한 코드

import json

with open("customer.jsonl") as f:
    for i, line in enumerate(f): print(i,line)