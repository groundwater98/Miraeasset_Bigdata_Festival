# The finetune dataset to chatGPT generation code 
import random

filename = 'customer.jsonl'

try:
    with open(filename, 'w', encoding='utf-8') as file:
        for _ in range(1000):
            asset = random.randint(0,9999999)
            trade = random.randint(0,9999)
            Qs = [f"\"I trade {trade} stocks per week and have ${asset}.\"", 
                f"\"With ${asset}, I trade {trade} stocks each week.\"",
                f"\"I invest ${asset} weekly in {trade} stocks.\"",
                f"\"I invest ${asset} and trade {trade} equities each week.\"",
                f"\"I have ${asset} and trade {trade} equities each week.\"",
                f"\"With ${asset}, I trade {trade} stocks each week.\"",
                f"\"I invest ${asset} weekly in {trade} stocks.\"",
                f"\"I have {asset} dollars and trade {trade} equities every week.\"",
            ]
            rich, offensive = False if asset < 100_000_000 else True, False if trade < 30 else True
            As = {
                (True, True): ["\"You are a wealthy, aggressive investor.\"", "\"You are a successful and ambitious investor.\"", "\"You are a successful, competitive investor.\""],
                (True, False): ["\"You are a wealthy, safe investor.\"", "You are a successful, reliable investor.\"", "\"You are a well-off, dependable investor.\""],
                (False, True): ["\"You are not wealthy, but you are an aggressive investor.\"", "\"Despite not being wealthy, you are a risk-taking investor.\"", "\"Although you are not wealthy, you are a bold investment.\""],
                (False, False): ["\"You are not wealthy, but you are a stable investor.\"", "\"You are not rich, you are a stable investor.\"", "\"You are a reliable investor; you are not wealthy.\""]
            }
            Q, A = random.choice(Qs), random.choice(As[(rich, offensive)])
            data = f'"prompt": {Q}, "completion": {A}'
            file.write("{"+data+"}\n")
    print(f"data_generation success!!")
except Exception as e:
    print(f"{e}, data_generation Failed!!")