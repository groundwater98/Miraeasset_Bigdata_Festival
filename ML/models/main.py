import openai
import torch
from Seq2seqLSTM import Encoder, Decoder, Attention, Seq2Seq
import pandas as pd
from training import CustomDataset

openai.api_key = "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"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Inference를 위한 Dataset => trained model로 새로운 입력 문장을 생성할 때 이 Dataset을 사용하여 입력 문장을 적합한 형태로 변환하고, 모델의 출력을 다시 사람이 이해할 수 있는 형태로 변환한다.
class CustomInferenceDataset: 
    def __init__(self, vocab):
        self.word2idx = vocab
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.pad_idx = 0

    # 주어진 sentence를 tokenize하고 각 토크늘 해당 index로 Encoding. 만약 주어진 token이 어휘 사전에 없다면 pad_idx로 대체
    def tokenize_and_encode(self, sentence):
        tokens = sentence.split() 
        return [self.word2idx.get(token, self.pad_idx) for token in tokens]  

    # 주어진 index의 token를 단어들의 list로 Decoding 그리고 문자열로 변환하고 pad_idx를 제외시킨다.
    def decode(self, tokens):
        return ' '.join([self.idx2word[token] for token in tokens if token != self.pad_idx])

def generate_query(model, sentence, dataset, max_len=50):
    '''
    model: 학습된 모델
    sentence: 변환하거나 번역하고자 하는 입력 문장
    dataset: CustomInferenceDataset 클래스의 인스턴스
    max_len: 생성할 출력 문장의 최대 길이
    '''
    model.eval() # 평가 모드
    with torch.no_grad(): # Gradient 계산을 비활성화
        tokens = dataset.tokenize_and_encode(sentence) 
        src_tensor = torch.LongTensor(tokens).unsqueeze(1).to(device) # 입력 문장을 텐서로 변환하고 텐서를 모델에 맞게 변형
        src_len = torch.LongTensor([len(tokens)])
        
        trg_indexes = [dataset.word2idx['<sos>']]  # Assuming you have <sos> token for start
        for _ in range(max_len): # 모델은 현재까지의 출력 토큰을 사용하여 다음 토큰을 예측한다.
            trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(1).to(device)
            output = model(src_tensor, trg_tensor)
            pred_token = output.argmax(2)[-1, :].item() # argmax(2)를 사용하여 가장 확률이 높은 토큰의 인덱스를 얻는다.
            trg_indexes.append(pred_token)
            if pred_token == dataset.word2idx['<eos>']:  # Assuming you have <eos> token for end
                break
        trg_tokens = trg_indexes[1:]
        
        # 출력 문장 끝에 <eos> 토큰이 있다면 제거
        if trg_tokens[-1] == dataset.word2idx['<eos>']:
            trg_tokens = trg_tokens[:-1]
        
        query = dataset.decode(trg_tokens)
        return query


def correct_sql_by_gpt3(user_input, input_query):
    # GPT-3로부터 SQL 쿼리 교정 요청
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "너는 이제부터 Seq2seq LSTM에서 나온 SQL Query를 교정해주는 역할이야. 답변은 다 생략하고 무조건 수정된 SQL Query만 출력해줘."},
            {"role": "user", "content": "삼성전자의 최신 뉴스를 요약해줘."},
            {"role": "assistant", "content": "SELECT title, link, description, pubdate FROM 삼성전자 ORDER BY pubdate DESC LIMIT 1"},
            {"role": "user", "content": user_input},
            {"role": "user", "content": f"사용자의 입력인 '{user_input}'에 기반해서 Seq2seq LSTM Model이 만들어낸 SQL Query인 '{input_query}'을 올바른 형태로 수정해줘."}
            
        ],
    )
    corrected_query = response.choices[0]['message']['content'].strip()
    return corrected_query

# Load vocabulary from training dataset
train_dataset = CustomDataset('train.csv')
vocab = train_dataset.word2idx
inference_dataset = CustomInferenceDataset(vocab)

# Load the trained model
INPUT_DIM = len(vocab) + 1
OUTPUT_DIM = INPUT_DIM
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5


enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
attn = Attention(HID_DIM)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT, attn)
model = Seq2Seq(enc, dec, device).to(device)

model.load_state_dict(torch.load('seq2seq-model.pt'))

# Get user input and generate SQL query
for _ in range(10):
    user_input = input("Enter a sentence: ")
    sql_query = generate_query(model, user_input, inference_dataset)
 #   print(f"Generated SQL Query: {sql_query}")
    corrected_sql_query = correct_sql_by_gpt3(user_input, sql_query)
    print(f"final SQL Query: {corrected_sql_query}")