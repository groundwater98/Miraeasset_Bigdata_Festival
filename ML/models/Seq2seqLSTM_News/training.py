import os
import torch
import math
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from Seq2seqLSTM import Encoder, Decoder, Attention, Seq2Seq


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CustomDataset(Dataset):
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)

        # Create a vocabulary based on the 'source' and 'target'
        self.vocab = set()  # 중복된 단어를 허용하지 않는 set
        for _, row in self.df.iterrows():
            self.vocab.update(row["source"].split())
            self.vocab.update(row["target"].split())

        # 단어장에 추가
        self.vocab.add("<sos>")
        self.vocab.add("<eos>")

        # Create a mapping from word to index
        self.word2idx = {
            word: idx for idx, word in enumerate(self.vocab)
        }  # (index, word) => {word: idx}
        self.idx2word = {
            idx: word for word, idx in self.word2idx.items()
        }  # (word, idx) => {idx, word}
        self.pad_idx = 0

    def __len__(self):
        return len(self.df)

    def tokenize_and_encode(self, sentence):
        tokens = sentence.split()  # 문장을 공백 기준으로 tokenize
        return [self.word2idx[token] for token in tokens]  # 각 토큰을 index로 변환

    def __getitem__(self, index):
        src = self.tokenize_and_encode(
            self.df["source"].iloc[index]
        )  # source column의 해당 index 값을 가져온다.
        trg = self.tokenize_and_encode(
            self.df["target"].iloc[index]
        )  # target column의 해당 index 값을 가져온다.
        # 타겟 시퀀스에 <sos>, <eos> 추가
        trg = [self.word2idx["<sos>"]] + trg + [self.word2idx["<eos>"]]

        return torch.tensor(src), torch.tensor(trg)


def collate_fn(batch):
    srcs, trgs = zip(*batch)
    srcs = pad_sequence(
        srcs, padding_value=0, batch_first=False
    )  # 여러 소스 시퀀스를 동일한 길이로 padding
    trgs = pad_sequence(
        trgs, padding_value=0, batch_first=False
    )  # 여러 타겟 시퀀스를 동일한 길이로 padding
    return srcs, trgs


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for i, (src, trg) in enumerate(iterator):
        src, trg = src.to(device), trg.to(device)  # src, trg => GPU
        optimizer.zero_grad()  # gradient 초기화
        output = model(src, trg)

        output_dim = output.shape[-1]  # 출력 차원 가져오기
        output = output[1:].view(-1, output_dim)  # 첫 번째 토큰인 <sos> 제외
        trg = trg[1:].view(-1)
        loss = criterion(output, trg)  # 예측된 output과 실제 trg를 사용하여 loss 계산
        loss.backward()  # backpropagation
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), clip
        )  # gradient 폭발을 방지하기 위해 gradient 크기를 제한
        optimizer.step()  # 계산된 gradient를 바탕으로 model parameter update
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)  # average loss 반환


def evaluate(model, iterator, criterion):
    model.eval()  # 평가 모드
    epoch_loss = 0

    with torch.no_grad():
        for i, (src, trg) in enumerate(iterator):
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg, 0)  # turn off teacher forcing
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def test(model, iterator, criterion):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, (src, trg) in enumerate(iterator):
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg, 0)  # turn off teacher forcing
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
            
    return epoch_loss / len(iterator)

# Load and process the data
train_dataset = CustomDataset("train.csv")
val_dataset = CustomDataset("val.csv")
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
test_dataset = CustomDataset("test.csv")
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

# Hyperparameters
INPUT_DIM = len(train_dataset.word2idx) + 1  # +1 for padding index
OUTPUT_DIM = INPUT_DIM
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

# Create the model instances
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
attn = Attention(HID_DIM)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT, attn)
model = Seq2Seq(enc, dec, device).to(device)

#optimizer = optim.SGD(model.parameters(),lr=0.01)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.pad_idx)

N_EPOCHS = 10
CLIP = 1


# PPL->Perplexity: 낮을 수록 좋다.
for epoch in range(N_EPOCHS):
    train_loss = train(model, train_loader, optimizer, criterion, CLIP)

    print(f"Epoch: {epoch + 1:02}")
    print(f"\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}")

torch.save(model.state_dict(), "seq2seq-model.pt")
