import torch 
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import pandas as pd


class CustomDataset(Dataset):
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)
        
        # Create a vocabulary based on the 'source' and 'target'
        self.vocab = set() # 중복된 단어를 허용하지 않는 set
        for _, row in self.df.iterrows():
            self.vocab.update(row['source'].split())
            self.vocab.update(row['target'].split())
        
        # 단어장에 추가
        self.vocab.add('<sos>')
        self.vocab.add('<eos>')

        # Create a mapping from word to index
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)} # (index, word) => {word: idx}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()} # (word, idx) => {idx, word}
        self.pad_idx = 0

    def __len__(self):
        return len(self.df)

    def tokenize_and_encode(self, sentence):
        tokens = sentence.split() # 문장을 공백 기준으로 tokenize
        return [self.word2idx[token] for token in tokens] # 각 토큰을 index로 변환

    def __getitem__(self, index):
        src = self.tokenize_and_encode(self.df['source'].iloc[index]) # source column의 해당 index 값을 가져온다.
        trg = self.tokenize_and_encode(self.df['target'].iloc[index]) # target column의 해당 index 값을 가져온다.
        # 타겟 시퀀스에 <sos>, <eos> 추가
        trg = [self.word2idx['<sos>']] + trg + [self.word2idx['<eos>']]

        return torch.tensor(src), torch.tensor(trg)


def collate_fn(batch):
    srcs, trgs = zip(*batch)
    srcs = pad_sequence(srcs, padding_value=0, batch_first=False) # 여러 소스 시퀀스를 동일한 길이로 padding
    trgs = pad_sequence(trgs, padding_value=0, batch_first=False) # 여러 타겟 시퀀스를 동일한 길이로 padding
    return srcs, trgs


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