# Test the models with LG_chem stock
# If the prediction is success, Expand the number of stock
import math 
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from sklearn.preprocessing import MinMaxScaler


base = os.path.abspath(__file__)
base = base.split('/')

def save_stock_plot(rawdata, stock_name="LG_chme"):
    global base
    try:
        plt.figure(figsize=(20,5))
        plt.plot(range(len(rawdata)), rawdata['Close'])
        path = "/".join(base[:-2]+["models"])
        file_name = f"/{stock_name}.jpg"
        path += file_name
        plt.savefig(path)
        print("Save Success!!")
    except Exception as e:
        print(f"Save Stock plot Failed!!: {e}")


class windowDataset(Dataset):
    def __init__(self, y, input_window=80, output_window=20, stride=5):
        #총 데이터의 개수
        L = y.shape[0]
        #stride씩 움직일 때 생기는 총 sample의 개수
        num_samples = (L - input_window - output_window) // stride + 1

        #input과 output
        X = np.zeros([input_window, num_samples])
        Y = np.zeros([output_window, num_samples])

        for i in np.arange(num_samples):
            start_x = stride*i
            end_x = start_x + input_window
            X[:,i] = y[start_x:end_x]

            start_y = stride*i + input_window
            end_y = start_y + output_window
            Y[:,i] = y[start_y:end_y]

        X = X.reshape(X.shape[0], X.shape[1], 1).transpose((1,0,2))
        Y = Y.reshape(Y.shape[0], Y.shape[1], 1).transpose((1,0,2))
        self.x = X
        self.y = Y
        
        self.len = len(X)
    def __getitem__(self, i):
        return self.x[i], self.y[i]
    def __len__(self):
        return self.len
    

class TFModel(nn.Module):
    def __init__(self,iw, ow, d_model, nhead, nlayers, dropout=0.5):
        super(TFModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayers) 
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        self.encoder = nn.Sequential(
            nn.Linear(1, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, d_model)
        )
        
        self.linear =  nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, 1)
        )

        self.linear2 = nn.Sequential(
            nn.Linear(iw, (iw+ow)//2),
            nn.ReLU(),
            nn.Linear((iw+ow)//2, ow)
        ) 

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, srcmask):
        src = self.encoder(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src.transpose(0,1), srcmask).transpose(0,1)
        output = self.linear(output)[:,:,0]
        output = self.linear2(output)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def gen_attention_mask(x):
    mask = torch.eq(x, 0)
    return mask


def evaluate(data_train, device, model):
    # 마지막 30*2일 입력으로 넣어서 그 이후 30일 예측 결과 얻음.
    input = torch.tensor(data_train[-30*2:]).reshape(1,-1,1).to(device).float().to(device)
    model.eval()
    
    src_mask = model.generate_square_subsequent_mask(input.shape[1]).to(device)
    predictions = model(input, src_mask)
    return predictions.detach().cpu().numpy()


def predict(stock, period):
    global base
    print(f"Notice: Since it is in the initial stage of the service, \
          we predict only the stock price of LG Chem, not the stock price \
          of the designated company.\n\n")
    # 이 코드대신 지수형이 spl로 얻어온 data가 rawdata가 되어야 함.
    print(f"Loading Stock Data ...")
    path = "/".join(base[:-3]+["data","lg_chem_closing_prices.csv"])
    rawdata = pd.read_csv(path)

    print(f"Saving Stock data as .png ...")
    save_stock_plot(rawdata, stock)

    print(f"Preprocessing Data with MinMaxScaling ...")
    min_max_scaler = MinMaxScaler()
    rawdata["Close"] = min_max_scaler.fit_transform(rawdata["Close"].to_numpy().reshape(-1,1))

    print(f"Spliting Data ...")
    train = rawdata[:-60]
    data_train = train["Close"].to_numpy()
    test = rawdata[-60:]
    data_test = test["Close"].to_numpy()

    print(f"Preparing Dataset ...")
    iw = 30*2
    ow = 30
    train_dataset = windowDataset(data_train, input_window=iw, output_window=ow, stride=1)
    train_loader = DataLoader(train_dataset, batch_size=64)
    #test_dataset = windowDataset(data_test, input_window=iw, output_window=ow, stride=1)
    #test_loader = DataLoader(test_dataset)

    print(f"Model Constructing ...")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    lr = 1e-4
    model = TFModel(30*2, 30, 512, 8, 4, 0.1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("Trainig ...")
    epoch = 10
    model.train()
    progress = tqdm(train_loader, total=len(train_loader), leave=True)
    for i in tqdm(range(epoch)):
        batchloss = 0.0
        for (inputs, outputs) in train_loader:
            optimizer.zero_grad()
            src_mask = model.generate_square_subsequent_mask(inputs.shape[1]).to(device)
            result = model(inputs.float().to(device),  src_mask)
            loss = criterion(result, outputs[:,:,0].float().to(device))
            loss.backward()
            optimizer.step()
            batchloss += loss
        progress.set_description("loss: {:0.6f}".format(batchloss.cpu().item() / len(train_loader)))

    print("Predicting ...")
    # 총 1719일의 data있음
    
    result = evaluate(data_test, device, model)
    result = min_max_scaler.inverse_transform(result)[0]
    real = rawdata["Close"].to_numpy()
    real = min_max_scaler.inverse_transform(real.reshape(-1,1))[:,0]
    # print(len(real))

    plt.figure(figsize=(20,5))
    plt.plot(range(1419,1719),real[1419:], label="real")
    plt.plot(range(1719-30,1719),result, label="predict")
    plt.legend()
    path = "/".join(base[:-2]+["models","prediction.jpg"])
    plt.savefig(path)
    print(f"Complete!!")
