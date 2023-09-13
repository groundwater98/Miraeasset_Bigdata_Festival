# Test the models with LG_chem stock
# If the prediction is success, Expand the number of stock
import math 
import os
import pdb
from datetime import datetime

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
    def __init__(self, y, input_window=80, output_window=20, stride=5, n_attr=1):
        #총 데이터의 개수

        L = y.shape[0]
        #stride씩 움직일 때 생기는 총 sample의 개수
        num_samples = (L - input_window - output_window) // stride + 1

        if n_attr == 1:
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
            X = X.reshape(X.shape[0], X.shape[1], n_attr)
            Y = Y.reshape(Y.shape[0], Y.shape[1], n_attr)
            X = X.transpose((1,0,2))
            Y = Y.transpose((1,0,2))
            self.x = X
            self.y = Y
        else:
            #input과 output
            X = np.zeros([input_window, n_attr, num_samples])
            Y = np.zeros([output_window, n_attr, num_samples])

            for i in np.arange(num_samples):
                start_x = stride*i
                end_x = start_x + input_window
                X[:,:,i] = y[start_x:end_x]

                start_y = stride*i + input_window
                end_y = start_y + output_window
                Y[:,:,i] = y[start_y:end_y]
            X = X.reshape(X.shape[2], X.shape[0], X.shape[1])
            Y = Y.reshape(Y.shape[2], Y.shape[0], Y.shape[1])
            self.x = X
            self.y = Y
        self.len = len(X)
    def __getitem__(self, i):
        return self.x[i], self.y[i]
        #return self.x[i], self.y[i, :-1], self.y[i,1:]
    def __len__(self):
        return self.len
    

class TFModel(nn.Module):
    def __init__(self,iw, ow, d_model, nhead, nlayers, dropout=0.5, n_attr=1):
        super(TFModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayers) 
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        self.encoder = nn.Sequential(
            nn.Linear(n_attr, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, d_model)
        )
        
        self.linear =  nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, n_attr)
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


class TFModel2(nn.Module):
    def __init__(self,d_model, nhead, nhid, nlayers, dropout=0.5, n_attr=7):
        super(TFModel2, self).__init__()
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, dim_feedforward=nhid, num_encoder_layers=nlayers, num_decoder_layers=nlayers,dropout=dropout)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.pos_encoder_d = PositionalEncoding(d_model, dropout)
        self.linear = nn.Linear(d_model, n_attr)
        self.encoder = nn.Linear(n_attr, d_model)
        self.encoder_d = nn.Linear(n_attr, d_model)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt, srcmask, tgtmask):
        src = self.encoder(src)
        src = self.pos_encoder(src)

        tgt = self.encoder_d(tgt)
        tgt = self.pos_encoder_d(tgt)
        output = self.transformer(src.transpose(0,1), tgt.transpose(0,1), srcmask, tgtmask)
        output = self.linear(output)
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


def evaluate(data_train, device, model, iw, n_attr, length):
    # 마지막 30*2일 입력으로 넣어서 그 이후 30일 예측 결과 얻음.
    input = torch.tensor(data_train[-iw:]).reshape(1,-1,n_attr).to(device).float().to(device)
    model.eval()
    
    src_mask = model.generate_square_subsequent_mask(input.shape[1]).to(device)
    predictions = model(input, src_mask)
    return predictions.detach().cpu().numpy()
    """
    input = torch.tensor(data_train[-iw:]).reshape(1,-1,n_attr).to(device).float().to(device)
    output = torch.tensor(data_train[-1].reshape(1,-1,n_attr)).float().to(device)
    model.eval()
    for i in range(length):
        src_mask = model.generate_square_subsequent_mask(input.shape[1]).to(device)
        tgt_mask = model.generate_square_subsequent_mask(output.shape[1]).to(device)

        predictions = model(input, output, src_mask, tgt_mask).transpose(0,1)
        predictions = predictions[:, -1:, :]
        output = torch.cat([output, predictions.to(device)], axis=1)
    return torch.squeeze(output, axis=0).detach().cpu().numpy()[1:]
    """

def predict(stock, period):
    global base
    print(f"Notice: Since it is in the initial stage of the service, \
          we predict only the stock price of LG Chem, not the stock price \
          of the designated company.\n\n")
    # 이 코드대신 지수형이 spl로 얻어온 data가 rawdata가 되어야 함.
    # 추가적인 정보 없는건 1729일
    
    print(f"Loading Stock Data ...")
    n_attr = 1
    path = "/".join(base[:-3]+["data","lg_chem_closing_prices.csv"])
    model_path = "/".join(base[:-2]+["Prediction", f"{stock}_{datetime.now().date()}.pth"])
    rawdata = pd.read_csv(path)
    
    print(f"Saving Stock data as .png ...")
    save_stock_plot(rawdata, stock)
    #pdb.set_trace()
    print(f"Preprocessing Data with MinMaxScaling ...")
    min_max_scaler = MinMaxScaler()
    rawdata["Close"] = min_max_scaler.fit_transform(rawdata["Close"].to_numpy().reshape(-1,n_attr))

    print(f"Spliting Data ...")
    iw = 30*7
    ow = 10
   
    train = rawdata[:-iw]
    data_train = train["Close"].to_numpy()
    test = rawdata[-iw:]
    data_test = test["Close"].to_numpy()

    print(f"Preparing Dataset ...")
    train_dataset = windowDataset(data_train, input_window=iw, output_window=ow, stride=1, n_attr=n_attr)
    train_loader = DataLoader(train_dataset, batch_size=64)
    #test_dataset = windowDataset(data_test, input_window=iw, output_window=ow, stride=1, n_attr=n_attr)
    #test_loader = DataLoader(test_dataset)
    """
    # 성능 올리기위해 종가말고 다른 것도 같이 넣음.
    # 총 1720일의 data있음
    print(f"Loading Stock Data ...")
    n_attr = 7
    path = "/".join(base[:-3]+["data","lg_chem_prices.csv"])
    rawdata = pd.read_csv(path)

    #print(f"Saving Stock data as .png ...")
    #save_stock_plot(rawdata, stock)
    print(f"Preprocessing Data with MinMaxScaling ...")
    min_max_scaler = MinMaxScaler()
    rawdata.loc[:,rawdata.columns] = min_max_scaler.fit_transform(rawdata.to_numpy())

    print(f"Spliting Data ...")
    iw = 60
    ow = 5
    #pdb.set_trace()
    train = rawdata[:-(iw)]
    data_train = train.to_numpy()
    test = rawdata[-(iw):]
    data_test = test.to_numpy()

    print(f"Preparing Dataset ...")
    train_dataset = windowDataset(data_train, input_window=iw, output_window=ow, stride=1, n_attr=n_attr)
    train_loader = DataLoader(train_dataset, batch_size=64)
    #test_dataset = windowDataset(data_test, input_window=iw, output_window=ow, stride=1, n_attr=n_attr)
    #test_loader = DataLoader(test_dataset)
    """
    print(f"Model Constructing ...")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    lr = 1e-4
    #model = TFModel2(256, 8, 256, 2, 0.1, n_attr).to(device)
    model = TFModel(iw, ow, 512, 8, 4, 0.4, n_attr).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    if not os.path.exists(model_path):
        print("Trainig ...")
        epoch = 10
        
        model.train()
        for i in range(epoch):
            batchloss = 0.0
            for (inputs, outputs) in tqdm(train_loader):
                optimizer.zero_grad()
                src_mask = model.generate_square_subsequent_mask(inputs.shape[1]).to(device)
                result = model(inputs.float().to(device),  src_mask)
                loss = criterion(result, outputs[:,:,0].float().to(device))
                loss.backward()
                optimizer.step()
                batchloss += loss
            print(f"{i+1}th epoch MSEloss:" + "{:0.6f}".format(batchloss.cpu().item() / len(train_loader)))
        """
        model.train()
        progress = tqdm(range(epoch))
        for i in progress:
            batchloss = 0.0
            
            for (inputs, dec_inputs, outputs) in train_loader:
                optimizer.zero_grad()
                src_mask = model.generate_square_subsequent_mask(inputs.shape[1]).to(device)
                tgt_mask = model.generate_square_subsequent_mask(dec_inputs.shape[1]).to(device)

                result = model(inputs.float().to(device), dec_inputs.float().to(device), src_mask, tgt_mask)
                loss = criterion(result.permute(1,0,2), outputs.float().to(device))
                
                loss.backward()
                optimizer.step()
                batchloss += loss
            progress.set_description("{:0.5f}".format(batchloss.cpu().item() / len(train_loader)))

        """
    torch.save(model, model_path)
    print("Predicting ...")
    result = evaluate(data_test, device, model, iw, n_attr, ow)
    

   
    result = min_max_scaler.inverse_transform(result)[0]
    real = rawdata["Close"].to_numpy()
    real = min_max_scaler.inverse_transform(real.reshape(-1,1))[:,0]
    
    #pdb.set_trace()
    """
    tmp = np.zeros((10,7))
    tmp[:,:] = result.reshape(10,-1)
    result = tmp
    result = min_max_scaler.inverse_transform(result).reshape(-1,10)[3]
    real = rawdata.to_numpy()
    real = min_max_scaler.inverse_transform(real)[:,3]
     """
    plt.figure(figsize=(20,5))
    #plt.plot(range(1419,1719),real[1420:], label="real")
    plt.plot(range(1419,1719),real[1418:],label="real")
    plt.plot(range(1719-ow,1719),result, label="predict")
    plt.legend()
    path = "/".join(base[:-2]+["models","prediction2.jpg"])
    plt.savefig(path)
    print(f"Complete!!")

    # 예측된 가격의 평균과, 직전의 값을 비교했을 때, 평균이 크면 사라, 작으면 사지 마라.
    mean_pred = np.mean(result)
    if mean_pred >= real[-1]:
        answer = f"""You should buy the stock you want to know the price, because we predict the price will rise. 
        Maybe it will be {mean_pred}won.""" 
    else:
        answer = f"""You shouldn't buy the stock you want to know the price, because we predict the price will go down. 
        Maybe it will be {mean_pred}won."""

    return answer 

if __name__=="__main__":
    print(predict("", ""))