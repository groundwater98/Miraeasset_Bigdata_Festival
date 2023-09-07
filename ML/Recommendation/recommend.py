import os 
import pdb

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

base_path = os.path.abspath(__file__).split('/')

# k-Nearest Neighbors class
class KNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y, code):
        self.X_train = X
        self.y_train = y
        self.code_name = code

    def predict(self, X):
        dists = torch.cdist(X, self.X_train)  # Compute Euclidean distances
        _, indices = dists.topk(self.k, largest=False, dim=1)  # Get k smallest distances' indices
        k_nearest_labels = self.y_train[indices]  # Get labels of k nearest neighbors
        code = []
        for l1 in k_nearest_labels:
            #print("l1")
            tmp = []
            for l2 in l1:
                #print("l2")
                for l3 in l2:
                    #print("l3")
                    #print(l3)
                    tmp.append(f"{str(int(l3.item())):>06}")
            code.append(tmp)
        print(len(code))
        result = [self.code_name[self.code_name["종목코드"].isin(elem)]["종목명"] for elem in code]
        #k_nearest_labels = self.code_name[self.code_name["종목코드"]==self.y_train[indices]]["종목명"]
        #predicted_labels = torch.mode(k_nearest_labels, dim=1).values  # Predict most common label
        return result, k_nearest_labels
    

def recommend(user_inform):
    global base_path
    model_path = "/".join(base_path[:-2]+["Recommendation", "knn_model.pth"])
    path = "/".join(base_path[:-3]+["data","cs_mkt_dataset","cs_data.csv"])

    # Create k-NN classifier
    print("Constructing k-NN classifier ...")
    knn = KNN(k=5)
      
    if not os.path.exists(model_path):
        print("Loading CustomerData  ...")
        data = pd.read_csv(path)

        print("Processing Nan value  ...")
        data = data.fillna(method='ffill')
        data = data.fillna(method="bfill")
        data.isnull().sum()

        # column data type 변경
        # 우선은 M1만으로 추천시스템 구축
        print("Constructing Data  ...")
        #print(f'{data[["CASH_AST_M1","DMST_AST_EVAL_M1","DMST_AST_PCHS_M1"]].values.dtype}')
        #print(f'{data[["DMST_AST1_ITM_M1"]].values.dtype}')
        data.loc[data["DMST_AST1_ITM_M1"]=="*"] = 0
        data[["DMST_AST1_ITM_M1"]] = data[["DMST_AST1_ITM_M1"]].astype(dtype='float64')
        #print(f'{data[["DMST_AST1_ITM_M1"]].values.dtype}')
        dm_trainX = torch.tensor(data[["CASH_AST_M1","DMST_AST_EVAL_M1","DMST_AST_PCHS_M1"]].values)
        dm_trainY = torch.tensor(data[["DMST_AST1_ITM_M1"]].values)

        print("Loading Stock Code Information ...")
        path = "/".join(base_path[:-3]+["data","cs_mkt_dataset","code.csv"])
        code_inform = pd.read_csv(path, encoding='cp949')
        
        knn.fit(dm_trainX.float(), dm_trainY.float(), code_inform)

        # Save model to a .pth file
        torch.save(knn, 'knn_model.pth')

    print("Loading Customer data who want to get recommendation ...")
    # X_test에 고객 정보가 있어야 함.
    # 지금 버전에서는 M1분기의 현금보유액, 매입주식평가액, 주식 매입액: 총 3개의 data필요.
    X_test = torch.tensor([[10000000.0,23000000.0,49000000.0]]).float()

    # Load model from the .pth file
    loaded_knn = torch.load('knn_model.pth')

    # Predict labels for test data using loaded model
    code_name, predicted_labels = loaded_knn.predict(X_test)
    #print("Predicted labels using loaded model:", predicted_labels)
    #print("Predicted code_name using loaded model:\n", code_name)
    #pdb.set_trace()
    code_name = code_name[0].values
    answer = f"The stocks that are right for you are as follows: Stocks: {', '.join(code_name)}"
    return answer