# -*- coding: utf-8 -*-
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data_processing import LoadData
from model import *

"""
     使用图卷积神经网络实现基于交通流量数据的预测
     Dataset description：
     PeMS04 ，加利福尼亚高速数据，"data.npz"，原始数据shape=(10195,307,3)——间隔5分钟预测1小时(307,3,36)->(307,3,12)
     其中，"3"代表交通流量3种特征(flow，speed，occupancy)。

"""


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Loading Dataset
    train_data = LoadData(data_path=["PeMS_04/PeMS04.csv", "PeMS_04/PeMS04.npz"], num_nodes=307, divide_days=[45, 14],
                          time_interval=5, history_length=6,
                          train_mode="train")

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=32)

    test_data = LoadData(data_path=["PeMS_04/PeMS04.csv", "PeMS_04/PeMS04.npz"], num_nodes=307, divide_days=[45, 14],
                         time_interval=5, history_length=6,
                         train_mode="test")

    test_loader = DataLoader(test_data, batch_size=64, shuffle=True, num_workers=32)




    # Loading Model

    # 可以选择底层实现的GCN或ChebNet model.
    # 关于模型的解释：可以参考本人知乎链接：
    # ChebNet：https://zhuanlan.zhihu.com/p/138420723
    # GCN：https://zhuanlan.zhihu.com/p/138686535

    #model = GCN(in_c=6 , hid_c=6 ,out_c=1)
    model = ChebNet(in_c=6, hid_c=32, out_c=1, K=2)      # 2阶切比雪夫模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(params=model.parameters())

    # Train model
    Epoch = 8

    model.train()
    for epoch in range(Epoch):
        epoch_loss = 0.0
        start_time = time.time()
        for data in train_loader:  # ["graph": [B, N, N] , "flow_x": [B, N, H, D], "flow_y": [B, N, 1, D]]
            model.zero_grad()
            predict_value = model(data, device).to(torch.device("cpu"))  # [0, 1] -> recover
            loss = criterion(predict_value, data["flow_y"])
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        end_time = time.time()

        print("Epoch: {:04d}, Loss: {:02.4f}, Time: {:02.2f} mins".format(epoch, 1000 * epoch_loss / len(train_data),
                                                                          (end_time-start_time)/60))

    # Test Model
    model.eval()
    with torch.no_grad():

        total_loss = 0.0
        for data in test_loader:
            predict_value = model(data, device).to(torch.device("cpu"))  # [B, N, 1, D]
            loss = criterion(predict_value, data["flow_y"])
            total_loss += loss.item()

        print("Test Loss: {:02.4f}".format(1000 * total_loss / len(test_data)))


if __name__ == '__main__':
    main()
