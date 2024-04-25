import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from copy import deepcopy
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import seaborn as sns
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import time

start = time.time()


def filtering(da, lamb):
    p = da.values
    y = []
    a = len(p)
    for i in range(a):
        k = p[i]
        if (k >= 0.01):
            y.append(1)
        else:
            y.append(0)
    n = len(y)
    y = p
    t = np.linspace(0, a, n)
    tau = cp.Variable(n)
    lambda_param = lamb  # 정규화 파라미터

    # 두 번째 차분 계산을 위한 행렬 D
    D = np.zeros((n - 2, n))
    for i in range(n - 2):
        D[i, i] = 1
        D[i, i + 1] = -2
        D[i, i + 2] = 1

    # 최적화 문제 정의 및 해결
    objective = cp.Minimize(0.5 * cp.sum_squares(y - tau) + lambda_param * cp.norm(D @ tau, 1))
    prob = cp.Problem(objective)
    prob.solve()
    re = tau.value
    return re


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the dataset
data = pd.read_csv("dataset.csv")


# Function to prepare the dataset
def prepare_dataset(data, lag_size):
    data = data.copy()
    data.index = pd.to_datetime(data['Date'])
    data_clean = data.dropna()

    return data_clean


# Function to compute the directional loss
def directional_loss(y_true, y_pred, alpha=0):
    difference = y_pred - y_true
    direction_penalty = torch.where((y_true[1:] > y_true[:-1]) & (y_pred[1:] < y_pred[:-1]) |
                                    (y_true[1:] < y_true[:-1]) & (y_pred[1:] > y_pred[:-1]),
                                    torch.abs(difference[1:]), torch.zeros_like(difference[1:]))
    mse = torch.mean(torch.square(difference))
    directional_error = torch.mean(direction_penalty)
    return mse + alpha * directional_error


# CNN-LSTM model class
class CNNLSTMModelWithSigmoid(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_filters, kernel_size):
        super(CNNLSTMModelWithSigmoid, self).__init__()
        self.conv1d = torch.nn.Conv1d(input_dim, num_filters, kernel_size, padding='same')
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool1d(2, stride=2)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(num_filters, hidden_dim, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = self.relu(x)
        if x.size(2) > 1:
            x = self.maxpool(x)
        x = x.permute(0, 2, 1)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        x, _ = self.lstm(x, (h0, c0))
        x = self.fc(x[:, -1, :])
        x = self.sigmoid(x)
        return x


def train_and_predict_binary_lstm(data_for_train, window_size, mode, lamb, line, label_percent):
    batch_size = 16
    epochs = 100
    input_dim = data_for_train.shape[1]
    all_predictions = []
    all_actuals = []
    date_index = []
    test = []
    use_window = 0
    # for start in range(len(data_for_train) - window_size, len(data_for_train) - window_size+1, 1):
    for start in range(700, len(data_for_train) - window_size+1, 1):
        end = start + window_size
        window_data = data_for_train.iloc[start:end]
        date_index = date_index + [window_data.iloc[-1:].index[0]]
        #print(window_data.iloc[-1:].index[0])
        train_data = window_data.iloc[:-1]
        test_data = window_data.iloc[-1:]
        k = data_for_train.iloc[0:end]
        h = filtering(k['return'], lamb)
        p = []
        for i in range(window_size - 1):
            kk = start + i + 1
            a = h[kk]
            pp = 0
            if (mode == 1):
                if (a >= line):
                    pp = 1
                elif (a < line):
                    pp = 0
                #print('pos')
            elif (mode == 0):
                if (a <= -line):
                    pp = 1
                elif (a > -line):
                    pp = 0
                #print('neg')
            p.append(pp)
        kkkk = np.array(p)
        if np.sum(kkkk)/window_size>label_percent:
            X_train = torch.tensor(train_data.values).float().to(device)
            test = test + [np.sum(kkkk)]
            use_window = use_window + 1
            print(window_data.iloc[-1:].index[0])
            y_train = torch.tensor(kkkk).float().view(-1, 1).to(device)
            X_test = torch.tensor(test_data.values).float().to(device)
            # print("X_train shape:", X_train.shape)
            # print("y_train shape:", y_train.shape)
            # print(kkkk)

            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

            model = CNNLSTMModelWithSigmoid(input_dim=input_dim, hidden_dim=100, num_layers=2, num_filters=64,
                                            kernel_size=3).to(device)
            optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01)

            for epoch in range(epochs):
                model.train()
                train_loss = 0
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(device), target.to(device)
                    optimizer.zero_grad()
                    output = model(data.unsqueeze(1))
                    loss = directional_loss(output, target)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                train_loss = train_loss / len(train_loader)
                # print(f'Epoch: {epoch + 1} | Train Loss: {train_loss:.4f}')
            model.eval()
            with torch.no_grad():
                predicted_label = model(X_test.unsqueeze(1)).view(-1).cpu().numpy()
                # actual_label = test_data[label_col].values
                all_predictions.extend(predicted_label)
                # all_actuals.extend(actual_label)

        else:
            all_predictions.extend([0])
            test = test + [np.sum(kkkk)]

    return np.array(all_actuals), np.array(all_predictions), np.array(date_index), test, use_window


# Variables for the evaluation
'''
lambda_lists = [0.05,0.07,0.1]
up_window_sizes = [30, 35, 40, 45, 50, 60]
down_window_sizes = [30, 35, 40, 45, 50, 60]
lines = [0.01,0.007,0.005]
label_percents = [0.2]
'''

up_lambda_lists = [0.07, 0.1]
down_lambda_list = [0.05, 0.1]
up_window_sizes = [30, 35, 40, 45]
down_window_sizes = [50, 55, 60, 65]
lines = [0.01, 0.007, 0.005]
label_percents = [0.1]


results_dir = 'results3/'
os.makedirs(results_dir, exist_ok=True)

from collections import Counter

for lamb_num in range(len(up_lambda_lists)):

    data_up = prepare_dataset(data.copy(), 3)
    data_down = data_up.copy()

    cols_up = [col for col in data_up.columns if col not in ['Date']]
    cols_down = [col for col in data_down.columns if col not in ['Date']]
    data_for_train_up = data_up[cols_up].astype(float)
    data_for_train_down = data_down[cols_down].astype(float)

    for label_percent in label_percents:
        for window_size_num in range(len(up_window_sizes)):
            for line in lines:
                # Train and predict with the upward trend model
                actual_up, predictions_up, up_date_index, up_test, up_use_window = train_and_predict_binary_lstm(data_for_train_up, up_window_sizes[window_size_num], 1, up_lambda_lists[lamb_num], line, label_percent)

                # Train and predict with the downward trend model
                actual_down, predictions_down, down_date_index, down_test, down_use_window = train_and_predict_binary_lstm(data_for_train_down, down_window_sizes[window_size_num], 0, down_lambda_list[lamb_num], line, label_percent)
                '''counts = Counter(up_test)
                for element, count in counts.items():
                    print(f'up_{line}_{element}:{count}')
                counts = Counter(down_test)
                for element, count in counts.items():
                    print(f'down_{line}_{element}:{count}')'''
                # Save results to CSV files
                results_up = pd.DataFrame({'Prediction': predictions_up}, index = up_date_index)
                results_down = pd.DataFrame({'Prediction': predictions_down}, index = down_date_index)
                results_up.to_csv(f'{results_dir}results_up_lambda{up_lambda_lists[lamb_num]}_window{up_window_sizes[window_size_num]}_line{line}_label{label_percent}.csv', index=True)
                results_down.to_csv(f'{results_dir}results_down_lambda{down_lambda_list[lamb_num]}_window{down_window_sizes[window_size_num]}_line{line}_label{label_percent}.csv', index=True)
                print(up_use_window)
                print(down_use_window)

# Notify that the process is completed
'Process completed. Files are saved in the specified directory.'


print("time :", time.time() - start)