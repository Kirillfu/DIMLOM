import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score, mean_squared_log_error
import yfinance as yf

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, device):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.device = device

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        out, h_n = self.gru(x, h0)

        out = self.fc(out[:, -1, :])

        return out

def before_train(code):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Задаем гиперпараметры модели
    input_size = 1
    hidden_size = 32
    num_layers = 2
    output_size = 1
    seq_length = 30 # длина входной последовательности (количество дней для предсказания)
    code_data = load_stock_dataset(code)

    dl_model = GRU(input_size, hidden_size, num_layers, output_size, device).to(device)
    test_y, test_predict, test_loss = preprocess_and_train(code_data, dl_model, seq_length, device, 0.03)
    return test_y, test_predict, test_loss

def load_stock_dataset(code):
    code_name = code + "-USD"
    code_ticker = yf.Ticker(code_name)
    code_data = code_ticker.history(period="max", interval="1d")
    return code_data

def preprocess_and_train(df, model, seq_length, device, res=0.03):
    close_prices = df['Close'].values.reshape(-1, 1)

    # Нормализуем данные
    scaler = MinMaxScaler()
    close_prices = scaler.fit_transform(close_prices)

    df = df.drop(columns=['Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits'])

    # Создаем входные и выходные последовательности
    x = []
    y = []
    for i in range(len(close_prices) - seq_length - 1):
        _x = close_prices[i:(i+seq_length)]
        _y = close_prices[i+seq_length]
        x.append(_x)
        y.append(_y)
    x = np.array(x)
    y = np.array(y)

    # Разбиваем данные на тренировочную и тестовую выборки
    train_size = int(len(x) * 0.7)
    test_size = len(x) - train_size
    train_x, test_x = torch.from_numpy(x[0:train_size,:,:]).type(torch.Tensor), torch.from_numpy(x[train_size:len(x),:,:]).type(torch.Tensor)
    train_y, test_y = torch.from_numpy(y[0:train_size,:]).type(torch.Tensor), torch.from_numpy(y[train_size:len(y),:]).type(torch.Tensor)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=res)

    for epoch in range(700): # temp 700
        optimizer.zero_grad()
        outputs = model(train_x.to(device))
        loss = criterion(outputs, train_y.to(device))
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            if epoch<100:
                print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
            else:
                if epoch % 50 == 0:
                    print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

    # Предсказываем значения на тестовой выборке
    model.eval()
    test_predict = model(test_x.to(device))

    # Вычисляем ошибку на тестовой выборке
    test_loss = criterion(test_predict, test_y.to(device))

    # Визуализируем результаты
    test_predict = test_predict.cpu().detach().numpy()
    test_y = test_y.cpu().detach().numpy()

    test_predict = scaler.inverse_transform(test_predict)
    test_y = scaler.inverse_transform(test_y)

    return test_y, test_predict, test_loss

def before_predict(code):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Задаем гиперпараметры модели
    input_size = 1
    hidden_size = 32
    num_layers = 2
    output_size = 1
    seq_length = 30 # длина входной последовательности (количество дней для предсказания)
    code_data = load_stock_dataset(code)

    dl_model = GRU(input_size, hidden_size, num_layers, output_size, device).to(device)
    all_prices, data_visual, result_predict = predict_n_days(code_data, dl_model, seq_length, device, 0.03)
    return all_prices, data_visual, result_predict


def predict_n_days(df, model, seq_length, device, res=0.03, n_days=30):
    close_prices = df['Close'].values.reshape(-1, 1)

    # Нормализуем данные
    scaler = MinMaxScaler()
    close_prices = scaler.fit_transform(close_prices)

    df = df.drop(columns=['Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits'])

    # Создаем входные и выходные последовательности
    x = []
    y = []
    for i in range(len(close_prices) - seq_length - 1):
        _x = close_prices[i:(i + seq_length)]
        _y = close_prices[i + seq_length]
        x.append(_x)
        y.append(_y)
    x = np.array(x)
    y = np.array(y)

    # Разбиваем данные на тренировочную и тестовую выборки
    train_size = len(x)
    train_x = torch.from_numpy(x[0:train_size, :, :]).type(torch.Tensor)
    train_y = torch.from_numpy(y[0:train_size, :]).type(torch.Tensor)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=res)

    for epoch in range(700):
        optimizer.zero_grad()
        outputs = model(train_x.to(device))
        loss = criterion(outputs, train_y.to(device))
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            if epoch < 100:
                print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
            else:
                if epoch % 50 == 0:
                    print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

    # Предсказываем значения на тестовой выборке
    model.eval()

    x_first = []
    result_predict = []
    x_first.append(close_prices[len(close_prices) - seq_length:])
    predict_x = np.array(x_first)
    predict_x = torch.from_numpy(predict_x).type(torch.Tensor)
    for i in range(n_days):
        test_predict = model(predict_x.to(device))
        test_predict = test_predict.cpu().detach().numpy()
        result_predict.append(test_predict[0][0])

        predict_x = predict_x[:, 1:, :]
        new_element = torch.tensor([test_predict])
        predict_x = torch.cat((predict_x, new_element), dim=1)

        # Визуализируем результаты
    result_predict = np.array(result_predict)
    len_visual = int(len(close_prices) * 0.3)
    all_prices = np.concatenate((close_prices[len(close_prices) - len_visual:], result_predict.reshape(-1, 1)))

    return all_prices, close_prices[len(close_prices) - len_visual:], result_predict



