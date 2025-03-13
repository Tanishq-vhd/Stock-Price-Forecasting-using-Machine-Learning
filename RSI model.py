import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


stock = yf.download('AAPL', start='2020-01-01', end='2024-01-01')[['Close']]


if stock.empty:
    raise ValueError("Stock data not loaded. Check ticker symbol or internet connection.")


stock['MA_10'] = stock['Close'].rolling(window=10).mean()
stock['MA_50'] = stock['Close'].rolling(window=50).mean()


delta = stock['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
stock['RSI'] = 100 - (100 / (1 + rs))


stock.dropna(inplace=True)

print(stock[['Close', 'MA_10', 'MA_50', 'RSI']].tail())


stock['Prediction'] = stock['Close'].shift(-1)


stock.dropna(inplace=True)


X = np.array(stock[['Close', 'MA_10', 'MA_50', 'RSI']])
y = np.array(stock['Prediction'])


if X.shape[0] == 0 or y.shape[0] == 0:
    raise ValueError("Error: X or y is empty. Check data preprocessing steps.")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

print(f"Train Size: {X_train.shape[0]}, Test Size: {X_test.shape[0]}")
