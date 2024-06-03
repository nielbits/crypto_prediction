import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import yfinance as yf
from sklearn.model_selection import TimeSeriesSplit
import os

plt.switch_backend('agg')

def get_yahoo_finance_data(tickers, interval='1d', start='2023-04-01', end='2024-05-01'):
    all_data = pd.DataFrame()
    for ticker in tickers:
        data = yf.download(ticker, start=start, end=end, interval=interval)
        data['Ticker'] = ticker
        all_data = pd.concat([all_data, data])
    all_data.index = pd.to_datetime(all_data.index)  # Ensure index is a DatetimeIndex
    return all_data

def add_technical_indicators(df):
    df['SMA'] = df['Close'].rolling(window=20).mean()
    df['RSI'] = compute_rsi(df['Close'])
    df['MACD'], df['SignalLine'] = compute_macd(df['Close'])
    df['BollingerUpper'], df['BollingerLower'] = compute_bollinger_bands(df['Close'])
    df.dropna(inplace=True)
    return df

def compute_rsi(series, period=14):
    delta = series.diff(1)
    gain = delta.mask(delta < 0, 0)
    loss = -delta.mask(delta > 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(series, fast=12, slow=26, signal=9):
    fast_ema = series.ewm(span=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, adjust=False).mean()
    macd = fast_ema - slow_ema
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def compute_bollinger_bands(series, window=20, num_std_dev=2):
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std_dev)
    lower_band = rolling_mean - (rolling_std * num_std_dev)
    return upper_band, lower_band

def normalize_data(df):
    scaler = MinMaxScaler()
    feature_columns = ['Open', 'High', 'Low', 'Close', 'SMA', 'RSI', 'MACD', 'SignalLine', 'BollingerUpper', 'BollingerLower']
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    return df, scaler


def plot_candlestick(df, predicted_dates, predicted_values, ticker, features, scaler_transform,display_plot=False, plot_tech_indicators=False, previous_data_length=360):
    df[features] = scaler_transform.inverse_transform(df[features])
    df = df.copy()
    df['Date'] = df.index.map(mdates.date2num)
    
    # Filter data to include only the required amount of previous data
    df = df.iloc[-previous_data_length:]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 16), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})
    
    for idx, row in df.iterrows():
        color = 'green' if row['Close'] > row['Open'] else 'red'
        ax1.plot([row['Date'], row['Date']], [row['Low'], row['High']], color=color)
        ax1.plot([row['Date'], row['Date']], [row['Open'], row['Close']], color=color, linewidth=2)
    
    if plot_tech_indicators:
        # Recalculate technical indicators using original data
        df['SMA'] = df['Close'].rolling(window=20).mean()
        df['RSI'] = compute_rsi(df['Close'])
        df['MACD'], df['SignalLine'] = compute_macd(df['Close'])
        df['BollingerUpper'], df['BollingerLower'] = compute_bollinger_bands(df['Close'])
        
        ax1.plot(df['Date'], df['SMA'], label='SMA', color='blue')
        ax1.fill_between(df['Date'], df['BollingerUpper'], df['BollingerLower'], color='gray', alpha=0.2, label='Bollinger Bands')
        ax2.plot(df['Date'], df['RSI'], label='RSI', color='purple', linestyle='--')
        ax2.axhline(70, color='red', linestyle='--')
        ax2.axhline(30, color='green', linestyle='--')
        ax2.set_ylim([0, 100])
        ax2.set_ylabel('RSI')
        ax2.legend(loc='upper left')
        ax3.plot(df['Date'], df['MACD'], label='MACD', color='orange')
        ax3.plot(df['Date'], df['SignalLine'], label='Signal Line', color='blue', linestyle='--')
        ax3.set_ylabel('MACD')
        ax3.legend(loc='upper left')
    
    # Convert predicted dates to the correct format
    predicted_dates = [mdates.date2num(pd.to_datetime(date)) for date in predicted_dates]
    ax1.plot(predicted_dates, predicted_values, 'bo--', label='Predicted', markersize=3)
    ax1.xaxis_date()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()
    ax1.set_ylabel('Price')
    ax1.set_title(f'Candlestick Chart with Technical Indicators and Prediction for {ticker}')
    ax1.legend(loc='upper left')
    
    plt.savefig(f'candlestick_prediction_{ticker}.png')
    if display_plot:
        plt.show()
    plt.close()

class CNNRNNModel(nn.Module):
    def __init__(self, input_channels=10, cnn_channels=16, rnn_hidden_size=32, dropout_prob=0.5, sequence_length=30):
        super(CNNRNNModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2),
            nn.Conv1d(cnn_channels, cnn_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels * 2),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2),
            nn.Dropout(dropout_prob)
        )
        self.fc_hidden = nn.Sequential(
            nn.Linear(cnn_channels * 2 * (sequence_length // 4), rnn_hidden_size),
            nn.BatchNorm1d(rnn_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )
        self.rnn = nn.LSTM(input_size=rnn_hidden_size, hidden_size=rnn_hidden_size, num_layers=1, batch_first=True)
        self.fc_output = nn.Linear(rnn_hidden_size, 1)
    
    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc_hidden(x)
        x = x.unsqueeze(1)
        x, _ = self.rnn(x)
        x = x[:, -1, :]
        final_output = self.fc_output(x)
        return final_output

def train_model(model, data, batch_size=32, epochs=50, sequence_length=30, log_dir='runs', patience=10, ticker=''):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    writer = SummaryWriter(log_dir=os.path.join(log_dir, ticker))
    
    x_data, y_data = [], []
    for i in range(len(data) - sequence_length):
        x_data.append(data[i:i + sequence_length])
        y_data.append(data[i + sequence_length, 3])  # We predict the Close price
    
    x_data, y_data = np.array(x_data), np.array(y_data)
    
    tscv = TimeSeriesSplit(n_splits=5)
    best_loss = float('inf')
    best_model = None
    patience_counter = 0
    
    for train_index, test_index in tscv.split(x_data):
        x_train, x_test = x_data[train_index], x_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]
        
        x_train = torch.tensor(x_train, dtype=torch.float32).permute(0, 2, 1).to(device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
        x_test = torch.tensor(x_test, dtype=torch.float32).permute(0, 2, 1).to(device)
        y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
        
        dataset = TensorDataset(x_train, y_train)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                output = model(x_batch)
                loss = criterion(output.squeeze(), y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * x_batch.size(0)
            
            total_loss /= len(loader.dataset)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}')
                writer.add_scalar('Loss/train', total_loss, epoch)
                
            model.eval()
            with torch.no_grad():
                val_output = model(x_test)
                val_loss = criterion(val_output.squeeze(), y_test).item()
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
    
    writer.close()
    if best_model is not None:
        model.load_state_dict(best_model)
    return model

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model

def predict_and_plot(model, data, ticker, scaler, sequence_length=30, interval='1d', display_plot=False, plot_tech_indicators=False, previous_data_length=360):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Filter data for the specified ticker
    data = data[data['Ticker'] == ticker]
    
    feature_columns = ['Open', 'High', 'Low', 'Close', 'SMA', 'RSI', 'MACD', 'SignalLine', 'BollingerUpper', 'BollingerLower']
    data_scaled = scaler.transform(data[feature_columns])
    
    # Prepare input data for prediction
    x_data = torch.tensor(data_scaled[-sequence_length:], dtype=torch.float32).unsqueeze(0).permute(0, 2, 1).to(device)
    
    predictions = []
    x_input = x_data
    
    for _ in range(sequence_length):
        with torch.no_grad():
            final_output = model(x_input)
            predicted = final_output.squeeze().item()
        
        predicted_value_scaled = np.array([0, 0, 0, predicted, 0, 0, 0, 0, 0, 0])
        predicted_value = scaler.inverse_transform([predicted_value_scaled])[0][3]
        predictions.append(predicted_value)
        
        new_input = torch.tensor([[0, 0, 0, predicted, 0, 0, 0, 0, 0, 0]], dtype=torch.float32).to(device)
        x_input = torch.cat((x_input[:, :, 1:], new_input.unsqueeze(2)), dim=2)
    
    print(f'Predicted values for next {sequence_length} intervals for {ticker}: {predictions}')
    
    # Extract the last portion of the data for plotting previous candlestick data
    last_data = data.iloc[-previous_data_length:]
    last_date = last_data.index[-1]  # Get the most recent date in the data


    # Generate future dates based on the last date and the interval
    if interval in ['15m', '1h', '4h']:
        future_dates = pd.date_range(start=last_date, periods=sequence_length+1, freq=interval)
    elif interval == '1d':
        future_dates = pd.date_range(start=last_date, periods=sequence_length+1, freq='D')
    elif interval == '1wk':
        future_dates = pd.date_range(start=last_date, periods=sequence_length+1, freq='W')
    
    future_dates = future_dates[1:]  # Skip the first date since it's the same as the last_date

    # Plot the candlestick chart with previous data and predicted values
    plot_candlestick(last_data, future_dates, predictions, ticker,feature_columns,scaler,
                      display_plot, plot_tech_indicators,previous_data_length)

  
if __name__ == "__main__":
    tickers = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'XRP-USD', 'SOL-USD', 'DOT-USD', 'DOGE-USD', 'UNI-USD', 'LTC-USD', 
               'LINK-USD', 'BCH-USD', 'XLM-USD', 'FIL-USD', 'TRX-USD', 'EOS-USD', 'ATOM-USD', 'ETC-USD', 'XTZ-USD', 'MKR-USD']
    
    chosen_ticker = 'ETH-USD'
    mode = 'train'
    continue_training = False  # Set this to False to train from scratch
    
    sequence_length = 120
    batch_size = 12
    epochs = 500
    interval = '1d'
    log_dir = 'runs'
    model_path = 'model.pth'
    plot_tech_indicators = True
    
    if mode == 'train':
        all_data = get_yahoo_finance_data(tickers, interval=interval)
        
        all_data = all_data.groupby('Ticker').apply(add_technical_indicators).reset_index(drop=True)
        
        if not all_data.empty:
            scaler = MinMaxScaler()
            model = CNNRNNModel(input_channels=10, rnn_hidden_size=batch_size, sequence_length=sequence_length)
            
            if continue_training and os.path.exists(model_path):
                model = load_model(model, model_path)
                print("Loaded existing model for further training.")
            else:
                print("Training a new model from scratch.")
            
            for ticker in tickers:
                print(f"Processing data for {ticker}...")
                ticker_data = all_data[all_data['Ticker'] == ticker]
                ticker_data, scaler = normalize_data(ticker_data)
                print(f"Training the model with {ticker} data...")
                model = train_model(model, ticker_data[['Open', 'High', 'Low', 'Close', 'SMA', 'RSI', 'MACD', 'SignalLine', 'BollingerUpper', 'BollingerLower']].values, batch_size=batch_size, epochs=epochs, sequence_length=sequence_length, log_dir=log_dir, ticker=ticker)
            
            save_model(model, model_path)
            
            print(f"Predicting and plotting for {chosen_ticker}...")
            predict_and_plot(model, all_data, chosen_ticker, scaler, sequence_length=sequence_length, interval=interval, display_plot=True, plot_tech_indicators=plot_tech_indicators)
        else:
            print("No data available for the specified parameters.")
    elif mode == 'predict':
        all_data = get_yahoo_finance_data([chosen_ticker], interval=interval)
        all_data = add_technical_indicators(all_data)
        all_data, scaler = normalize_data(all_data)
        
        model = CNNRNNModel(input_channels=10, rnn_hidden_size=batch_size, sequence_length=sequence_length)
        model = load_model(model, model_path)
        
        print(f"Predicting and plotting for {chosen_ticker}...")
        predict_and_plot(model, all_data, chosen_ticker, scaler, sequence_length=sequence_length, interval=interval, display_plot=True, plot_tech_indicators=plot_tech_indicators)
