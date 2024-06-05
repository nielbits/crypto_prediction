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
import mplfinance as mpf
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

def get_yahoo_finance_data(tickers, interval='1d', start='2023-04-01', end='2024-05-01'):
    all_data = pd.DataFrame()
    for ticker in tickers:
        data = yf.download(ticker, start=start, end=end, interval=interval)
        data['Ticker'] = ticker
        all_data = pd.concat([all_data, data])
    all_data.index = pd.to_datetime(all_data.index)  # Ensure index is a DatetimeIndex
    return all_data

def add_technical_indicators(df):
    df['SMA'] = SMAIndicator(close=df['Close'], window=20).sma_indicator()
    df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
    macd = MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd.macd()
    df['SignalLine'] = macd.macd_signal()
    bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BollingerUpper'] = bb.bollinger_hband()
    df['BollingerLower'] = bb.bollinger_lband()
    df.dropna(inplace=True)
    return df

def normalize_data(df):
    scaler = MinMaxScaler()
    feature_columns = ['Open', 'High', 'Low', 'Close', 'SMA', 'RSI', 'MACD', 'SignalLine', 'BollingerUpper', 'BollingerLower']
    df_features = df[feature_columns]
    df[feature_columns] = scaler.fit_transform(df_features)
    return df, scaler

def plot_candlestick(df, predicted_dates, predicted_values, actual_values, ticker, features, scaler_transform, display_plot=False, plot_tech_indicators=False, historical_data_length=360):
    df = df.copy()

    # Filter data to include only the required amount of previous data
    df = df.iloc[-historical_data_length:]
    
    # Ensure Date is a column
    if 'Date' not in df.columns:
        df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Initialize the list of additional plots
    addplots = []

    if plot_tech_indicators:
        addplots = [
            mpf.make_addplot(df['SMA'], color='blue', panel=0),
            mpf.make_addplot(df['BollingerUpper'], color='gray', panel=0),
            mpf.make_addplot(df['BollingerLower'], color='gray', panel=0),
            mpf.make_addplot(df['RSI'], color='purple', panel=1),
            mpf.make_addplot(df['MACD'], color='orange', panel=2),
            mpf.make_addplot(df['SignalLine'], color='blue', panel=2)
        ]

    # Create the candlestick chart
    fig, axlist = mpf.plot(df[['Open','High','Low','Close','Volume']], type='candle', style='charles', volume=True, 
                           addplot=addplots, returnfig=True, figsize=(14, 10), panel_ratios=(3,1,1), show_nontrading=True, mav=(21,200))

    ax1 = axlist[0]

    # Convert predicted dates to the correct format
    predicted_dates = mdates.date2num(pd.to_datetime(predicted_dates))
    ax1.plot(predicted_dates, predicted_values, 'bo--', label='Predicted', markersize=3)
    
    if actual_values is not None:
        ax1.plot(predicted_dates, actual_values, 'rx', label='Actual', markersize=5)  # Plot actual values as brown crosses
    
    ax1.set_title(f'Candlestick Chart with Technical Indicators and Prediction for {ticker}')
    ax1.legend(loc='upper left')
    
    # Save the plot with higher DPI for better resolution
    plt.savefig(f'candlestick_prediction_{ticker}.png', dpi=300)
    if display_plot:
        plt.show()
    plt.close()

class CNNRNNModel(nn.Module):
    def __init__(self, input_channels=10, cnn_channels=16, rnn_hidden_size=64, dropout_prob=0.5, sequence_length=30):
        super(CNNRNNModel, self).__init__()
        
        # Define CNN layers
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, cnn_channels, kernel_size=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2),
            nn.Conv1d(cnn_channels, cnn_channels * 2, kernel_size=4),
            nn.BatchNorm1d(cnn_channels * 2),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2),
            nn.Dropout(dropout_prob)
        )

        # Calculate the output size after the CNN layers
        cnn_output_size = self._get_cnn_output_size(sequence_length, input_channels)

        # Define FC and RNN layers
        self.fc_hidden = nn.Sequential(
            nn.Linear(cnn_output_size, rnn_hidden_size),
            nn.BatchNorm1d(rnn_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )
        self.rnn = nn.LSTM(input_size=rnn_hidden_size, hidden_size=rnn_hidden_size, num_layers=1, batch_first=True)
        self.fc_output = nn.Linear(rnn_hidden_size, 1)

    def _get_cnn_output_size(self, sequence_length, input_channels):
        # Create a dummy input to calculate the output size
        dummy_input = torch.zeros(1, input_channels, sequence_length)
        output = self.cnn(dummy_input)
        return output.view(1, -1).size(1)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc_hidden(x)
        x = x.unsqueeze(1)
        x, _ = self.rnn(x)
        x = x[:, -1, :]
        final_output = self.fc_output(x)
        return final_output

def train_model(model, data, batch_size=32, epochs=50, sequence_length=30, learning_rate=0.001, log_dir='runs', patience=50, ticker=''):
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
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            for x_batch, y_batch in loader:
                if x_batch.size(0) > 1:  # Ensure batch size is greater than 1 for batch normalization
                    optimizer.zero_grad()
                    output = model(x_batch)
                    loss = criterion(output.squeeze(), y_batch)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * x_batch.size(0)
            
            total_loss /= len(loader.dataset)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.8f}')
                
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
    return model, total_loss

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model

def predict_and_plot(model, data, ticker, scaler, sequence_length=30, interval='1d', display_plot=False, plot_tech_indicators=False, historical_data_length=360, align_prediction_with_data=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Filter data for the specified ticker
    data = data[data['Ticker'] == ticker]
    
    # Ensure historical data length is within the start and end dates
    if historical_data_length > len(data):
        historical_data_length = len(data)
        print(f"Adjusted historical data length to {historical_data_length} due to available data range.")
    
    # Preserve the date index as a column
    data['Date'] = data.index
    
    feature_columns = ['Open', 'High', 'Low', 'Close', 'SMA', 'RSI', 'MACD', 'SignalLine', 'BollingerUpper', 'BollingerLower']
    data_scaled = scaler.transform(data[feature_columns])
    
    # Prepare input data for prediction without offset
    if align_prediction_with_data:
        x_data = torch.tensor(data_scaled[-2*sequence_length:-sequence_length], dtype=torch.float32).unsqueeze(0).permute(0, 2, 1).to(device)
    else:
        x_data = torch.tensor(data_scaled[-sequence_length:], dtype=torch.float32).unsqueeze(0).permute(0, 2, 1).to(device)

    predictions = []

    x_input = x_data
    
    for _ in range(sequence_length):
        with torch.no_grad():
            final_output = model(x_input)
            predicted = final_output.squeeze().item()
        
        # Ensure correct scaling
        predicted_value_scaled = np.zeros(len(feature_columns))
        predicted_value_scaled[feature_columns.index('Close')] = predicted
        
        # Inverse transform the predicted value
        predicted_value = scaler.inverse_transform([predicted_value_scaled])[0][feature_columns.index('Close')]
        predictions.append(predicted_value)
        
        # Update input for the next prediction step
        new_input = np.array([predicted_value_scaled])
        new_input = torch.tensor(new_input, dtype=torch.float32).to(device)
        x_input = torch.cat((x_input[:, :, 1:], new_input.unsqueeze(2)), dim=2)

    if align_prediction_with_data:
        # Extract the last portion of the data for plotting previous candlestick data
        last_data = data.iloc[-(historical_data_length + sequence_length):-sequence_length]
    else:
        # Extract the last portion of the data for plotting previous candlestick data
        last_data = data.iloc[-historical_data_length:]
    last_date = last_data['Date'].iloc[-1]  # Get the most recent date in the data

    # Generate future dates based on the last date and the interval
    if interval in ['15m', '1h', '4h']:
        future_dates = pd.date_range(start=last_date, periods=sequence_length+1, freq=interval)
    elif interval == '1d':
        future_dates = pd.date_range(start=last_date, periods=sequence_length+1, freq='D')
    elif interval == '1wk':
        future_dates = pd.date_range(start=last_date, periods=sequence_length+1, freq='W')
    
    future_dates = future_dates[1:]  # Skip the first date since it's the same as the last_date

    if align_prediction_with_data:
        # Ensure future_dates are within the available data range
        future_dates = future_dates[future_dates <= data.index.max()]
        if len(future_dates) == 0:
            print("No future dates available within the data range.")
            return
        # Extract the actual closing prices for the prediction period
        actual_close_prices = data.loc[future_dates, 'Close'].values
    else:
        actual_close_prices = None
    
    # Plot the candlestick chart with previous data and predicted values
    plot_candlestick(last_data, future_dates, predictions[:len(future_dates)], actual_close_prices, ticker, feature_columns, scaler, display_plot, plot_tech_indicators, historical_data_length)

if __name__ == "__main__":
    tickers = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'XRP-USD', 'SOL-USD', 'DOT-USD', 'DOGE-USD', 'UNI-USD', 'LTC-USD', 
               'LINK-USD', 'BCH-USD', 'XLM-USD', 'FIL-USD', 'TRX-USD', 'EOS-USD', 'ATOM-USD', 'ETC-USD', 'XTZ-USD', 'MKR-USD']
    
    chosen_ticker = 'BTC-USD'
    mode = 'predict'
    continue_training = False  # Set this to False to train from scratch
    
    # Hyperparameters
    sequence_length = 15  # Sequence length
    batch_size = 12       # Batch size
    epochs = 50          # Number of epochs
    learning_rate = 0.0001 # Learning rate
    interval = '1d'       # Data interval
    log_dir = 'runs'      # Directory for TensorBoard logs
    model_path = 'model.pth'  # Model save path
    plot_tech_indicators = True
    start_date = '2024-03-01'
    end_date = '2024-06-01'
    align_prediction_with_data = True  # Set this flag to align prediction with available data
    historical_data_length = 60  # Set the length of historical data to be used
    training_loops = 10  # Define how many times the training loop will run

    scaler_dict = {}
    total_losses = []

    if mode == 'train':
        all_data = get_yahoo_finance_data(tickers, interval=interval, start=start_date, end=end_date)
        
        # Apply technical indicators without resetting the index
        all_data = all_data.groupby('Ticker').apply(add_technical_indicators).reset_index(level=0, drop=True)
        
        if not all_data.empty:
            model = CNNRNNModel(input_channels=10, rnn_hidden_size=batch_size, sequence_length=sequence_length)
            
            if continue_training and os.path.exists(model_path):
                model = load_model(model, model_path)
                print("Loaded existing model for further training.")
            else:
                print("Training a new model from scratch.")
            
            for _ in range(training_loops):
                for ticker in tickers:
                    print(f"Processing data for {ticker}...")
                    ticker_data = all_data[all_data['Ticker'] == ticker]
                    ticker_data, scaler_dict[ticker] = normalize_data(ticker_data)
                    print(f"Training the model with {ticker} data...")
                    model, total_loss = train_model(model, ticker_data[['Open', 'High', 'Low', 'Close', 'SMA', 'RSI', 'MACD', 'SignalLine', 'BollingerUpper', 'BollingerLower']].values, batch_size=batch_size, epochs=epochs, sequence_length=sequence_length, learning_rate=learning_rate, log_dir=log_dir, ticker=ticker)
                    total_losses.append(total_loss)
            
            save_model(model, model_path)
            
            # Calculate and display the average loss
            avg_loss = np.mean(total_losses)
            print(f"Average Loss over all models: {avg_loss}")
            
            # Log the average loss to TensorBoard
            writer = SummaryWriter(log_dir=log_dir)
            writer.add_scalar('Loss/average', avg_loss)
            writer.close()
            
            predict_and_plot(model, all_data, chosen_ticker, scaler_dict[chosen_ticker], sequence_length=sequence_length, interval=interval, display_plot=True, plot_tech_indicators=plot_tech_indicators, historical_data_length=historical_data_length, align_prediction_with_data=align_prediction_with_data)
        else:
            print("No data available for the specified parameters.")
    elif mode == 'predict':
        all_data = get_yahoo_finance_data([chosen_ticker], interval=interval, start=start_date, end=end_date)
        all_data = add_technical_indicators(all_data)
        all_data, scaler_dict[chosen_ticker] = normalize_data(all_data)
        features = ['Open', 'High', 'Low', 'Close', 'SMA', 'RSI', 'MACD', 'SignalLine', 'BollingerUpper', 'BollingerLower']
        model = CNNRNNModel(input_channels=10, rnn_hidden_size=batch_size, sequence_length=sequence_length)
        model = load_model(model, model_path)
        all_data[features] = scaler_dict[chosen_ticker].inverse_transform(all_data[features])
        print(f"Predicting and plotting for {chosen_ticker}...")
        predict_and_plot(model, all_data, chosen_ticker, scaler_dict[chosen_ticker], sequence_length=sequence_length, interval=interval, display_plot=True, plot_tech_indicators=plot_tech_indicators, historical_data_length=historical_data_length, align_prediction_with_data=align_prediction_with_data)
