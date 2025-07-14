import joblib
import shutil as sh
import numpy as np
import pandas as pd
import wandb
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
from models.LSTM.lstm_atten_lstm import LSTM_Attention_LSTM
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT_DIR = "weight/lstm_atten_lstm/"

def create_dataloader(path, stockList, window_size=10, batch_size=32):
    """
    Create DataLoader for time series data.
    
    Parameters:
    - data: numpy array of shape (n_samples, n_features)
    - window_size: size of the sliding window
    - batch_size: size of each batch
    
    Returns:
    - DataLoader object
    """
    df_ = {}
    for i in stockList:
        df_[i] = pd.read_csv(path + i + ".csv", index_col = 'Date')
    
    df_new = {}
    for i in stockList:
        df_new[i] = {}
        df_new[i]["Train"] = df_[i].query('Date <=  "2023-12-31"').reset_index(drop = False)
        df_new[i]["Val"]  = df_[i].query('Date >= "2024-01-01" and Date <= "2024-12-31"').reset_index(drop = False)
        df_new[i]["Test"] = df_[i].query('Date >= "2025-01-01"').reset_index(drop = False)

    for i in stockList:
        df_new[i]["Train"].drop(columns=['Open', 'High','Low','Adj Close','Volume'],inplace = True)
        df_new[i]["Val"].drop(columns=['Open', 'High','Low','Adj Close','Volume'],inplace = True)
        df_new[i]["Test"].drop(columns=['Open', 'High','Low','Adj Close','Volume'],inplace = True)

    transform_train = {}
    transform_test = {}
    transform_val = {}

    for num, i in enumerate(stockList):
        sc = MinMaxScaler(feature_range=(0,1))
        a0 = np.array(df_new[i]["Train"]['Close'])
        a1 = np.array(df_new[i]["Val"]['Close'])
        a2 = np.array(df_new[i]["Test"]['Close'])
        a0 = a0.reshape(a0.shape[0],1)
        a1 = a1.reshape(a1.shape[0],1)
        a2 = a2.reshape(a2.shape[0],1)
        transform_train[i] = sc.fit_transform(a0)
        transform_val[i] = sc.transform(a1)
        transform_test[i] = sc.transform(a2)
        joblib.dump(sc, "weight/lstm_atten_lstm/scaler_" + i + ".pkl")

    del a0
    del a1
    del a2

    trainset = {}
    valset = {}
    testset = {}
    for j in stockList:
        trainset[j] = {}
        X_train = []
        y_train = []
        for i in range(window_size,3014):
            X_train.append(transform_train[j][i-window_size:i,0])
            y_train.append(transform_train[j][i,0])
        X_train, y_train = np.array(X_train), np.array(y_train)
        trainset[j]["X"] = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))
        trainset[j]["y"] = y_train

        testset[j] = {}
        X_test = []
        y_test = []
        for i in range(window_size, 121):
            X_test.append(transform_test[j][i-window_size:i,0])
            y_test.append(transform_test[j][i,0])
        X_test, y_test = np.array(X_test), np.array(y_test)
        testset[j]["X"] = np.reshape(X_test, (X_test.shape[0], X_train.shape[1], 1))
        testset[j]["y"] = y_test

        valset[j] = {}
        X_val = []
        y_val = []
        for i in range(window_size, 251):
            X_val.append(transform_val[j][i-window_size:i,0])
            y_val.append(transform_val[j][i,0])
        X_val, y_val = np.array(X_val), np.array(y_val)
        valset[j]["X"] = np.reshape(X_val, (X_val.shape[0], X_train.shape[1], 1))
        valset[j]["y"] = y_val

    print(f"Data for {j} loaded successfully.")
    arr_buff = []
    for i in stockList:
        buff = {}
        buff["X_train"] = trainset[i]["X"].shape
        buff["y_train"] = trainset[i]["y"].shape
        buff["X_val"] = valset[i]["X"].shape
        buff["y_val"] = valset[i]["y"].shape
        buff["X_test"] = testset[i]["X"].shape
        buff["y_test"] = testset[i]["y"].shape
        arr_buff.append(buff)

    print(pd.DataFrame(arr_buff, index=stockList))
    # Create DataLoader for each stock
    dataloaders = {}
    for stock in stockList:
        train_data = TensorDataset(torch.tensor(trainset[stock]["X"], dtype=torch.float32),
                                torch.tensor(trainset[stock]["y"], dtype=torch.float32))
        val_data = TensorDataset(torch.tensor(valset[stock]["X"], dtype=torch.float32),
                                torch.tensor(valset[stock]["y"], dtype=torch.float32))
        test_data = TensorDataset(torch.tensor(testset[stock]["X"], dtype=torch.float32),
                                torch.tensor(testset[stock]["y"], dtype=torch.float32))

        dataloaders[stock] = {
            'train': DataLoader(train_data, batch_size=batch_size, shuffle=True),
            'val': DataLoader(val_data, batch_size=batch_size, shuffle=False),
            'test': DataLoader(test_data, batch_size=batch_size, shuffle=False)
        }


    return dataloaders

def train_model(model, stock, dataloader, criterion, optimizer, window_size, num_features, num_epochs=50, batch_size=32):
    """
    Train the model using the provided DataLoader.
    
    Parameters:
    - model: PyTorch model to train
    - dataloader: DataLoader object containing training data
    - criterion: loss function
    - optimizer: optimizer for training
    - num_epochs: number of epochs to train
    
    Returns:
    - None
    """
    wandb.init(project='LSTM_Attention_LSTM',
                name="LAL_" + str(stock),
                config={
                    "window_size": window_size,
                    "num_features": num_features,
                    "epochs": num_epochs,
                    "batch_size": batch_size
                })
    
    best_val = float("inf")
    best_path = os.path.join(CKPT_DIR, f"{stock}_best.pt")

    model.train()

    for epoch in tqdm(range(num_epochs)):
        model.train()
        for inputs, targets in dataloader['train']:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)[:, -1, 0]
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        wandb.log({"epoch": epoch + 1, "loss": loss.item()})
        # Validate the model
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for inputs, targets in dataloader['val']:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

                outputs = model(inputs)[:, -1, 0]
                loss = criterion(outputs, targets)
                val_loss += loss.item()
            val_loss /= len(dataloader['val'])
            print(f'Validation Loss: {val_loss:.4f}')
            wandb.log({"epoch": epoch + 1, "loss": loss.item(), "val_loss": val_loss})
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), best_path)
                wandb.save(best_path)

    wandb.finish()
    
    # Test model
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    criterion = nn.MSELoss()

    # Test model and draw chart
    print("Testing model...")
    with torch.no_grad():
        test_loss = 0.0
        for inputs, targets in dataloader['test']:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            outputs = model(inputs)[:, -1, 0]
            loss = criterion(outputs, targets)
            test_loss += loss.item()
        test_loss /= len(dataloader['test'])
        print(f'Test Loss: {test_loss:.4f}')

    # Predict on test set
    y_pred = []
    y_true = []
    with torch.no_grad():
        for inputs, targets in dataloader['test']:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)[:, -1, 0]
            
            y_pred.append(outputs.cpu().numpy())
            y_true.append(targets.cpu().numpy())

    y_pred = np.concatenate(y_pred, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    # Inverse transform predictions
    scaler = joblib.load("weight/lstm_atten_lstm/scaler_" + stock + ".pkl")

    y_pred_scaled = scaler.inverse_transform(y_pred.reshape(-1,1)).flatten()
    y_true_scaled = scaler.inverse_transform(y_true.reshape(-1,1)).flatten()
    
    # Plot
    plt.figure(figsize=(14, 7))
    plt.plot(y_true_scaled, label='True Price', color='blue')
    plt.plot(y_pred_scaled, label='Predicted Price', color='red')
    plt.title(f'Test Predictions vs True Values for {stock}')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.savefig(f'output/lstm_atten_lstm/predictions_{stock}.png')

if __name__ == "__main__":
    path = "data/"
    stockList = ['AMZN', 'NVDA','AAPL','BIDU','GOOG','INTC','MSFT','NFLX','TCEHY','TSLA']

    window_size = 24
    batch_size = 128
    num_features = 1

    dataloaders = create_dataloader(path, stockList, window_size, batch_size)

    for stock in stockList:
        model = LSTM_Attention_LSTM(input_dim=num_features, hidden_dim=100, output_dim=1).to(DEVICE)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        print(f"Training model for {stock}...")
        train_model(model, stock, dataloaders[stock], criterion, optimizer, window_size, num_features, num_epochs=100, batch_size=batch_size)

    print("Training complete.")
