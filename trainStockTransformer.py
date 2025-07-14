import joblib
import numpy as np
import pandas as pd
import wandb
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from models.Transformers.stockTransformer import Stockformer # Giả định bạn lưu định nghĩa Stockformer vào file stockformer.py

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT_DIR = "weight/stockformer/"

def create_dataloader(path, stockList, window_size=10, batch_size=32):
    """
    Create DataLoader for time series data for multiple stocks (giống code mẫu).
    Trả về dict dataloader cho từng stock.
    """
    df_ = {}
    for i in stockList:
        df_[i] = pd.read_csv(path + i + ".csv", index_col=None)

    df_new = {}
    for i in stockList:
        df_new[i] = {}
        df_new[i]["Train"] = df_[i].query('Date <=  "2023-12-31"').reset_index(drop=True)
        df_new[i]["Val"]  = df_[i].query('Date >= "2024-01-01" and Date <= "2024-12-31"').reset_index(drop=True)
        df_new[i]["Test"] = df_[i].query('Date >= "2025-01-01"').reset_index(drop=True)

    for i in stockList:
        for split in ["Train", "Val", "Test"]:
            for col in ['Open', 'High','Low','Adj Close','Volume']:
                if col in df_new[i][split].columns:
                    df_new[i][split].drop(columns=[col], inplace=True)

    transform_train = {}
    transform_val = {}
    transform_test = {}
    for i in stockList:
        sc = MinMaxScaler(feature_range=(0,1))
        a0 = np.array(df_new[i]["Train"]['Close']).reshape(-1,1)
        a1 = np.array(df_new[i]["Val"]['Close']).reshape(-1,1)
        a2 = np.array(df_new[i]["Test"]['Close']).reshape(-1,1)
        transform_train[i] = sc.fit_transform(a0)
        transform_val[i] = sc.transform(a1)
        transform_test[i] = sc.transform(a2)
        joblib.dump(sc, f"weight/stockformer/scaler_{i}.pkl")
    # Tạo sliding window cho từng stock
    trainset, valset, testset = {}, {}, {}
    for j in stockList:
        # Train
        X_train, y_train = [], []
        train_len = len(transform_train[j])
        for i in range(window_size, train_len):
            X_train.append(transform_train[j][i-window_size:i, 0])
            y_train.append(transform_train[j][i, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)
        trainset[j] = {
            "X": X_train.reshape(X_train.shape[0], X_train.shape[1], 1),
            "y": y_train
        }
        # Val
        X_val, y_val = [], []
        val_len = len(transform_val[j])
        for i in range(window_size, val_len):
            X_val.append(transform_val[j][i-window_size:i, 0])
            y_val.append(transform_val[j][i, 0])
        X_val, y_val = np.array(X_val), np.array(y_val)
        valset[j] = {
            "X": X_val.reshape(X_val.shape[0], X_val.shape[1], 1),
            "y": y_val
        }
        # Test
        X_test, y_test = [], []
        test_len = len(transform_test[j])
        for i in range(window_size, test_len):
            X_test.append(transform_test[j][i-window_size:i, 0])
            y_test.append(transform_test[j][i, 0])
        X_test, y_test = np.array(X_test), np.array(y_test)
        testset[j] = {
            "X": X_test.reshape(X_test.shape[0], X_test.shape[1], 1),
            "y": y_test
        }
    # Báo lại kích thước
    arr_buff = []
    for i in stockList:
        buff = {
            "X_train": trainset[i]["X"].shape,
            "y_train": trainset[i]["y"].shape,
            "X_val": valset[i]["X"].shape,
            "y_val": valset[i]["y"].shape,
            "X_test": testset[i]["X"].shape,
            "y_test": testset[i]["y"].shape,
        }
        arr_buff.append(buff)
    print(pd.DataFrame(arr_buff, index=stockList))

    # Tạo DataLoader cho từng stock
    dataloaders = {}
    for stock in stockList:
        train_data = TensorDataset(
            torch.tensor(trainset[stock]["X"], dtype=torch.float32),
            torch.tensor(trainset[stock]["y"], dtype=torch.float32)
        )
        val_data = TensorDataset(
            torch.tensor(valset[stock]["X"], dtype=torch.float32),
            torch.tensor(valset[stock]["y"], dtype=torch.float32)
        )
        test_data = TensorDataset(
            torch.tensor(testset[stock]["X"], dtype=torch.float32),
            torch.tensor(testset[stock]["y"], dtype=torch.float32)
        )
        dataloaders[stock] = {
            'train': DataLoader(train_data, batch_size=batch_size, shuffle=True),
            'val': DataLoader(val_data, batch_size=batch_size, shuffle=False),
            'test': DataLoader(test_data, batch_size=batch_size, shuffle=False)
        }
    return dataloaders

def train_model(model, stock, dataloader, criterion, optimizer, window_size, num_features, num_epochs=50, batch_size=32):
    """
    Train the model (vd: Stockformer) cho từng stock, giữ giống pipeline như code mẫu.
    """
    import wandb
    import os
    import torch
    import numpy as np
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import joblib

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CKPT_DIR = "weight/stockformer/"
    os.makedirs(CKPT_DIR, exist_ok=True)
    OUTPUT_DIR = "output/stockformer/"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    wandb.init(project='Stockformer',
            name="Stockformer_" + str(stock),
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
            # Stockformer trả về (batch, 1)
            outputs = model(inputs).squeeze(-1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        wandb.log({"epoch": epoch + 1, "loss": loss.item()})
        # Validate
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for inputs, targets in dataloader['val']:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs).squeeze(-1)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
            val_loss /= len(dataloader['val'])
            print(f'Validation Loss: {val_loss:.4f}')
            wandb.log({"epoch": epoch + 1, "val_loss": val_loss})
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), best_path)
                wandb.save(best_path)
    wandb.finish()

    # Test best model
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print("Testing model...")
    with torch.no_grad():
        test_loss = 0.0
        for inputs, targets in dataloader['test']:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs).squeeze(-1)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
        test_loss /= len(dataloader['test'])
        print(f'Test Loss: {test_loss:.4f}')
    # Predict and plot
    y_pred, y_true = [], []
    with torch.no_grad():
        for inputs, targets in dataloader['test']:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs).squeeze(-1)
            y_pred.append(outputs.cpu().numpy())
            y_true.append(targets.cpu().numpy())
    y_pred = np.concatenate(y_pred, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    # Inverse transform
    scaler = joblib.load(f"weight/stockformer/scaler_{stock}.pkl")
    y_pred_scaled = scaler.inverse_transform(y_pred.reshape(-1,1)).flatten()
    y_true_scaled = scaler.inverse_transform(y_true.reshape(-1,1)).flatten()
    # Plot
    plt.figure(figsize=(14, 7))
    plt.plot(y_true_scaled, label='True Price', color='blue')
    plt.plot(y_pred_scaled, label='Predicted Price', color='red')
    plt.title(f'Stockformer: Test Predictions vs True Values for {stock}')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.savefig(f'{OUTPUT_DIR}predictions_{stock}.png')

if __name__ == "__main__":
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs("output/stockformer", exist_ok=True)
    path = "data/"
    stockList = ['AMZN', 'NVDA','AAPL','BIDU','GOOG','INTC','MSFT','NFLX','TCEHY','TSLA']
    window_size = 24
    batch_size = 128
    num_features = 1  # Chỉ sử dụng giá đóng cửa
    dataloaders = create_dataloader(path, stockList, window_size, batch_size)
    for stock in stockList:
        # input_dim = num_features = 1 nếu đơn biến, >1 nếu đa biến (bổ sung channel cho Stockformer)
        model = Stockformer(input_dim=1, embed_dim=128, num_heads=8, num_layers=2, out_dim=1).to(DEVICE)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        train_model(model, stock, dataloaders[stock], criterion, optimizer, window_size, num_features=1, num_epochs=100, batch_size=batch_size)
    print("Training complete.")