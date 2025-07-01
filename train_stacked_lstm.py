from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from models.LSTM.stacked_lstm import StackedLSTM
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error


import wandb
from wandb.integration.keras import WandbCallback
import shutil as sh

import joblib

import warnings
warnings.filterwarnings("ignore")


def train_stacked_lstm(window_size, 
                    num_features, 
                    trainset,
                    epochs=100, 
                    batch_size=32, 
                    valset=None,
                    testset=None):
    
    print("Model Summary:")
    print(StackedLSTM(window_size, num_features).summary())
    

    for i in stockList:
        wandb.init(project='DAT',
                name="final",
                config={
                    "window_size": window_size,
                    "num_features": num_features,
                    "epochs": epochs,
                    "batch_size": batch_size
                })
        
        model = StackedLSTM(window_size, num_features)

        print("Fitting to", i)
        model.fit(trainset[i]["X"], trainset[i]["y"], 
                epochs=epochs, batch_size=batch_size, 
                validation_data=(valset[i]["X"], valset[i]["y"]), 
                callbacks=[WandbCallback()])
    
        sh.copy("wandb/latest-run/files/model-best.h5", 
                "weight/stacked_lstm/model_" + i + ".h5")

        wandb.finish()


        for i in stockList:
            model = load_model('weight/stacked_lstm/model_' + i + '.h5')
            scaler = joblib.load("weight/stacked_lstm/scaler_" + i + ".pkl")

            y_pred_scaled = model.predict(testset[i]["X"])
            y_pred = scaler.inverse_transform(np.concatenate([y_pred_scaled, np.zeros((y_pred_scaled.shape[0], trainset[i]["X"].shape[1]-1))], axis=1))[:,0]
            y_true = scaler.inverse_transform(np.concatenate([testset[i]["y"].reshape(-1,1), np.zeros((testset[i]["y"].shape[0], trainset[i]["X"].shape[1]-1))], axis=1))[:,0]
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            print(f"MAE: {mae:.4f}")
            print(f"RMSE: {rmse:.4f}")

            plt.figure(figsize=(12, 6))
            plt.plot(y_true, label='Actual')
            plt.plot(y_pred, label='Predicted')
            plt.legend()
            plt.title('Model prediction on test data')
            plt.savefig(f"output/stacked_lstm/final_plot({i}).png", dpi=300)


    return model


def data_loader(path, stockList, window_size):
    
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
        joblib.dump(sc, "weight/stacked_lstm/scaler_" + i + ".pkl")

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

    return trainset, valset, testset

if __name__ == "__main__":
    path = "data/"
    stockList = ['AMZN', 'NVDA','AAPL','BIDU','GOOG','INTC','MSFT','NFLX','TCEHY','TSLA']

    window_size = 24
    num_features = 1

    trainset, valset, testset = data_loader(path, stockList, window_size)

    model = train_stacked_lstm(window_size, 
                            num_features, 
                            trainset, 
                            epochs=100, batch_size=64,
                            valset=valset,
                            testset=testset)