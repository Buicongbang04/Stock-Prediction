import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import os
plt.style.use("fivethirtyeight")

st.title("Stock Price Prediction with LSTM")
st.write(
    'Dự đoán giá cổ phiếu sử dụng LSTM. Bạn có thể nhập mã cổ phiếu và nhận các biểu đồ phân tích cùng dự đoán.'
)

# --- Khởi tạo đường dẫn lưu ảnh
if not os.path.exists("static"):
    os.makedirs("static")

# --- Nhập mã cổ phiếu
stock = st.text_input('Nhập mã cổ phiếu (ví dụ: POWERGRID.NS, AAPL):', value='POWERGRID.NS')
start = dt.datetime(2025, 1, 1)
end = dt.datetime(2025, 7, 1)

# --- window size phù hợp input_shape=(10,1)
window_size = 10

if st.button("Dự đoán"):
    # --- Lấy dữ liệu cổ phiếu
    df = yf.download(stock, start=start, end=end)
    if df.empty:
        st.error("Không tìm thấy dữ liệu cho mã này!")
    else:
        data_desc = df.describe()
        # --- Tính toán EMA
        ema20 = df.Close.ewm(span=20, adjust=False).mean()
        ema50 = df.Close.ewm(span=50, adjust=False).mean()
        ema100 = df.Close.ewm(span=100, adjust=False).mean()
        ema200 = df.Close.ewm(span=200, adjust=False).mean()
        # --- Chia tập train/test
        data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
        data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_training_array = scaler.fit_transform(data_training)
        # Chuẩn bị dữ liệu cho dự đoán
        past_days = data_training.tail(window_size)
        final_df = pd.concat([past_days, data_testing], ignore_index=True)
        input_data = scaler.fit_transform(final_df)
        x_test, y_test = [], []
        for i in range(window_size, input_data.shape[0]):
            x_test.append(input_data[i - window_size:i])
            y_test.append(input_data[i, 0])
        x_test, y_test = np.array(x_test), np.array(y_test)
        # --- Load model và dự đoán
        model = load_model('weight/stacked_lstm/model_AAPL.h5')
        y_predicted = model.predict(x_test)
        # Inverse scaling
        scaler_ = scaler.scale_
        scale_factor = 1 / scaler_[0]
        y_predicted = y_predicted * scale_factor
        y_test = y_test * scale_factor
        # --- Vẽ EMA 20 & 50
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(df.Close, 'y', label='Closing Price')
        ax1.plot(ema20, 'g', label='EMA 20')
        ax1.plot(ema50, 'r', label='EMA 50')
        ax1.set_title("Closing Price vs Time (20 & 50 Days EMA)")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Price")
        ax1.legend()
        ema_chart_path = "static/ema_20_50_streamlit.png"
        fig1.savefig(ema_chart_path)
        st.pyplot(fig1)
        st.download_button(
            label="Tải xuống biểu đồ EMA 20 & 50",
            data=open(ema_chart_path, "rb").read(),
            file_name="ema_20_50.png"
        )
        plt.close(fig1)
        # --- Vẽ EMA 100 & 200
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(df.Close, 'y', label='Closing Price')
        ax2.plot(ema100, 'g', label='EMA 100')
        ax2.plot(ema200, 'r', label='EMA 200')
        ax2.set_title("Closing Price vs Time (100 & 200 Days EMA)")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Price")
        ax2.legend()
        ema_chart_path_100_200 = "static/ema_100_200_streamlit.png"
        fig2.savefig(ema_chart_path_100_200)
        st.pyplot(fig2)
        st.download_button(
            label="Tải xuống biểu đồ EMA 100 & 200",
            data=open(ema_chart_path_100_200, "rb").read(),
            file_name="ema_100_200.png"
        )
        plt.close(fig2)
        # --- Vẽ Predictions
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        ax3.plot(y_test, 'g', label="Original Price", linewidth=1)
        ax3.plot(y_predicted, 'r', label="Predicted Price", linewidth=1)
        ax3.set_title("Prediction vs Original Trend")
        ax3.set_xlabel("Time")
        ax3.set_ylabel("Price")
        ax3.legend()
        prediction_chart_path = "static/stock_prediction_streamlit.png"
        fig3.savefig(prediction_chart_path)
        st.pyplot(fig3)
        st.download_button(
            label="Tải xuống biểu đồ Dự đoán vs Thực tế",
            data=open(prediction_chart_path, "rb").read(),
            file_name="prediction_vs_actual.png"
        )
        plt.close(fig3)
        # --- Thống kê dữ liệu
        st.subheader("Thống kê dữ liệu")
        st.dataframe(data_desc)
        # --- Lưu data CSV và cho tải về
        csv_file_path = f"static/{stock}_dataset_streamlit.csv"
        df.to_csv(csv_file_path)
        with open(csv_file_path, "rb") as f:
            st.download_button(
                label="Tải về file dữ liệu gốc (CSV)",
                data=f, file_name=f"{stock}_dataset.csv", mime="text/csv"
            )