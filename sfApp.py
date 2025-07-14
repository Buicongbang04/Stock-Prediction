import joblib
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import datetime as dt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from io import BytesIO
import torch
import os
from typing import Dict
from models.Transformers.stockTransformer import Stockformer  

# ------------------------------
# Page & global configuration
# ------------------------------
st.set_page_config(page_title="Stockformer Stock Predictor", layout="wide")
plt_style = "plotly_dark"  # Consistent theme for all figures
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW_SIZE = 24
WEIGHT_DIR = "weight/stockformer"
SUPPORTED_TICKERS = [
    "AMZN", "NVDA", "AAPL", "BIDU", "GOOG", "INTC", "MSFT", "NFLX", "TCEHY", "TSLA",
]

@st.cache_resource(show_spinner=False)
def load_trained_model(ticker: str):
    weight_path = os.path.join(WEIGHT_DIR, f"{ticker}_best.pt")
    if not os.path.isfile(weight_path):
        raise FileNotFoundError(
            f"Không tìm thấy file weight {weight_path}. Hãy chắc chắn đã train và lưu mô hình."
        )
    # Chỉnh các thông số phải đúng với lúc bạn train
    model = Stockformer(
        input_dim=1,
        embed_dim=128,  # Giá trị này bạn đặt theo lúc train
        num_heads=8,
        num_layers=2,
        out_dim=1,
    ).to(DEVICE)
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    model.eval()
    return model

@st.cache_resource(show_spinner=False)
def load_scaler(ticker: str):
    weight_path = os.path.join(WEIGHT_DIR, f"scaler_{ticker}.pkl")
    if not os.path.isfile(weight_path):
        raise FileNotFoundError(
            f"Không tìm thấy file scaler {weight_path}. Hãy chắc chắn đã train và lưu mô hình."
        )
    return joblib.load(weight_path)

def calculate_ema(df: pd.DataFrame) -> Dict[str, pd.Series]:
    ema20 = df["Close"].ewm(span=20, adjust=False).mean()
    ema50 = df["Close"].ewm(span=50, adjust=False).mean()
    ema100 = df["Close"].ewm(span=100, adjust=False).mean()
    ema200 = df["Close"].ewm(span=200, adjust=False).mean()
    return ema20, ema50, ema100, ema200

def plot_ema(df: pd.DataFrame, 
            ema20: pd.Series=None, 
            ema50: pd.Series=None, 
            ema100: pd.Series=None, 
            ema200: pd.Series=None, 
            title: str="Exponential Moving Averages"):
    if ema20 is not None and ema50 is not None:
        data = pd.DataFrame({
            "Date": df.index,
            "Close": df["Close"].squeeze(),
            "EMA 20": ema20.squeeze(),
            "EMA 50": ema50.squeeze()
        }).reset_index(drop=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], name="Close", mode="lines"))
        fig.add_trace(go.Scatter(x=data["Date"], y=data["EMA 20"], name="EMA 20", mode="lines"))
        fig.add_trace(go.Scatter(x=data["Date"], y=data["EMA 50"], name="EMA 50", mode="lines"))
        fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Price", template=plt_style, height=500)
        return fig
    if ema100 is not None and ema200 is not None:
        data = pd.DataFrame({
            "Date": df.index,
            "Close": df["Close"].squeeze(),
            "EMA 100": ema100.squeeze(),
            "EMA 200": ema200.squeeze()
        }).reset_index(drop=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], name="Close", mode="lines"))
        fig.add_trace(go.Scatter(x=data["Date"], y=data["EMA 100"], name="EMA 100", mode="lines"))
        fig.add_trace(go.Scatter(x=data["Date"], y=data["EMA 200"], name="EMA 200", mode="lines"))
        fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Price", template=plt_style, height=500)
        return fig

def plot_prediction(pred_dates, y_true: np.ndarray, y_pred: np.ndarray):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pred_dates, y=y_true, name="Actual", mode="lines", line=dict(color="cyan")))
    fig.add_trace(go.Scatter(x=pred_dates + pd.DateOffset(hours=1), y=y_pred, name="Predicted", mode="lines", line=dict(color="magenta")))
    fig.update_layout(title="Prediction vs Actual", xaxis_title="Time Step", yaxis_title="Price", template=plt_style, height=500)
    return fig

# ------------------------------
with st.sidebar:
    st.header("Thiết lập")
    ticker = st.selectbox("Chọn mã cổ phiếu", SUPPORTED_TICKERS, index=SUPPORTED_TICKERS.index("BIDU"))
    start_date = st.date_input("Ngày bắt đầu", dt.date(2024, 1, 1))
    end_date = st.date_input("Ngày kết thúc", dt.date(2025, 7, 1))
    st.caption("Dữ liệu được lấy từ Yahoo Finance.")
    run_button = st.button("Dự đoán", type="primary")

st.title("📈 Dự đoán giá cổ phiếu bằng mô hình Stockformer")
st.markdown(
    "Sử dụng mô hình Stockformer huấn luyện riêng cho từng mã cổ phiếu để dự báo xu hướng giá và hiển thị **EMA** 20/50/100/200 ngày."
)

# ------------------------------
if run_button:
    # Load model
    try:
        model = load_trained_model(ticker)
        scaler = load_scaler(ticker)
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    st.info("Đang tải dữ liệu từ Yahoo Finance…")
    raw = yf.download(ticker, start=start_date, end=end_date, interval="1h", progress=False)
    if raw.empty:
        st.error("Không có dữ liệu cho khoảng thời gian đã chọn.")
        st.stop()
    df = raw[["Close"]].copy()
    df.dropna(inplace=True)
    if df.empty:
        st.error("Không có dữ liệu cho khoảng thời gian đã chọn.")
        st.stop()
    df.index = pd.to_datetime(df.index)
    if len(df) < 2:
        st.error("Không đủ dữ liệu để vẽ biểu đồ.")
        st.stop()
    # Tính EMA
    ema20, ema50, ema100, ema200 = calculate_ema(df)
    # Chuẩn bị dữ liệu cho mô hình
    data = df["Close"].values.reshape(-1, 1)
    scaler.fit(data)
    scaled_seq = scaler.transform(data)
    X_test, y_true_scaled = [], []
    for i in range(WINDOW_SIZE, len(scaled_seq)):
        X_test.append(scaled_seq[i - WINDOW_SIZE: i])
        y_true_scaled.append(scaled_seq[i, 0])
    if len(scaled_seq) >= WINDOW_SIZE:  # để tránh index âm
        X_test.append(scaled_seq[-WINDOW_SIZE:])
    X_test = np.asarray(X_test, dtype=np.float32)
    y_true_scaled = np.asarray(y_true_scaled, dtype=np.float32)
    # Chuẩn hóa shape cho Stockformer: (batch, seq_len, input_dim)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    # Dự đoán
    st.info("Đang dự đoán…")
    with torch.no_grad():
        inputs = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        # Output shape (batch, 1)
        outputs = model(inputs).cpu().numpy().flatten()
    y_pred = scaler.inverse_transform(outputs.reshape(-1, 1)).flatten()
    y_true = scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()
    st.success("Hoàn thành dự đoán!")
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("EMA 20 & 50")
        fig_short = plot_ema(df, ema20=ema20, ema50=ema50, title=f"{ticker} – Close vs EMA 20/50")
        st.plotly_chart(fig_short, use_container_width=True)
        st.subheader("EMA 100 & 200")
        fig_long = plot_ema(df, ema100=ema100, ema200=ema200, title=f"{ticker} – Close vs EMA 100/200")
        st.plotly_chart(fig_long, use_container_width=True)
    with col_right:
        pred_dates = df.index[WINDOW_SIZE:]
        st.subheader("Actual vs Predicted")
        fig_pred = plot_prediction(pred_dates, y_true, y_pred)
        st.plotly_chart(fig_pred, use_container_width=True)
        st.subheader("Thống kê mô tả")
        st.dataframe(raw.describe(), use_container_width=True, height=300)
    st.divider()
    st.subheader("Tải xuống dữ liệu lịch sử")
    buff = BytesIO()
    df.to_csv(buff)
    buff.seek(0)
    st.download_button("💾 Tải CSV", data=buff, file_name=f"{ticker}_historical.csv", mime="text/csv")
else:
    st.info("Chọn tham số ở thanh bên, sau đó nhấn **Dự đoán** để bắt đầu.")