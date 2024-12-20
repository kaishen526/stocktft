import datetime
import warnings
warnings.filterwarnings("ignore")

import PySimpleGUI as sg
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

import yfinance as yf
import pandas as pd
import numpy as np
import torch
import requests

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.loggers import TensorBoardLogger

ALPHA_VANTAGE_API_KEY = "BIUXFLM54A6Q9MWW"
WSB_SENTIMENT_FILE = r"C:\Users\kaish\Desktop\ThetaTerminal\wsb20211201.xlsx"


def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_widget = figure_canvas_agg.get_tk_widget()
    figure_widget.pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

def load_tickers_from_excel(file_path):
    df = pd.read_excel(file_path)
    tickers = df['Symbol'].dropna().tolist()
    tickers = [t.strip() for t in tickers if t.strip()]
    return tickers

def add_technical_indicators(df):
    df = df.sort_values("Date")
    df['LogRet'] = df['Close'].diff().fillna(0)
    df['RealizedVol'] = df['LogRet'].rolling(window=10).std().fillna(0.0).astype('float32')
    df['day_of_week'] = pd.to_datetime(df['Date']).dt.dayofweek.astype(str)
    return df

def fetch_alphavantage_sentiment_for_ticker(ticker, start_date, end_date):
    def to_av_format(dt):
        return dt.strftime("%Y%m%dT%H%M")

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    time_from_str = to_av_format(start_dt)
    time_to_str = to_av_format(end_dt)

    url = f"https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "time_from": time_from_str,
        "time_to": time_to_str,
        "limit": "1000",
        "apikey": ALPHA_VANTAGE_API_KEY
    }

    r = requests.get(url, params=params)
    data = r.json()

    if "feed" not in data:
        return pd.DataFrame(columns=["Date", "sentiment"])

    daily_scores = {}
    for article in data["feed"]:
        time_published = article.get("time_published")
        sentiment_score = article.get("overall_sentiment_score", 0.0)
        date_str = time_published[:8]
        date_dt = pd.to_datetime(date_str, format="%Y%m%d")

        if start_dt <= date_dt <= end_dt:
            daily_scores.setdefault(date_dt.date(), []).append(float(sentiment_score))

    sentiment_data = []
    for d, scores in daily_scores.items():
        sentiment_data.append((d, np.mean(scores)))

    sentiment_df = pd.DataFrame(sentiment_data, columns=['Date', 'sentiment'])
    return sentiment_df

def merge_sentiment_data(df_reset, ticker, start_date, end_date):
    sentiment_df = fetch_alphavantage_sentiment_for_ticker(ticker, start_date, end_date)
    df_reset['Date'] = pd.to_datetime(df_reset['Date'])
    if not sentiment_df.empty:
        sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])

    df_reset = df_reset.merge(sentiment_df, on='Date', how='left')
    df_reset['sentiment'] = df_reset['sentiment'].fillna(0.0).astype('float32')
    return df_reset

def merge_wsb_sentiment_data(df_reset, ticker):
    wsb_df = pd.read_excel(WSB_SENTIMENT_FILE)
    wsb_df['Datetime'] = pd.to_datetime(wsb_df['Datetime'])
    wsb_df = wsb_df.rename(columns={'Ticker': 'stock_id', 'Sentiment': 'wsb_sentiment', 'Datetime': 'Date'})
    wsb_df = wsb_df[wsb_df['stock_id'].str.upper() == ticker.upper()]

    df_reset = df_reset.merge(wsb_df[['Date','stock_id','wsb_sentiment']], on=['Date','stock_id'], how='left')
    df_reset['wsb_sentiment'] = df_reset['wsb_sentiment'].fillna(0.0).astype('float32')
    return df_reset

def load_data_for_single_ticker(ticker, start_date, end_date, prediction_end):
    if not ticker:
        raise ValueError("Empty ticker symbol encountered.")

    df = yf.download(ticker, start=start_date, end=prediction_end)
    if df.empty:
        raise ValueError(f"No data for {ticker}, please check the ticker or date range.")

    df_prices_original = df[['Open', 'High', 'Low', 'Close']].ffill()
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna().copy()
    all_dates = pd.date_range(start=df.index.min(), end=prediction_end, freq='D')
    df = df.reindex(all_dates, method='ffill').dropna()
    df.index.name = 'Date'

    close_raw = df['Close'].copy()

    for col in ['Open', 'High', 'Low', 'Close']:
        df[col] = np.log(df[col])
    df['Volume'] = np.log(df['Volume'] + 1)

    obv = [0.0]
    for i in range(1, len(df)):
        if close_raw.iloc[i] > close_raw.iloc[i - 1]:
            obv.append(obv[-1] + np.exp(df['Volume'].iloc[i] - 1))
        elif close_raw.iloc[i] < close_raw.iloc[i - 1]:
            obv.append(obv[-1] - np.exp(df['Volume'].iloc[i] - 1))
        else:
            obv.append(obv[-1])
    df['OBV'] = obv

    ticker_info = yf.Ticker(ticker).info
    df['DividendRate'] = ticker_info.get('dividendRate', 0.0)
    df['MarketCap'] = ticker_info.get('marketCap', 0.0)
    df['PERatio'] = ticker_info.get('trailingPE', 0.0)
    df['PriceToBook'] = ticker_info.get('priceToBook', 0.0)
    df['DebtToEquity'] = ticker_info.get('debtToEquity', 0.0)
    df['ShortRatio'] = ticker_info.get('shortRatio', 0.0)
    df['AnalystTargetPrice'] = ticker_info.get('targetMeanPrice', 0.0)

    df['stock_id'] = ticker
    min_date_ = df.index.min()
    df['time_idx'] = (df.index - min_date_).days

    df_reset = df.reset_index()
    df_reset = add_technical_indicators(df_reset)
    df_reset['Date'] = pd.to_datetime(df_reset['Date'])

    df_reset = merge_sentiment_data(df_reset, ticker, start_date, prediction_end)
    df_reset = merge_wsb_sentiment_data(df_reset, ticker)

    fundamentals = ['DividendRate', 'MarketCap', 'PERatio', 'PriceToBook', 'DebtToEquity', 'ShortRatio','AnalystTargetPrice', 'RealizedVol']
    for col in fundamentals:
        df_reset[col] = df_reset[col].astype('float32')

    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'OBV', 'RealizedVol', 'sentiment', 'wsb_sentiment']
    for col in numeric_cols:
        df_reset[col] = df_reset[col].astype('float32', errors='ignore')

    return df_reset, df_prices_original

def combine_all_tickers(tickers, start_date, end_date, prediction_end):
    all_data = []
    first_ticker_prices = None

    for i, ticker in enumerate(tickers):
        try:
            df_reset, df_prices_original = load_data_for_single_ticker(ticker, start_date, end_date, prediction_end)
            all_data.append(df_reset)
            if first_ticker_prices is None:
                first_ticker_prices = df_prices_original
        except ValueError as e:
            print(f"Error loading data for {ticker}: {e}, skipping.")

    if not all_data:
        raise ValueError("No valid data for any given ticker.")

    df_all = pd.concat(all_data, ignore_index=True)
    return df_all, first_ticker_prices

def create_datasets(df, prediction_start, prediction_end):
    # Using a fixed 30-day horizon
    fixed_prediction_length = 30

    prediction_start = pd.Timestamp(prediction_start)
    prediction_end = pd.Timestamp(prediction_end)

    final_time_idx = df['time_idx'].max()
    training_cutoff = final_time_idx * 0.7

    if training_cutoff < df['time_idx'].min():
        raise ValueError("Not enough historical data for the given prediction horizon.")

    time_varying_unknown_reals = ['High', 'Low', 'Close', 'Volume', 'OBV', 'RealizedVol', 'sentiment', 'wsb_sentiment']
    static_reals = ['DividendRate', 'MarketCap', 'PERatio', 'PriceToBook', 'DebtToEquity', 'ShortRatio','AnalystTargetPrice']
    time_varying_known_categoricals = ['day_of_week']

    target = "Open"

    training = TimeSeriesDataSet(
        df[lambda x: x.time_idx <= training_cutoff],
        time_idx='time_idx',
        target=target,
        group_ids=["stock_id"],
        static_categoricals=["stock_id"],
        static_reals=static_reals,
        time_varying_unknown_reals=time_varying_unknown_reals,
        time_varying_known_categoricals=time_varying_known_categoricals,
        min_encoder_length=30,
        max_encoder_length=60,
        min_prediction_length=30,
        max_prediction_length=30,
        add_relative_time_idx=True,
        add_encoder_length=True,
        allow_missing_timesteps=True
    )

    validation = TimeSeriesDataSet.from_dataset(training, df[df.time_idx > training_cutoff], predict=False)
    return training, validation, 30  # fixed 30-day horizon

def train_tft(training, validation, checkpoint_path=None):
    train_dataloader = training.to_dataloader(train=True, batch_size=128, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=160, num_workers=0)

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
    lr_logger = LearningRateMonitor()
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='best_model',
        save_top_k=1,
        mode='min'
    )
    logger = TensorBoardLogger("lightning_logs", name="tft_model")
    model_summary = ModelSummary(max_depth=1)

    trainer = pl.Trainer(
        max_epochs=5,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        gradient_clip_val=0.1,
        callbacks=[lr_logger, early_stop_callback, model_summary, checkpoint_callback],
        logger=logger,
    )

    if checkpoint_path is not None:
        print(f"Continuing training from checkpoint: {checkpoint_path}")
        tft = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path)
    else:
        stock_id_cardinality = len(training.categorical_encoders["stock_id"].classes_)
        day_of_week_cardinality = len(training.categorical_encoders["day_of_week"].classes_)

        embedding_sizes = {
            "stock_id": (stock_id_cardinality, 64),
            "day_of_week": (day_of_week_cardinality, 8)
        }

        tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=0.001,
            hidden_size=256,
            attention_head_size=16,
            dropout=0.05,
            hidden_continuous_size=16,
            output_size=5,
            loss=QuantileLoss(quantiles=[0.05,0.25,0.5,0.75,0.95]),
            reduce_on_plateau_patience=4,
            lstm_layers=3,
            embedding_sizes=embedding_sizes
        )

    trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    return trainer, tft

def iterative_prediction(model, training, df_all, pstart, pend):
    current_data = df_all.copy()
    current_data = current_data.sort_values("Date")
    current_date = pd.Timestamp(pstart)
    end_date = pd.Timestamp(pend)

    stock_id = current_data['stock_id'].iloc[0]

    while current_date <= end_date:
        cutoff_date = current_date - pd.Timedelta(days=1)
        last_60 = current_data.loc[current_data['Date'] <= cutoff_date].tail(60)

        if len(last_60) < 60:
            print("Not enough historical data to predict further.")
            break

        # Add 30 future placeholder rows
        future_dates = pd.date_range(start=cutoff_date + pd.Timedelta(days=1), periods=30, freq='D')
        last_time_idx = last_60['time_idx'].max()
        placeholders = []
        for i, fd in enumerate(future_dates):
            new_row = last_60.iloc[-1].copy()
            new_row['Date'] = fd
            new_row['time_idx'] = last_time_idx + i + 1
            # Use a finite placeholder instead of NaN for 'Open'
            new_row['Open'] = 0.0  # log(1)=0 stable placeholder
            placeholders.append(new_row)
        placeholders = pd.DataFrame(placeholders)
        extended_data = pd.concat([last_60, placeholders], ignore_index=True)

        # Create dataset from the training dataset
        temp_dataset = TimeSeriesDataSet.from_dataset(
            training,
            extended_data,
            predict=True,
            stop_randomization=True
        )

        temp_dataloader = temp_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)
        pred, x = model.predict(temp_dataloader, return_x=True)
        pred_np = pred.detach().cpu().numpy()

        pred_median = pred_np[0, :]
        pred_prices = np.exp(pred_median)  # from log-space to real price

        # Update predicted 30 days in extended_data with predicted values
        for i, fdate in enumerate(future_dates):
            idx = extended_data.index[extended_data['Date'] == fdate]
            if len(idx) > 0:
                extended_data.loc[idx, 'Open'] = np.log(pred_prices[i])
                extended_data.loc[idx, 'High'] = np.log(pred_prices[i])
                extended_data.loc[idx, 'Low'] = np.log(pred_prices[i])
                extended_data.loc[idx, 'Close'] = np.log(pred_prices[i])

        current_data = pd.concat([current_data, extended_data.iloc[-30:]], ignore_index=True)

        current_date += pd.Timedelta(days=30)

    return current_data


if __name__ == "__main__":
    excel_file = "C:/Users/kaish/Desktop/ThetaTerminal/test1.xlsx"
    start_date = "2018-01-02"
    end_date = "2023-12-31"
    pstart = "2024-01-02"
    pend = "2026-01-01"
    tickers = load_tickers_from_excel(excel_file)
    print("Loaded tickers:", tickers)

    df_all, first_ticker_prices = combine_all_tickers(tickers, start_date, end_date, pend)
    print("Combined dataset shape:", df_all.shape)
    print("Sample of combined dataset:", df_all.head())

    training, validation, user_prediction_length = create_datasets(df_all, pstart, pend)
    print("Number of validation samples:", len(validation))
    print("Number of validation batches:", len(validation.to_dataloader(train=False, batch_size=160)))
    print(training)

    best_tft = TemporalFusionTransformer.load_from_checkpoint(
        r"C:\Users\kaish\Stock-TFT-main\Code\20241218\checkpoints\best_model-v15.ckpt")

    predicted_data = iterative_prediction(best_tft, training, df_all, pstart, pend)

    # Convert log back to price for both historical and predicted data
    df_all['Open_price'] = np.exp(df_all['Open'])
    predicted_data['Open_price'] = np.exp(predicted_data['Open'])

    # Plot both historical and predicted data over the entire time frame
    fig, ax = plt.subplots(figsize=(10, 5))
    # Historical data
    ax.plot(df_all['Date'], df_all['Open_price'], label='Historical Data', color='blue')
    # Iterative forecast
    ax.plot(predicted_data['Date'], predicted_data['Open_price'], label='Iterative 30-Day Forecast', color='red', linestyle='--')

    ax.set_xlabel('Date')
    ax.set_ylabel('Open Price')
    ax.set_title('Historical Data vs Iterative 30-Day Forecast')
    ax.legend()
    plt.tight_layout()
    plt.show()
