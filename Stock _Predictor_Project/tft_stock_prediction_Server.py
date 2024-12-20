import datetime
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')  # For backend suitable for servers without display
import matplotlib.pyplot as plt

import yfinance as yf
import pandas as pd
import numpy as np
import torch
import requests
import pytorch_lightning as pl

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks import ModelSummary, Callback
from pytorch_lightning.loggers import TensorBoardLogger

ALPHA_VANTAGE_API_KEY = "BIUXFLM54A6Q9MWW"  # Replace with your key

WSB_SENTIMENT_FILE = "/home/user/wsb20211201.xlsx"

START_DATE = "2018-01-04"
END_DATE = "2023-12-31"
PSTART = "2024-01-02"
PEND = "2025-01-01"
CHECKPOINT = None  # Set this to a path of a .ckpt file if you want to resume training

# Multiple tickers
tickers = [
    'XOM', 'CVX', 'SHEL', 'COP', 'TTE', 'ENB', 'PBR', 'BP', 'EOG', 'EPD', 'CNQ', 'WMB', 'ET', 'EQNR', 'OKE',
    'KMI', 'SLB', 'PSX', 'MPLX', 'TRP', 'LNG', 'FANG', 'MPC', 'SU', 'OXY', 'E', 'HES', 'BKR', 'VLO', 'TRGP', 'IMO',
    'WDS', 'CQP', 'CVE', 'TPL', 'EQT', 'HAL', 'CCJ', 'EXE', 'DVN', 'PBA', 'TS', 'CTRA', 'YPF', 'EC', 'WES', 'PAA',
]

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
    if WSB_SENTIMENT_FILE is None:
        df_reset['wsb_sentiment'] = 0.0
        return df_reset

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

    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(ticker, level='Ticker', axis=1)

    if df.empty:
        raise ValueError(f"No data for {ticker}, please check the ticker or date range.")

    df_prices_original = df[['Open', 'High', 'Low', 'Close']].ffill()
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna().copy()
    all_dates = pd.date_range(start=df.index.min(), end=prediction_end, freq='D')
    df = df.reindex(all_dates, method='ffill').dropna()
    df.index.name = 'Date'

    close_raw = df['Close'].copy()

    # Log transform
    for col in ['Open', 'High', 'Low', 'Close']:
        df[col] = np.log(df[col])
    df['Volume'] = np.log(df['Volume'] + 1)

    # OBV calculation
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

    # Merge Alpha Vantage sentiment
    df_reset = merge_sentiment_data(df_reset, ticker, start_date, prediction_end)

    # Merge WSB sentiment data
    df_reset = merge_wsb_sentiment_data(df_reset, ticker)

    print("After merging sentiment data for:", ticker)
    print(df_reset[['Date', 'sentiment', 'wsb_sentiment']].head(30))
    print("Sentiment stats:")
    print(df_reset[['sentiment','wsb_sentiment']].describe())

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
    prediction_start = pd.Timestamp(prediction_start)
    prediction_end = pd.Timestamp(prediction_end)

    user_prediction_length = (prediction_end - prediction_start).days
    if user_prediction_length < 1:
        raise ValueError("Prediction length must be at least 1 day.")

    final_time_idx = df['time_idx'].max()
    training_cutoff = final_time_idx - user_prediction_length

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
        min_prediction_length=user_prediction_length,
        max_prediction_length=user_prediction_length,
        add_relative_time_idx=True,
        add_encoder_length=True,
        allow_missing_timesteps=True
    )

    validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)
    return training, validation, user_prediction_length

# Custom callback to store training losses after each epoch
class StoreLossesCallback(Callback):
    def __init__(self):
        super().__init__()
        self.train_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        # 'train_loss' is logged automatically by PyTorch Lightning
        loss = trainer.callback_metrics.get("train_loss", None)
        if loss is not None:
            self.train_losses.append(loss.item())

def plot_train_loss(train_losses):
    plt.figure(figsize=(8,4))
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.tight_layout()
    plt.savefig('train_loss_over_epochs.png')
    print("Train loss plot saved to train_loss_over_epochs.png")

def train_tft(training, validation, checkpoint_path=None):
    train_dataloader = training.to_dataloader(train=True, batch_size=16, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=160, num_workers=0)

    # Add our loss storage callback
    loss_callback = StoreLossesCallback()

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
    lr_logger = LearningRateMonitor()

    # This callback saves the best model (.ckpt file)
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
        max_epochs=100,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        gradient_clip_val=0.1,
        callbacks=[lr_logger, early_stop_callback, model_summary, checkpoint_callback, loss_callback],
        logger=logger,
    )

    if checkpoint_path is not None:
        print(f"Continuing training from checkpoint: {checkpoint_path}")
        tft = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path)
    else:
        tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=0.001,
            hidden_size=256,
            attention_head_size=16,
            dropout=0.05,
            hidden_continuous_size=16,
            output_size=5,
            loss=QuantileLoss(quantiles=[0.05, 0.25, 0.5, 0.75, 0.95]),
            reduce_on_plateau_patience=4,
            lstm_layers=3
        )

    trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # After training, plot the stored training losses
    if len(loss_callback.train_losses) > 0:
        plot_train_loss(loss_callback.train_losses)

    print("Best model saved at:", trainer.checkpoint_callback.best_model_path)
    return trainer, tft

def plot_predictions(preds_cpu, df_prices_original, prediction_start, prediction_end):
    median_pred = preds_cpu[:, 2]
    lower_pred = preds_cpu[:, 0]
    upper_pred = preds_cpu[:, 4]

    prediction_start = pd.Timestamp(prediction_start)
    prediction_end = pd.Timestamp(prediction_end)
    user_prediction_length = (prediction_end - prediction_start).days
    train_end_ts = prediction_start - pd.Timedelta(days=1)
    df_ref = df_prices_original.loc[df_prices_original.index <= train_end_ts]
    if df_ref.empty:
        df_ref = df_prices_original.iloc[:60]

    median_prices = np.exp(median_pred)
    lower_prices = np.exp(lower_pred)
    upper_prices = np.exp(upper_pred)

    future_dates = pd.date_range(prediction_start, prediction_end, freq='D')[:len(median_pred)]
    df_future = df_prices_original.loc[
        (df_prices_original.index >= prediction_start) & (df_prices_original.index <= prediction_end)]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_ref.index, df_ref['Open'], label='Reference Data (Actual)', color='blue')
    if len(median_prices) > 0 and len(future_dates) > 0:
        ax.plot(future_dates, median_prices, label='Predicted Data (Median)', color='red')
        ax.fill_between(future_dates, lower_prices, upper_prices, color='gray', alpha=0.3, label='Uncertainty (5%-95%)')
    else:
        print("No predicted prices or empty future dates.")

    if len(df_future) > 0:
        ax.plot(df_future.index, df_future['Open'], label='Actual Future Data', color='green', linestyle='--')

    ax.set_xlabel('Date')
    ax.set_ylabel('Open Price')
    ax.set_title('Stock Predictions with Alpha Vantage & WSB Sentiment (Unified Dataset)')
    ax.legend()
    plt.tight_layout()
    return fig

def evaluate_and_plot(trainer, tft, validation, df_prices_original, prediction_start, prediction_end, checkpoint_path=None):
    val_dataloader = validation.to_dataloader(train=False, batch_size=160, num_workers=0)

    # If you have a previously trained model in a .ckpt file, specify checkpoint_path here:
    # checkpoint_path = "checkpoints/best_model.ckpt"
    if checkpoint_path is not None:
        print(f"Loading model from checkpoint: {checkpoint_path}")
        best_tft = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path)
    else:
        best_model_path = trainer.checkpoint_callback.best_model_path
        if not best_model_path:
            print("No best model checkpoint found. Using current model.")
            best_tft = tft
        else:
            best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

    predictions = best_tft.predict(val_dataloader, return_x=True)

    preds_cpu = predictions.output.detach().cpu().numpy()
    print("DEBUG: predictions after detach/cpu:", preds_cpu.shape)

    if preds_cpu.ndim == 2:
        preds_cpu = np.expand_dims(preds_cpu, axis=-1)
        preds_cpu = np.repeat(preds_cpu, 5, axis=-1)

    if preds_cpu.shape[0] == 0:
        print("No predictions returned.")
        return None

    sample_pred = preds_cpu[0]
    fig = plot_predictions(sample_pred, df_prices_original, prediction_start, prediction_end)
    return fig


if __name__ == "__main__":
    start_date = START_DATE
    end_date = END_DATE
    pstart = PSTART
    pend = PEND
    checkpoint = CHECKPOINT

    print("Loaded tickers:", tickers)

    df_all, first_ticker_prices = combine_all_tickers(tickers, start_date, end_date, pend)
    print("Combined dataset shape:", df_all.shape)
    print("Sample of combined dataset:", df_all.head())

    training, validation, user_prediction_length = create_datasets(df_all, pstart, pend)
    print("Number of validation samples:", len(validation))
    print("Number of validation batches:", len(validation.to_dataloader(train=False, batch_size=160)))

    # Train the TFT model
    trainer, tft_model = train_tft(training, validation, checkpoint_path=checkpoint)

    # Evaluate and plot predictions
    fig = evaluate_and_plot(trainer, tft_model, validation, first_ticker_prices, pstart, pend)
    if fig is not None:
        fig.savefig("prediction_results.png")
        print("Prediction figure saved to prediction_results.png")
    else:
        print("No figure generated.")
