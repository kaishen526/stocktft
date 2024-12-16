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

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.loggers import TensorBoardLogger


def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_widget = figure_canvas_agg.get_tk_widget()
    figure_widget.pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

def load_tickers_from_excel(file_path):
    df = pd.read_excel(file_path)
    tickers = df['Symbol'].dropna().unique().tolist()
    return tickers

def load_all_data_for_tickers(tickers, start_date, end_date, prediction_end):
    all_data = []
    for ticker in tickers:
        df = yf.download(ticker, start=start_date, end=prediction_end)
        if df.empty:
            print(f"No data for {ticker}, skipping.")
            continue
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna().copy()
        all_dates = pd.date_range(start=df.index.min(), end=prediction_end, freq='D')
        df = df.reindex(all_dates, method='ffill').dropna()
        df.index.name = 'Date'

        for col in ['Open', 'High', 'Low', 'Close']:
            df[col] = np.log(df[col])

        df['Volume'] = np.log(df['Volume'] + 1)

        df['stock_id'] = ticker
        min_date_ = df.index.min()
        df['time_idx'] = (df.index - min_date_).days

        all_data.append(df.reset_index())

    if not all_data:
        raise ValueError("No valid data returned for the given tickers.")
    df_all = pd.concat(all_data, ignore_index=True)
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df_all[col] = df_all[col].astype('float32')
    return df_all

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

    features = ['High', 'Low', 'Close', 'Volume']
    target = "Open"

    print("Final time_idx:", final_time_idx)
    print("Training cutoff (time_idx):", training_cutoff)
    print("User prediction length:", user_prediction_length)

    training = TimeSeriesDataSet(
        df[lambda x: x.time_idx <= training_cutoff],
        time_idx='time_idx',
        target=target,
        group_ids=["stock_id"],
        static_categoricals=["stock_id"],
        time_varying_unknown_reals=features,
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

def train_tft(training, validation, checkpoint_path=None):
    train_dataloader = training.to_dataloader(train=True, batch_size=16, num_workers=0)
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
        max_epochs=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        gradient_clip_val=0.1,
        callbacks=[lr_logger, early_stop_callback, model_summary, checkpoint_callback],
        logger=logger,
    )

    if checkpoint_path is not None:
        print(f"Loading model from checkpoint: {checkpoint_path}")
        tft = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path)
        return trainer, tft
    else:
        tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=0.05,
            hidden_size=32,
            attention_head_size=4,
            dropout=0.1,
            hidden_continuous_size=4,
            output_size=3,
            loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
            reduce_on_plateau_patience=4,
        )

        trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        return trainer, tft

def plot_predictions(median_pred, df_prices_original, prediction_start, prediction_end):
    prediction_start = pd.Timestamp(prediction_start)
    prediction_end = pd.Timestamp(prediction_end)
    user_prediction_length = (prediction_end - prediction_start).days
    train_end_ts = prediction_start - pd.Timedelta(days=1)
    df_ref = df_prices_original.loc[df_prices_original.index <= train_end_ts]
    if df_ref.empty:
        df_ref = df_prices_original.iloc[:60]

    predicted_prices = np.exp(median_pred)
    future_dates = pd.date_range(prediction_start, prediction_end, freq='D')[:len(median_pred)]
    df_future = df_prices_original.loc[(df_prices_original.index >= prediction_start) & (df_prices_original.index <= prediction_end)]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_ref.index, df_ref['Open'], label='Reference Data (Actual)', color='blue')
    if len(predicted_prices) > 0 and len(future_dates) > 0:
        ax.plot(future_dates, predicted_prices, label='Predicted Data (Median)', color='red')
    else:
        print("No predicted prices or empty future dates.")

    if len(df_future) > 0:
        ax.plot(df_future.index, df_future['Open'], label='Actual Future Data', color='green', linestyle='--')

    ax.set_xlabel('Date')
    ax.set_ylabel('Open Price')
    ax.set_title('Stock Price Predictions')
    ax.legend()
    plt.tight_layout()
    return fig

def evaluate_and_plot_multiple(trainer, tft, validation, df_prices_original, prediction_start, prediction_end, checkpoint_path=None):
    # Similar logic to original evaluate_and_plot,
    # but now we handle multiple samples and create a figure per sample.
    val_dataloader = validation.to_dataloader(train=False, batch_size=160, num_workers=0)

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

    predictions, x = best_tft.predict(val_dataloader, return_x=True)
    print("DEBUG: predictions shape:", predictions.shape)

    if predictions.shape[0] == 0:
        print("No predictions returned.")
        return []

    preds_cpu = predictions.detach().cpu().numpy()
    print("DEBUG: predictions after detach/cpu:", preds_cpu.shape)

    # We'll create a figure for each sample (N dimension)
    figs = []
    N = preds_cpu.shape[0]
    for i in range(N):
        # If (N,T,Q) -> median_pred = preds_cpu[i,:,1]
        # If (N,T) -> median_pred = preds_cpu[i,:]
        if preds_cpu.ndim == 3:
            median_pred = preds_cpu[i, :, 1]
        else:
            median_pred = preds_cpu[i, :]

        fig = plot_predictions(median_pred, df_prices_original, prediction_start, prediction_end)
        figs.append((f"Sample_{i}", fig))

    return figs

###############################################################
# GUI PART
###############################################################
sg.theme("LightBlue")

layout = [
    [sg.Text("Stock Prediction Input")],
    [sg.Text("Upload Excel File with Tickers:"), sg.Input(key='-EXCEL-', enable_events=True), sg.FileBrowse(file_types=(("Excel Files", "*.xlsx"),))],
    [sg.Text("Select Start Date for Reference Data (YYYY-MM-DD):"), sg.Input(key='-START-', default_text='2015-01-02')],
    [sg.Text("Select End Date for Reference Data (YYYY-MM-DD):"), sg.Input(key='-END-', default_text='2019-12-31')],
    [sg.Text("Select Start Date for Prediction (YYYY-MM-DD):"), sg.Input(key='-PSTART-', default_text='2020-01-02')],
    [sg.Text("Select End Date for Prediction (YYYY-MM-DD):"), sg.Input(key='-PEND-', default_text='2020-03-31')],
    [sg.Text("Optional: Load Existing Checkpoint:"), sg.Input(key='-CKPT-', enable_events=True), sg.FileBrowse(file_types=(("Checkpoint Files", "*.ckpt"),))],
    [sg.Button("Run Prediction")],
    [sg.Button("Exit")]
]

window = sg.Window("TFT Stock Prediction", layout, finalize=True)

while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    if event == "Run Prediction":
        try:
            excel_file = values['-EXCEL-']
            start_date = values['-START-']
            end_date = values['-END-']
            pstart = values['-PSTART-']
            pend = values['-PEND-']
            checkpoint_path = values['-CKPT-'] if values['-CKPT-'] else None

            if excel_file:
                tickers = load_tickers_from_excel(excel_file)
                df = load_all_data_for_tickers(tickers, start_date, end_date, pend)
                first_ticker = tickers[0]
                df_prices_original = yf.download(first_ticker, start=start_date, end=pend).ffill()[['Open','High','Low','Close']]
            else:
                ticker = sg.popup_get_text('Enter a single ticker if no Excel file provided:', default_text='AAPL')
                if not ticker:
                    sg.popup("No ticker provided, cannot proceed.")
                    continue
                tickers = [ticker]
                df = load_all_data_for_tickers(tickers, start_date, end_date, pend)
                df_prices_original = yf.download(ticker, start=start_date, end=pend).ffill()[['Open','High','Low','Close']]

            training, validation, user_prediction_length = create_datasets(df, pstart, pend)
            print("Number of validation samples:", len(validation))
            print("Number of validation batches:", len(validation.to_dataloader(train=False, batch_size=160)))

            trainer, tft_model = train_tft(training, validation, checkpoint_path=checkpoint_path)
            figs = evaluate_and_plot_multiple(trainer, tft_model, validation, df_prices_original, pstart, pend, checkpoint_path=checkpoint_path)

            if not figs:
                sg.popup("No figure was returned. Check console logs and date ranges.")
            else:
                # Create a new window with tabs for each sample
                # Replace the tab creation portion with the following code:
                tabs = []
                for sample_name, fig in figs:
                    canvas_key = f"-CANVAS_{sample_name}-"
                    # Enable scrolling in both directions by setting vertical_scroll_only=False
                    tab_layout = [
                        [sg.Column(
                            [[sg.Canvas(key=canvas_key, size=(1200, 800))]],
                            size=(1100, 580),
                            scrollable=True,
                            vertical_scroll_only=False
                        )]
                    ]
                    tabs.append(sg.Tab(sample_name, tab_layout))

                tab_group_layout = [[sg.TabGroup([tabs], key="-TABGROUP-", expand_x=True, expand_y=True)]]
                result_window = sg.Window("Prediction Results", tab_group_layout, resizable=True, finalize=True)

                # Draw each figure on its respective canvas
                for sample_name, fig in figs:
                    canvas_key = f"-CANVAS_{sample_name}-"
                    draw_figure(result_window[canvas_key].TKCanvas, fig)

                if checkpoint_path is None:
                    sg.popup("Training completed. Model checkpoint saved in 'checkpoints' directory. You can reuse this checkpoint for future predictions.")
                else:
                    sg.popup("Prediction completed using the loaded model checkpoint.")

                # Event loop for result window
                while True:
                    ev, vals = result_window.read()
                    if ev == sg.WIN_CLOSED:
                        break
                result_window.close()

        except Exception as e:
            sg.popup("Error", str(e))

window.close()
