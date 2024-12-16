# ---------------------------------------------------#
#
#   File       : PytorchForecasting.py
#   Description: Assembling and training the model using PytorchForecasting and PyTorch Lightning
#
# ----------------------------------------------------#

import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data.encoders import NaNLabelEncoder
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelSummary
import pytorch_lightning as pl
import yfinance as yf

warnings.filterwarnings("ignore")  # to avoid printing out absolute paths


class TFT:
    """
    Temporal Fusion Transformer Setup and Training
    """

    def __init__(self):
        self.prediction_length = None
        self.training = None
        self.validation = None
        self.trainer = None
        self.model = None
        self.batch_size = 16
        self.df_original = None
        self.min_date_ = None
        self.prediction_start = None
        self.prediction_end = None
        self.train_end = None

    def load_data_yfinance(self, ticker, start_date, end_date, prediction_start, prediction_end):
        # Download data from yfinance
        df = yf.download(ticker, start=start_date, end=prediction_end)
        df.reset_index(inplace=True)
        df = df[['Date', 'Open', 'High', 'Low', 'Close']].dropna()

        # Reindex to ensure daily data up to prediction_end
        all_dates = pd.date_range(start=df['Date'].min(), end=prediction_end, freq='D')
        df = df.set_index('Date').reindex(all_dates, method='ffill').dropna()
        df.index.name = 'Date'
        df = df.reset_index()

        # Convert to log returns
        for col in ['Open', 'High', 'Low', 'Close']:
            df[col] = np.log(df[col]).diff()
        df.dropna(inplace=True)

        # Add ID column
        df["Open_Prediction"] = "Open"

        # Create integer time index
        self.min_date_ = df['Date'].min()
        df['Date'] = (df['Date'] - self.min_date_).dt.days

        features = ['High', 'Low', 'Close']
        time_index = "Date"
        target = "Open"

        self.prediction_length = (prediction_end - prediction_start).days
        if self.prediction_length < 1:
            raise ValueError("Prediction length must be at least 1 day. Adjust your prediction_start and prediction_end.")

        training_cutoff = df[time_index].max() - self.prediction_length

        # Ensure training_cutoff leaves room for future predictions
        if training_cutoff < df[time_index].min():
            raise ValueError(
                "Not enough historical data for the given prediction horizon. Reduce prediction length or choose an earlier prediction_end."
            )

        self.training = TimeSeriesDataSet(
            df[lambda x: x[time_index] <= training_cutoff],
            time_idx=time_index,
            target=target,
            categorical_encoders={"Open_Prediction": NaNLabelEncoder().fit(df.Open_Prediction)},
            group_ids=["Open_Prediction"],
            min_encoder_length=500 // 2,
            max_encoder_length=500,
            min_prediction_length=1,
            max_prediction_length=self.prediction_length,
            time_varying_unknown_reals=features,
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True
        )

        self.validation = TimeSeriesDataSet.from_dataset(self.training, df, predict=True, stop_randomization=True)

        # Store original (non-log) prices for plotting
        df_prices = yf.download(ticker, start=start_date, end=prediction_end).ffill()
        self.df_original = df_prices[['Open', 'High', 'Low', 'Close']]
        self.prediction_start = prediction_start
        self.prediction_end = prediction_end
        self.train_end = end_date

    def create_tft_model(self):
        # configure network and trainer
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
        lr_logger = LearningRateMonitor()  # log the learning rate
        logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard

        model_summary = ModelSummary(max_depth=1)

        self.trainer = pl.Trainer(
            max_epochs=10,
            accelerator="gpu",
            devices=1,
            gradient_clip_val=0.1,
            limit_train_batches=30,
            callbacks=[lr_logger, early_stop_callback, model_summary],
            logger=logger,
        )

        self.model = TemporalFusionTransformer.from_dataset(
            self.training,
            learning_rate=0.05,
            hidden_size=4,
            attention_head_size=1,
            dropout=0.1,
            hidden_continuous_size=4,
            output_size=7,  # This includes quantiles [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
            loss=QuantileLoss(),
            reduce_on_plateau_patience=4,
        )
        print(f"Number of parameters in network: {self.model.size() / 1e3:.1f}k")

    def train(self):
        train_dataloader = self.training.to_dataloader(train=True, batch_size=self.batch_size, num_workers=0)
        val_dataloader = self.validation.to_dataloader(train=False, batch_size=self.batch_size * 10, num_workers=0)

        self.trainer.fit(
            self.model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

    def evaluate_and_plot(self, number_of_examples=1):
        if self.trainer.checkpoint_callback is None or self.trainer.checkpoint_callback.best_model_path == "":
            print("No best model checkpoint found. Check training logs and early stopping conditions.")
            return
        best_model_path = self.trainer.checkpoint_callback.best_model_path
        best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

        val_dataloader = self.validation.to_dataloader(train=False, batch_size=self.batch_size * 10, num_workers=0)
        predictions, x = best_tft.predict(val_dataloader, return_x=True)

        if predictions.shape[0] == 0:
            print("No predictions returned. Check data configuration and ensure that prediction_length > 0.")
            return

        # **Changed median quantile index to 4 for the 0.5 quantile**
        median_pred = predictions[..., 4].detach().cpu().numpy().flatten()

        df = self.df_original.copy()
        train_end_ts = pd.Timestamp(self.train_end)
        pred_start_ts = pd.Timestamp(self.prediction_start)
        pred_end_ts = pd.Timestamp(self.prediction_end)

        df_ref = df.loc[df.index <= train_end_ts]
        if len(df_ref) == 0:
            # fallback if no exact match
            df_ref = df[df.index < train_end_ts]
            if len(df_ref) == 0:
                print("No reference data found for plotting. Check date ranges.")
                return

        df_future = df.loc[(df.index >= pred_start_ts) & (df.index <= pred_end_ts)]
        future_dates = pd.date_range(pred_start_ts, pred_end_ts, freq='D')[:len(median_pred)]

        last_ref_price = df_ref['Open'].iloc[-1]
        predicted_prices = np.exp(np.cumsum(median_pred)) * last_ref_price

        print("DEBUG INFO:")
        print("predictions shape:", predictions.shape)
        print("median_pred shape:", median_pred.shape)
        print("predicted_prices shape:", predicted_prices.shape)
        print("future_dates:", future_dates)
        print("df_ref shape:", df_ref.shape)
        print("df_future shape:", df_future.shape)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df_ref.index, df_ref['Open'], label='Reference Data (Actual)', color='blue')
        if len(predicted_prices) > 0 and len(future_dates) > 0:
            ax.plot(future_dates, predicted_prices, label='Predicted Data (Median)', color='red')
        else:
            print("No predicted prices or empty future dates. Check alignment and data availability.")

        if len(df_future) > 0:
            ax.plot(df_future.index, df_future['Open'], label='Actual Future Data', color='green', linestyle='--')

        ax.set_xlabel('Date')
        ax.set_ylabel('Open Price')
        ax.set_title('Stock Price Predictions')
        ax.legend()
        plt.tight_layout()
        return fig
