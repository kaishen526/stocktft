import pandas as pd

def preprocess_dataframe(df):
    """
    Preprocess the input DataFrame to ensure it matches the format
    required by the main pipeline and PytorchForecasting setup.

    Expected df format:
    - Columns: ["Date", "Open", "High", "Low", "Close"]
    - 'Date' column should be datetime or convertible to datetime.
    - No missing values that would disrupt the forecasting.

    Steps:
    1. Convert 'Date' column to datetime if not already.
    2. Sort by 'Date'.
    3. Drop rows with missing values in critical columns.
    4. Return cleaned DataFrame with a DatetimeIndex set on 'Date' (optional).

    Note: We do not perform normalization here since PytorchForecasting's
    TimeSeriesDataSet will handle scaling if needed.
    """

    # Ensure 'Date' is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Drop rows where date conversion failed
    df = df.dropna(subset=['Date'])

    # Sort by date
    df = df.sort_values('Date')

    # Drop rows with missing critical columns
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])

    # Optionally set index to 'Date'
    df = df.set_index('Date')

    # The DataFrame is now ready to be passed downstream.
    return df


# Example usage:
# In main.py or PytorchForecasting.py, after downloading data from yfinance:
#
# df = yf.download(ticker, start=start_date, end=prediction_end)
# df.reset_index(inplace=True)
# df = df[['Date', 'Open', 'High', 'Low', 'Close']]
#
# from Data_Preprocessing import preprocess_dataframe
# df_clean = preprocess_dataframe(df)
#
# df_clean can now be fed into the TFT pipeline.
#
# This file no longer depends on reading CSVs directly. Instead, it focuses on
# taking a DataFrame and ensuring it is properly cleaned and formatted.
