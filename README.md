# Temporal Fusion Transformer - NEA
<p align="center">
      <a href="https://www.python.org/">
        <img src="https://img.shields.io/badge/Python-3.12-ff69b4.svg" /></a>
       <a href= "https://pytorch.org/">
        <img src="https://img.shields.io/badge/PyTorch-1.13-2BAF2B.svg" /></a>
      <img src="https://img.shields.io/badge/license-MIT-blue.svg"/>

</p>

We present a stock prediction framework using Temporal Fusion Transformers (TFT) that inte grates historical market data with sentiment sig nals from financial news and social media. Our model demonstrates strong generalization capa bilities when trained on multiple energy sector stocks. Starting with single-stock predictions, we expand to a dataset of 50 NYSE/NASDAQ-listed energy stocks with market caps exceeding 1000M. Results show that multi-stock training enables the model to capture broader sector trends and generate reliable predictions for both seen and un seen stocks. The framework processes daily price data, fundamental metrics, and sentiment analysis from Alpha Vantage and Reddit’s WallStreetBets, providing probabilistic forecasts through an inter active interface.


The project write up can be read in Writeup.
 
Use intructions:
      1. pip install -r requirements.txt
      2. In Stock_Predictor_Project, 
               run tft_stock_prediction_GUI.py (if you would like a user interface)
            or run tft_stock_prediction_Server.py (if you run on GPU cloud)

