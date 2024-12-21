import requests
import pandas as pd
import datetime
import time

ALPHA_VANTAGE_API_KEY = "BIUXFLM54A6Q9MWW"  # Replace with your actual key

tickers = [
    'XOM', 'CVX', 'SHEL', 'COP', 'TTE', 'ENB', 'PBR', 'BP', 'EOG', 'EPD', 'CNQ', 'WMB', 'ET', 'EQNR', 'OKE',
    'KMI', 'SLB', 'PSX', 'MPLX', 'TRP', 'LNG', 'FANG', 'MPC', 'SU', 'OXY', 'E', 'HES', 'BKR', 'VLO', 'TRGP', 'IMO',
    'WDS', 'CQP', 'CVE', 'TPL', 'EQT', 'HAL', 'CCJ', 'EXE', 'DVN', 'PBA', 'TS', 'CTRA', 'YPF', 'EC', 'WES', 'PAA',
    'OVV', 'PR', 'DTM', 'AR', 'RRC', 'APA', 'CHRD', 'AM', 'SUN', 'MTDR', 'DINO', 'ENLC', 'NOV', 'CNX', 'VIST', 'CHX',
    'VNOM', 'CRC', 'SOBO', 'CRK', 'MGY', 'CIVI', 'SM', 'AROC', 'TGS', 'MUR', 'NXE', 'WHD', 'NOG', 'HESM', 'PAGP',
    'KGS', 'CEIX', 'NFE', 'ARLP', 'KNTK', 'PBF', 'UEC', 'HP', 'FRO', 'GPOR', 'LBRT', 'VAL', 'UGP', 'BSM', 'PTEN',
    'VRN', 'HAFN', 'CSAN', 'CRGY', 'BTU', 'USAC', 'OII', 'AESI', 'TDW', 'SDRL', 'STNG', 'PEYUF', 'DKL', 'BTE', 'BKV',
    'CVI', 'TRMD', 'SOC', 'DNN', 'CLMT', 'NEXT', 'HPK', 'GLP', 'TALO', 'CMBT', 'INSW', 'STR', 'WKC', 'BWLP', 'MNR',
    'DMLP', 'DHT', 'VET', 'KOS', 'WTTR', 'NRP', 'HLX', 'RES', 'TNK', 'XPRO', 'GEL', 'KRP', 'ACDC', 'FLNG', 'VTLE',
    'LEU', 'EFXT', 'UUUU', 'DK', 'SEI', 'LPG'
]

start_date = "2018-01-02"
end_date = "2025-01-01"


def to_av_format(dt):
    return dt.strftime("%Y%m%dT%H%M")


def fetch_news_sentiment(ticker, start_date, end_date):
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    time_from_str = to_av_format(start_dt)
    time_to_str = to_av_format(end_dt)

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "time_from": time_from_str,
        "time_to": time_to_str,
        "limit": "1000",
        "apikey": ALPHA_VANTAGE_API_KEY
    }

    response = requests.get(url, params=params)
    data = response.json()

    if "feed" not in data:
        # No data returned, could be rate limit or no news found
        return pd.DataFrame(columns=["Date", "SentimentScore", "Ticker"])

    daily_scores = {}
    for article in data["feed"]:
        time_published = article.get("time_published")
        sentiment_score = article.get("overall_sentiment_score", 0.0)
        date_str = time_published[:8]  # first 8 chars: YYYYMMDD
        date_dt = datetime.datetime.strptime(date_str, "%Y%m%d")

        # Check date range
        if start_dt <= date_dt <= end_dt:
            daily_scores.setdefault(date_dt.date(), []).append(float(sentiment_score))

    sentiment_data = []
    for d, scores in daily_scores.items():
        sentiment_data.append((d, sum(scores) / len(scores), ticker))

    df = pd.DataFrame(sentiment_data, columns=['Date', 'SentimentScore', 'Ticker'])
    return df


if __name__ == "__main__":
    combined_df = pd.DataFrame(columns=['Date', 'SentimentScore', 'Ticker'])

    for t in tickers:
        print(f"Fetching data for {t}")
        df = fetch_news_sentiment(t, start_date, end_date)

        if not df.empty:
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        else:
            print(f"No data returned for {t}")

        # To respect any rate limits, you might add a small sleep:
        time.sleep(12)  # Adjust as needed based on your plan limits

    # After retrieving all sentiment data, save to a single CSV
    combined_csv_filename = f"combined_sentiment_{start_date}_to_{end_date}.csv"
    combined_df.to_csv(combined_csv_filename, index=False)
    print(f"All data saved to {combined_csv_filename}")
