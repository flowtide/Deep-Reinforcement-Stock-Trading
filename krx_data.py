import argparse
from pykrx import stock
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import os

def fetch_and_save_data(ticker, start, end, output_dir, chart=True):
    print(f"get_market_ticker_name '{ticker}'")
    name = stock.get_market_ticker_name(ticker)
    prefix = os.path.join(output_dir, f"{ticker}_{name}-{start}~{end}")
    filename = f"{prefix}.csv"
    print(f"get_market_ohlcv '{name}'")
    df = stock.get_market_ohlcv(start, end, ticker)
    
    # Rename columns and format DataFrame
    columns = {
        '시가': 'Open',
        '고가': 'High',
        '저가': 'Low',
        '종가': 'Close',
        '거래량': 'Volume'
    }
    df2 = df.rename(columns=columns).reset_index()
    df2.rename(columns={'날짜': 'Date'}, inplace=True)
    if '등락률' in df2.columns:
        df2.drop(columns=['등락률'], inplace=True)
    
    print(f"Data saved to {filename}")
    df2.to_csv(filename, index=False)

    if chart:
        chart_path = f"{prefix}.png"
        df2['Date'] = pd.to_datetime(df2['Date'])
        
        plt.figure(figsize=(15, 5), dpi=100)
        plt.plot(df2['Date'], df2['Close'], color='black', label=f"{ticker} ({name})")
        
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        plt.gcf().autofmt_xdate()
        plt.legend()
        plt.grid()
        
        print(f"Chart saved to {chart_path}")
        plt.savefig(chart_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch KRX stock data and save to CSV with an optional chart.")
    parser.add_argument("start", type=str, help="Start date in YYYYMMDD format (e.g., '20220101')")
    parser.add_argument("end", type=str, help="End date in YYYYMMDD format (e.g., '20241231')")
    parser.add_argument("ticker", type=str, nargs='+', help="One or more stock tickers (e.g., '105560')")
    parser.add_argument("--no-chart", action="store_false", dest="chart", help="If provided, do NOT save the chart. Default is to save the chart.")
    parser.add_argument("-d", "--dir", default=".", type=str, help="Output directory")
    
    args = parser.parse_args()
    
    # Loop through each ticker and fetch/save data
    for ticker in args.ticker:
        fetch_and_save_data(ticker, args.start, args.end, args.dir, args.chart)
