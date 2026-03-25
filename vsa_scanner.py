import yfinance as yf
import pandas as pd
import pandas_ta as ta
import requests
import os

# Fetch secrets from the GitHub environment
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
    except Exception as e:
        print(f"Error sending Telegram message: {e}")

def get_latest_news(stock_object):
    try:
        # Fetch the top 3 news items
        news_items = stock_object.news[:3]
        headlines = [f"• {item['title']}" for item in news_items]
        return "\n".join(headlines)
    except Exception:
        return "• No recent news found."

def analyze_stock(ticker):
    stock = yf.Ticker(ticker)
    
    # Fetch 1 month of hourly data
    df = stock.history(period="1mo", interval="1h")
    
    if df.empty or len(df) < 21:
        print(f"Not enough data for {ticker}.")
        return None

    # Calculate VSA and RSI Indicators
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['Avg_Vol_20'] = df['Volume'].rolling(window=20).mean()
    df.ta.rsi(length=14, append=True) 
    
    # Grab the data for the most recently closed hour
    latest = df.iloc[-1]
    
    current_price = latest['Close']
    open_price = latest['Open']
    current_vol = latest['Volume']
    sma_20 = latest['SMA_20']
    avg_vol = latest['Avg_Vol_20']
    rsi = latest['RSI_14']

    # NEW: Loosened volume multipliers for hourly sensitivity
    vol_multiplier = 1.2 if ticker == 'META' else 1.5 
    signal = None
    
    # NEW: Loosened RSI Accumulation threshold (< 45)
    if current_price < sma_20 and current_vol > (vol_multiplier * avg_vol) and current_price > open_price and rsi < 45:
        news_headlines = get_latest_news(stock)
        signal = (f"🟢 <b>ACCUMULATION: {ticker}</b>\n"
                  f"Price: ${current_price:.2f} | RSI: {rsi:.1f}\n"
                  f"Volume: {current_vol/avg_vol:.1f}x average\n\n"
                  f"<b>Latest Headlines:</b>\n{news_headlines}")
        
    # NEW: Loosened RSI Distribution threshold (> 55)
    elif current_price > sma_20 and current_vol > (vol_multiplier * avg_vol) and current_price < open_price and rsi > 55:
        news_headlines = get_latest_news(stock)
        signal = (f"🔴 <b>DISTRIBUTION: {ticker}</b>\n"
                  f"Price: ${current_price:.2f} | RSI: {rsi:.1f}\n"
                  f"Volume: {current_vol/avg_vol:.1f}x average\n\n"
                  f"<b>Latest Headlines:</b>\n{news_headlines}")

    return signal

if __name__ == "__main__":
    if not os.path.exists('tickers.txt'):
        print("tickers.txt not found.")
        exit()
        
    with open('tickers.txt', 'r') as f:
        tickers = [line.strip() for line in f if line.strip()]
    
    messages = []
    for ticker in tickers:
        print(f"Analyzing {ticker}...")
        result = analyze_stock(ticker)
        if result:
            messages.append(result)
            
    # Send all triggered signals in a single Telegram message
    if messages:
        final_message = "\n\n".join(messages)
        send_telegram_message(final_message)
        print("Alerts sent to Telegram.")
    else:
        print("No VSA/RSI signals detected for this hour.")
        # Heartbeat message kept intact
        send_telegram_message("✅ <b>VSA Scanner Update</b>\nScan complete. No significant institutional volume anomalies detected right now.")
