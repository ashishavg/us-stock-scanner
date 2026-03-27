"""
CIEN & Multi-Ticker Combined Trading Scanner
============================================
Module 1 - CIEN Hourly Pulse   : 5-min RSI + ATR + Volume + Gemini LLM -> Telegram
Module 2 - VSA Multi-Ticker    : Hourly VSA + RSI -> Telegram
Schedule  : 09:00-22:00 UTC Mon-Fri (4:30 AM - 6:00 PM ET, pre + post market covered)

Dependencies: yfinance, pandas, requests, google-genai ONLY — no pandas-ta
RSI computed with pure pandas calc_rsi() — works on any pandas version.
"""

import os
import requests
import yfinance as yf
import pandas as pd
from google import genai

# ---------------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID")
NEWS_API_KEY       = os.environ.get("NEWS_API_KEY")
GEMINI_API_KEY     = os.environ.get("GEMINI_API_KEY")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# ---------------------------------------------------------------------------
# VSA Configuration — tune thresholds here
# ---------------------------------------------------------------------------
VSA_CONFIG = {
    "default_vol_multiplier": 1.3,
    "meta_vol_multiplier":    1.2,
    "rsi_accumulation_max":   50,
    "rsi_distribution_min":   50,
    "rsi_period":             14,
    "sma_period":             20,
    "vol_avg_period":         20,
    "data_period":            "1mo",
    "data_interval":          "1h",
}

# ---------------------------------------------------------------------------
# Shared Utilities
# ---------------------------------------------------------------------------

def flatten_columns(df):
    """Flatten multi-index columns from newer yfinance versions."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def calc_rsi(series, period=14):
    """Pure-pandas RSI — no external TA library needed."""
    delta = series.diff()
    gain  = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss  = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs    = gain / loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


def send_telegram_message(message, parse_mode="Markdown"):
    """Unified Telegram sender — supports Markdown and HTML."""
    url     = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": parse_mode}
    try:
        r = requests.post(url, json=payload, timeout=10)
        r.raise_for_status()
        print(f"[Telegram] Sent ({parse_mode}).")
    except Exception as e:
        print(f"[Telegram] Failed: {e}")


def get_yf_news(stock_obj, n=3):
    """Returns n bullet headlines from yfinance built-in news."""
    try:
        return "\n".join(f"- {item['title']}" for item in stock_obj.news[:n])
    except Exception:
        return "- No recent news available."


def get_newsapi_headlines(n=5):
    """Fetches CIEN + macro headlines via NewsAPI. Falls back to yfinance news."""
    if not NEWS_API_KEY:
        print("[NewsAPI] Key not set — falling back to yfinance news.")
        return get_yf_news(yf.Ticker("CIEN"), n=n).split("\n")
    query = (
        "(Ciena OR CIEN OR AWS OR Google Cloud OR Meta OR Microsoft OR AI data center)"
        " OR (Federal Reserve OR rate cut)"
    )
    url = (
        "https://newsapi.org/v2/everything"
        f"?q={query}&sortBy=publishedAt&language=en&pageSize={n}&apiKey={NEWS_API_KEY}"
    )
    try:
        data = requests.get(url, timeout=10).json()
        return [a["title"] for a in data.get("articles", [])[:n]] or ["No major news found."]
    except Exception:
        return ["Failed to fetch NewsAPI headlines."]


# ---------------------------------------------------------------------------
# Module 1 — CIEN Intraday Pulse
# ---------------------------------------------------------------------------

def get_cien_technical_strength():
    cien = yf.Ticker("CIEN")
    df   = flatten_columns(cien.history(period="1d", interval="5m", prepost=True))

    result = {
        "signal": "NEUTRAL", "details": "Insufficient intraday data.",
        "price": 0.0, "pct_change": 0.0,
        "stop_loss": 0.0, "take_profit": 0.0, "atr": 0.0,
    }
    if df.empty or len(df) < 15:
        print("[CIEN] Not enough 5-min bars — market may not have opened yet.")
        return result

    current_price = round(float(df["Close"].iloc[-1]), 2)
    open_price    = float(df["Open"].iloc[0])
    pct_change    = round(((current_price - open_price) / open_price) * 100, 2)

    rsi = round(float(calc_rsi(df["Close"], 14).iloc[-1]), 1)

    df["vol_sma10"] = df["Volume"].rolling(10).mean()
    vol_surge       = bool(df["Volume"].iloc[-1] > (df["vol_sma10"].iloc[-1] * 1.5))

    hl  = df["High"] - df["Low"]
    hc  = (df["High"] - df["Close"].shift()).abs()
    lc  = (df["Low"]  - df["Close"].shift()).abs()
    atr = round(float(pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean().iloc[-1]), 2)

    rising = bool(df["Close"].iloc[-1] > df["Close"].iloc[-2])
    signal, details, sl, tp = "NEUTRAL", f"RSI: {rsi}. Volume normal.", 0.0, 0.0

    if rising and vol_surge:
        signal, details = "STRONG UP", f"RSI: {rsi}. Heavy buy volume detected."
        sl = round(current_price - 1.5 * atr, 2)
        tp = round(current_price + 3.0 * atr, 2)
    elif not rising and vol_surge:
        signal, details = "STRONG DOWN", f"RSI: {rsi}. Heavy sell volume detected."
        sl = round(current_price + 1.5 * atr, 2)
        tp = round(current_price - 3.0 * atr, 2)

    result.update({
        "signal": signal, "details": details, "price": current_price,
        "pct_change": pct_change, "stop_loss": sl, "take_profit": tp, "atr": atr,
    })
    return result


def get_gemini_sentiment(price, signal, details, headlines):
    prompt = f"""
    Current CIEN Price: ${price}.
    Technical Algorithmic Strength: {signal} ({details})
    Recent Headlines: {headlines}

    Classify the 1-hour outlook as HIGHER, NEUTRAL, or LOWER.
    Respond STRICTLY in this format (two lines only):
    SENTIMENT: [HIGHER/NEUTRAL/LOWER]
    RATIONALE: [2-sentence rationale]
    """
    try:
        resp      = gemini_client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        lines     = resp.text.strip().split("\n")
        sentiment = lines[0].replace("SENTIMENT:", "").strip()
        rationale = lines[1].replace("RATIONALE:", "").strip() if len(lines) > 1 else resp.text.strip()
        emoji     = " 🟢" if "HIGHER" in sentiment else (" 🔴" if "LOWER" in sentiment else " ⚪")
        return sentiment + emoji, rationale
    except Exception as e:
        return "ERROR", f"Gemini call failed: {e}"


def run_cien_pulse():
    print("[CIEN Pulse] Running...")
    tech                 = get_cien_technical_strength()
    headlines            = get_newsapi_headlines()
    sentiment, rationale = get_gemini_sentiment(tech["price"], tech["signal"], tech["details"], headlines)

    setup_block = ""
    if tech["signal"] != "NEUTRAL":
        setup_block = (
            f"*SUGGESTED TRADE SETUP (ATR: ${tech['atr']})*\n"
            f"- Entry:       ${tech['price']}\n"
            f"- Stop Loss:   ${tech['stop_loss']}\n"
            f"- Take Profit: ${tech['take_profit']}\n\n"
        )

    message = (
        f"*CIEN Hourly Pulse*\n"
        f"Price: ${tech['price']} ({tech['pct_change']}% today)\n"
        f"Order Strength: {tech['signal']} ({tech['details']})\n"
        f"Outlook: {sentiment}\n\n"
        f"{setup_block}"
        f"Rationale: {rationale}\n\n"
        f"Top Headline:\n_{headlines[0]}_"
    )
    send_telegram_message(message, parse_mode="Markdown")
    print(f"[CIEN Pulse] Done -> {tech['signal']} | {sentiment}")


# ---------------------------------------------------------------------------
# Module 2 — Multi-Ticker VSA Scanner
# ---------------------------------------------------------------------------

def analyze_vsa(ticker):
    stock = yf.Ticker(ticker)
    df    = flatten_columns(stock.history(
        period=VSA_CONFIG["data_period"],
        interval=VSA_CONFIG["data_interval"],
        auto_adjust=True,
    ))

    if df.empty or len(df) < VSA_CONFIG["sma_period"] + 1:
        print(f"[VSA] Not enough data for {ticker}.")
        return None

    df["SMA_20"]     = df["Close"].rolling(VSA_CONFIG["sma_period"]).mean()
    df["Avg_Vol_20"] = df["Volume"].rolling(VSA_CONFIG["vol_avg_period"]).mean()
    df["RSI_14"]     = calc_rsi(df["Close"], VSA_CONFIG["rsi_period"])

    latest  = df.iloc[-1]
    price   = float(latest["Close"])
    open_p  = float(latest["Open"])
    vol     = float(latest["Volume"])
    avg_vol = float(latest["Avg_Vol_20"])
    sma_20  = float(latest["SMA_20"])
    rsi     = float(latest["RSI_14"])

    vol_mult  = VSA_CONFIG["meta_vol_multiplier"] if ticker == "META" else VSA_CONFIG["default_vol_multiplier"]
    vol_ratio = vol / avg_vol if avg_vol > 0 else 0.0

    if (price < sma_20 and vol > vol_mult * avg_vol
            and price > open_p and rsi < VSA_CONFIG["rsi_accumulation_max"]):
        return (
            f"🟢 <b>ACCUMULATION: {ticker}</b>\n"
            f"Price: ${price:.2f} | RSI: {rsi:.1f} | SMA20: ${sma_20:.2f}\n"
            f"Volume: {vol_ratio:.1f}x average\n\n"
            f"<b>Headlines:</b>\n{get_yf_news(stock)}"
        )
    elif (price > sma_20 and vol > vol_mult * avg_vol
            and price < open_p and rsi > VSA_CONFIG["rsi_distribution_min"]):
        return (
            f"🔴 <b>DISTRIBUTION: {ticker}</b>\n"
            f"Price: ${price:.2f} | RSI: {rsi:.1f} | SMA20: ${sma_20:.2f}\n"
            f"Volume: {vol_ratio:.1f}x average\n\n"
            f"<b>Headlines:</b>\n{get_yf_news(stock)}"
        )
    return None


def run_vsa_scanner(tickers):
    print(f"[VSA Scanner] Scanning {len(tickers)} ticker(s)...")
    signals = []
    for ticker in tickers:
        print(f"  -> {ticker}")
        result = analyze_vsa(ticker)
        if result:
            signals.append(result)

    if signals:
        send_telegram_message("\n\n---\n\n".join(signals), parse_mode="HTML")
        print(f"[VSA] {len(signals)} signal(s) sent.")
    else:
        send_telegram_message(
            "<b>VSA Scanner</b>\nScan complete. No institutional volume anomalies detected.",
            parse_mode="HTML",
        )
        print("[VSA] No signals this hour.")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_cien_pulse()

    TICKERS_FILE = "tickers.txt"
    if not os.path.exists(TICKERS_FILE):
        print(f"[VSA] {TICKERS_FILE} not found. Create it with one ticker per line.")
    else:
        with open(TICKERS_FILE) as f:
            tickers = [line.strip().upper() for line in f if line.strip()]
        if tickers:
            run_vsa_scanner(tickers)
        else:
            print("[VSA] tickers.txt is empty.")
