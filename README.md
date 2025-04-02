# programming-II-project-1
# AI STOCK FORECASTING AND RISK SIMULATION TOOL (Refactored and Clear Version)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import torch
from datetime import datetime

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# List of AI-related stocks
AI_TICKERS = ["NVDA", "MSFT", "GOOGL", "AMD", "PLTR", "AI", "FTNT", "BIDU"]

# --- DATA COLLECTION ---
def download_stock_data(ticker, start_date):
    data = yf.download(ticker, start=start_date)["Adj Close"].dropna()
    return data

def compute_log_returns(prices):
    return np.log(prices / prices.shift(1)).dropna()

# --- MONTE CARLO SIMULATION USING PYTORCH ---
def monte_carlo_forecast(prices, days=252, simulations=1000):
    log_returns = compute_log_returns(prices)
    mu = torch.tensor(log_returns.mean(), dtype=torch.float32, device=device)
    sigma = torch.tensor(log_returns.std(), dtype=torch.float32, device=device)
    last_price = torch.tensor(prices[-1], dtype=torch.float32, device=device)

    rand = torch.randn((simulations, days), device=device)
    drift = mu - 0.5 * sigma**2
    daily_returns = torch.exp(drift + sigma * rand)
    forecast = last_price * torch.cumprod(daily_returns, dim=1)

    return forecast.cpu().numpy(), float(mu.cpu()), float(sigma.cpu()), float(last_price.cpu())

# --- RISK METRICS ---
def value_at_risk(simulations, alpha=0.05):
    final_prices = simulations[:, -1]
    return np.percentile(final_prices, alpha * 100)

def sharpe_ratio(mu, sigma):
    annual_return = mu * 252
    annual_volatility = sigma * np.sqrt(252)
    return annual_return / annual_volatility if annual_volatility > 0 else np.nan

# --- FUNDAMENTAL ANALYSIS ---
def fetch_fundamentals(ticker):
    info = yf.Ticker(ticker).info
    return {
        "P/E": info.get("trailingPE"),
        "P/S": info.get("priceToSalesTrailing12Months"),
        "ROE": info.get("returnOnEquity"),
        "D/E": info.get("debtToEquity"),
        "Beta": info.get("beta"),
        "FCF": info.get("freeCashflow"),
        "EPS Growth": info.get("forwardEps", 0) / info.get("trailingEps", 1),
        "Revenue Growth": info.get("revenueGrowth"),
        "Operating Margin": info.get("operatingMargins"),
        "Market Cap": info.get("marketCap")
    }

# --- VISUALIZATION ---
def plot_simulations(simulations, last_price, ticker):
    plt.figure(figsize=(10, 6))
    plt.plot(simulations.T, alpha=0.1)
    plt.axhline(y=last_price, color='red', linestyle='--', label='Current Price')
    plt.title(f"Monte Carlo Forecast: {ticker}")
    plt.xlabel("Days")
    plt.ylabel("Simulated Price")
    plt.legend()
    plt.grid(True)
    plt.show()

# --- ANALYSIS LOOP ---
def analyze_ai_stocks(tickers, start_date="2019-01-01"):
    summary = []

    for ticker in tickers:
        try:
            print(f"\nAnalyzing {ticker}...")
            prices = download_stock_data(ticker, start_date)
            forecast, mu, sigma, last_price = monte_carlo_forecast(prices)
            plot_simulations(forecast, last_price, ticker)

            fundamentals = fetch_fundamentals(ticker)
            fundamentals.update({
                "Ticker": ticker,
                "Expected Return %": mu * 252 * 100,
                "Volatility %": sigma * np.sqrt(252) * 100,
                "Sharpe Ratio": sharpe_ratio(mu, sigma),
                "VaR 95%": value_at_risk(forecast)
            })

            summary.append(fundamentals)
            for key, value in fundamentals.items():
                if key != "Ticker":
                    print(f"{key}: {value:.4f}" if isinstance(value, (float, int)) else f"{key}: {value}")

        except Exception as e:
            print(f"Error analyzing {ticker}: {e}")

    return pd.DataFrame(summary).set_index("Ticker")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    summary_df = analyze_ai_stocks(AI_TICKERS)
    print("\n--- Summary of All Stocks ---")
    print(summary_df.round(3))
