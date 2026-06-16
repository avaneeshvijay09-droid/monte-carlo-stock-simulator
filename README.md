# 📈 Monte Carlo Stock Price Simulator

A web app that simulates thousands of possible future paths for a stock's
price, using real historical data and the Monte Carlo method — a technique
used throughout finance, physics, and engineering to model uncertainty.

**[🔗 Live demo](#)** *(link added after deployment)*

## What it does

1. Downloads a stock's real historical daily prices (via the [Yahoo Finance](https://finance.yahoo.com/) API).
2. Calculates how that stock has historically behaved: its average daily
   return ("drift") and how much it tends to fluctuate ("volatility").
3. Uses those two numbers to simulate thousands of independent, randomly
   generated possible future price paths.
4. Visualizes the full spread of outcomes as an interactive chart, plus
   summary statistics (expected price, optimistic/pessimistic ranges).

## Why Monte Carlo simulation?

No one can predict exactly what a stock will do tomorrow, let alone in a
year. Instead of pretending otherwise with one falsely-precise number,
Monte Carlo simulation embraces the uncertainty: it generates thousands of
*plausible* futures based on the stock's historical statistical behavior,
then lets us reason about the **range and likelihood** of outcomes — which
is much closer to how real risk analysis is done in finance, insurance,
and engineering.

## The math, briefly

Each simulated day, a stock's price is modeled as:

```
price_tomorrow = price_today × exp( (μ − 0.5σ²) + σ × Z )
```

where:
- **μ (mu)** = the stock's historical average daily log return (drift)
- **σ (sigma)** = the stock's historical daily volatility (standard deviation of returns)
- **Z** = a random number drawn from a standard normal ("bell curve") distribution — a new random draw every simulated day

This is called **Geometric Brownian Motion**, the standard model used to
simulate stock prices in quantitative finance. Repeating this day-by-day
calculation thousands of times, with fresh random numbers each time,
produces the "cloud" of possible future paths shown in the app.

## Tech stack

- **Python** — core language
- **[Streamlit](https://streamlit.io/)** — turns the Python script into an interactive website
- **[NumPy](https://numpy.org/)** — fast simulation math
- **[pandas](https://pandas.pydata.org/)** — historical data handling
- **[yfinance](https://github.com/ranaroussi/yfinance)** — free historical stock price data
- **[Plotly](https://plotly.com/python/)** — interactive charts

## Running it locally

```bash
git clone <your-repo-url>
cd monte-carlo-stock-simulator
python -m venv venv

# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

## Project structure

```
monte-carlo-stock-simulator/
├── app.py              # Streamlit UI: inputs, layout, charts
├── simulation.py        # Core Monte Carlo simulation logic (no UI code)
├── requirements.txt     # Python dependencies
└── README.md
```

## ⚠️ Disclaimer

This project is for educational purposes only. It is **not** financial
advice and should not be used to make real investment decisions. Real
markets are influenced by far more than historical statistics — news
events, company fundamentals, broader economic conditions, and investor
psychology all play a role that this simplified model does not capture.
