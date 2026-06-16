"""
simulation.py
--------------
This file is the "math brain" of the app. It has no buttons or charts in it —
just plain functions that:
    1. Download real historical stock prices from the internet (Yahoo Finance).
    2. Measure how that stock has historically behaved (its average daily
       return, and how volatile/jumpy it is).
    3. Use that information to run a Monte Carlo simulation: thousands of
       randomly-generated possible future price paths.

Keeping this separate from app.py (the website code) means we could test this
logic on its own, or reuse it in a totally different project later.
"""

import numpy as np
import pandas as pd
import yfinance as yf


def get_historical_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    """
    Downloads historical daily stock prices for a given ticker symbol
    (e.g. "AAPL" for Apple, "TSLA" for Tesla) using the yfinance library.

    Parameters
    ----------
    ticker : str
        The stock's ticker symbol.
    period : str
        How far back to pull data from. "2y" means 2 years of daily prices.
        More history generally gives a more reliable estimate of volatility.

    Returns
    -------
    pd.DataFrame
        A table of historical prices, indexed by date.
    """
    data = yf.download(ticker, period=period, progress=False, auto_adjust=True)

    if data.empty:
        raise ValueError(
            f"No data found for ticker '{ticker}'. Double check the symbol."
        )

    # Recent versions of yfinance always return columns in a "multi-level"
    # format (because the same function supports requesting several tickers
    # at once). Since we only ever ask for ONE ticker here, we flatten that
    # extra level away so the rest of our code can just say data["Close"]
    # and get a simple column of numbers, instead of a nested table.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data


def calculate_daily_returns(data: pd.DataFrame) -> pd.Series:
    """
    Converts a series of raw prices into daily PERCENTAGE returns.

    Why percentage returns instead of raw price changes? Because a $1 move
    means something very different for a $10 stock (10%!) than a $1000 stock
    (0.1%). Percentage returns let us fairly compare and combine days.

    We use LOG returns (natural logarithm of price ratios) rather than simple
    percentage returns because log returns are what the Geometric Brownian
    Motion formula is built around -- they have nicer mathematical properties
    (e.g. they can be added across days cleanly).
    """
    close_prices = data["Close"]
    log_returns = np.log(close_prices / close_prices.shift(1))
    return log_returns.dropna()  # drop the first row, which has no "yesterday" to compare to


def run_monte_carlo_simulation(
    last_price: float,
    daily_returns: pd.Series,
    num_days: int = 252,
    num_simulations: int = 1000,
) -> np.ndarray:
    """
    Runs the actual Monte Carlo simulation: generates many possible future
    price paths for a stock, starting from its most recent known price.

    Parameters
    ----------
    last_price : float
        The most recent real (known) closing price -- the starting point for
        every simulated path.
    daily_returns : pd.Series
        Historical daily log returns, used to measure the stock's average
        behavior and volatility.
    num_days : int
        How many trading days into the future to simulate. 252 trading days
        is roughly equal to 1 calendar year.
    num_simulations : int
        How many separate random future paths to generate. More simulations
        give a smoother, more reliable picture of possible outcomes, but take
        a little longer to compute.

    Returns
    -------
    np.ndarray
        A 2D array of shape (num_days + 1, num_simulations). Each COLUMN is
        one simulated future price path, starting at last_price.
    """
    # "mu" (drift) = the average daily log return historically observed.
    # This represents the stock's typical day-to-day trend.
    mu = daily_returns.mean()

    # "sigma" (volatility) = the standard deviation of daily log returns.
    # This measures how unpredictable/jumpy the stock has historically been.
    sigma = daily_returns.std()

    # We'll fill this array with simulated prices.
    # Row 0 is "today" (the real last known price) for every simulation.
    simulated_paths = np.zeros((num_days + 1, num_simulations))
    simulated_paths[0] = last_price

    # Generate ALL the random shocks we'll need in one go (this is much
    # faster in numpy than generating them one at a time in a loop).
    # Each value is drawn from a standard normal ("bell curve") distribution.
    random_shocks = np.random.standard_normal((num_days, num_simulations))

    # The Geometric Brownian Motion formula, applied one simulated day at a time:
    #   price_tomorrow = price_today * exp( (mu - 0.5 * sigma^2) + sigma * random_shock )
    #
    # The "mu - 0.5 * sigma^2" part is the expected drift, slightly adjusted
    # downward to correctly account for the mathematical effect of
    # compounding random log returns. The "sigma * random_shock" part is the
    # day's random surprise.
    drift = mu - 0.5 * sigma ** 2

    for day in range(1, num_days + 1):
        daily_growth_factor = np.exp(drift + sigma * random_shocks[day - 1])
        simulated_paths[day] = simulated_paths[day - 1] * daily_growth_factor

    return simulated_paths
