"""
app.py
------
This is the website itself. Streamlit turns this plain Python script into an
interactive page: every time the user changes a slider or text box, Streamlit
re-runs this whole file from top to bottom with the new values, and redraws
the page. We keep all the actual math in simulation.py and just call it from
here -- this file is only responsible for layout, inputs, and charts.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from simulation import (
    calculate_daily_returns,
    get_historical_data,
    run_monte_carlo_simulation,
)

# ---------------------------------------------------------------------------
# PAGE CONFIG
# This must be the first Streamlit command in the script. It controls the
# browser tab's title/icon and makes the page use the full screen width
# instead of a narrow centered column.
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Monte Carlo Stock Simulator",
    page_icon="📈",
    layout="wide",
)

# ---------------------------------------------------------------------------
# A handful of colors we'll reuse consistently for whichever tickers the
# user enters, so the same stock is always the same color across every
# chart and metric on the page. Plain CSS variable-style hex codes.
# ---------------------------------------------------------------------------
TICKER_COLORS = [
    "#4C78A8",  # blue
    "#F58518",  # orange
    "#54A24B",  # green
    "#E45756",  # red
    "#B279A2",  # purple
    "#FF9DA6",  # pink
    "#9C755F",  # brown
    "#72B7B2",  # teal
]
MAX_TICKERS = 8

# ---------------------------------------------------------------------------
# CUSTOM STYLING
# Streamlit's default look is plain by design. A small amount of custom CSS,
# injected via st.markdown with unsafe_allow_html=True, lets us add a nicer
# header banner and cleaner metric "cards" without writing a single line of
# JavaScript or touching any other framework.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .hero-banner {
        background: linear-gradient(135deg, #4C78A8 0%, #54A24B 100%);
        padding: 1.75rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .hero-banner h1 {
        color: white;
        margin-bottom: 0.3rem;
    }
    .hero-banner p {
        color: rgba(255,255,255,0.9);
        margin-bottom: 0;
        font-size: 1.05rem;
    }
    div[data-testid="stMetric"] {
        background-color: rgba(128,128,128,0.08);
        border: 1px solid rgba(128,128,128,0.15);
        border-radius: 10px;
        padding: 0.9rem 1rem 0.6rem 1rem;
    }
    </style>
    <div class="hero-banner">
        <h1>📈 Monte Carlo Stock Price Simulator</h1>
        <p>Simulate thousands of possible futures for one or more stocks,
        based on their real historical behavior -- and compare them
        side by side.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# EXPLAINER SECTION
# Wrapped in an st.expander so it's collapsed by default -- visible and easy
# to find for anyone curious, but not in the way of people who just want to
# run the simulation.
# ---------------------------------------------------------------------------
with st.expander("❓ What is a Monte Carlo simulation, and why use one here?"):
    st.markdown(
        """
No one can predict exactly what a stock will be worth next year. Instead of
pretending otherwise with a single falsely-precise guess, a **Monte Carlo
simulation** embraces that uncertainty directly: it generates *thousands* of
different possible futures using random chance, and then looks at the whole
spread of outcomes.

**How it works, step by step:**
1. We download a stock's real historical daily prices.
2. We measure two things from that history: its **drift** (the average
   day-to-day return -- the general trend) and its **volatility** (how much
   it tends to swing up or down from day to day -- how "risky" or jumpy it
   is).
3. We simulate one possible future, one day at a time: take yesterday's
   price, nudge it by the drift, then add a *random* shock sized according
   to the volatility (bigger for volatile stocks, smaller for stable ones).
4. We repeat step 3 for as many days as requested (e.g. 252 trading days =
   about 1 year), which produces **one** possible future price path.
5. We repeat the *entire process* thousands of times, each time with fresh
   random numbers, producing thousands of independent possible futures.

The specific formula used each simulated day is called **Geometric Brownian
Motion**, the standard model for this in quantitative finance:

```
price_tomorrow = price_today × exp( (μ − 0.5σ²) + σ × Z )
```

where **μ** (mu) is the historical drift, **σ** (sigma) is the historical
volatility, and **Z** is a fresh random number from a bell-curve
distribution, drawn anew every simulated day.

Looking at the *whole cloud* of simulated outcomes -- rather than one
single predicted number -- gives a far more honest picture of risk: a wide,
spread-out cloud means a highly unpredictable stock; a narrow cloud means a
more historically stable one.

⚠️ **This is a simplified educational model.** It only knows about a
stock's *past* statistical behavior -- it has no idea about news, company
fundamentals, or anything that hasn't happened yet. It is not financial
advice.
        """
    )

# ---------------------------------------------------------------------------
# SIDEBAR -- user inputs live here, off to the side, so they don't crowd
# the main results area.
# ---------------------------------------------------------------------------
st.sidebar.header("Simulation Settings")

tickers_raw = st.sidebar.text_input(
    "Stock ticker symbol(s)",
    value="",
    placeholder="e.g. AAPL, MSFT, TSLA",
    help="Enter one ticker, or several separated by commas, to compare "
    f"them side by side (max {MAX_TICKERS}). Example: AAPL, MSFT, TSLA",
)

history_period = st.sidebar.selectbox(
    "Historical data to analyze",
    options=["1y", "2y", "5y", "10y"],
    index=1,  # defaults to "2y"
    help="More history gives a more stable estimate of volatility, but may "
    "include outdated market conditions.",
)

num_days = st.sidebar.slider(
    "Trading days to simulate into the future",
    min_value=30,
    max_value=504,  # roughly 2 years of trading days
    value=252,  # roughly 1 year
    step=1,
    help="There are about 252 trading days in a year.",
)

num_simulations = st.sidebar.slider(
    "Number of simulated paths (per stock)",
    min_value=100,
    max_value=5000,
    value=1000,
    step=100,
    help="More simulations = smoother, more reliable statistics, but slower "
    "to compute -- and this cost multiplies by however many tickers you "
    "enter.",
)

run_button = st.sidebar.button("Run Simulation", type="primary")


# ---------------------------------------------------------------------------
# CACHED DATA FETCH
# The @st.cache_data decorator tells Streamlit: "if this function gets
# called again with the SAME arguments, don't re-run it -- just reuse the
# result from last time." Without this, moving a slider (which reruns the
# whole script) would re-download the same stock data over and over, which
# is slow and unnecessary.
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)  # cached result expires after 1 hour
def load_data(ticker: str, period: str) -> pd.DataFrame:
    return get_historical_data(ticker, period)


def parse_tickers(raw_text: str) -> list[str]:
    """
    Turns whatever the user typed (e.g. "aapl, msft  , tsla") into a clean,
    de-duplicated list of uppercase ticker symbols (e.g. ["AAPL", "MSFT",
    "TSLA"]), preserving the order they were entered in.
    """
    seen = set()
    cleaned = []
    for piece in raw_text.split(","):
        symbol = piece.strip().upper()
        if symbol and symbol not in seen:
            seen.add(symbol)
            cleaned.append(symbol)
    return cleaned


# ---------------------------------------------------------------------------
# MAIN LOGIC -- only runs the (slow) simulation when the user clicks the
# button, rather than on every single slider tweak.
# ---------------------------------------------------------------------------
if run_button:
    tickers = parse_tickers(tickers_raw)

    if not tickers:
        st.error("Please enter at least one stock ticker symbol.")
        st.stop()

    if len(tickers) > MAX_TICKERS:
        st.warning(
            f"You entered {len(tickers)} tickers -- only the first "
            f"{MAX_TICKERS} will be simulated, to keep things readable."
        )
        tickers = tickers[:MAX_TICKERS]

    # We'll fill this dictionary with one entry per successfully-simulated
    # ticker, e.g. results["AAPL"] = {"last_price": ..., "simulated_paths": ...}
    results = {}

    for ticker in tickers:
        try:
            with st.spinner(f"Downloading historical data for {ticker}..."):
                historical_data = load_data(ticker, history_period)
        except ValueError as error:
            # If one ticker fails (e.g. typo'd symbol), warn about that one
            # specifically but keep going with the rest, rather than
            # crashing the whole page.
            st.warning(str(error))
            continue

        daily_returns = calculate_daily_returns(historical_data)
        last_price = float(historical_data["Close"].iloc[-1])

        with st.spinner(f"Running {num_simulations:,} simulations for {ticker}..."):
            simulated_paths = run_monte_carlo_simulation(
                last_price=last_price,
                daily_returns=daily_returns,
                num_days=num_days,
                num_simulations=num_simulations,
            )

        results[ticker] = {
            "last_price": last_price,
            "simulated_paths": simulated_paths,
            "ending_prices": simulated_paths[-1],  # final day's prices across all sims
        }

    if not results:
        st.error("None of the entered tickers could be simulated. Check the symbols and try again.")
        st.stop()

    ticker_color = {ticker: TICKER_COLORS[i % len(TICKER_COLORS)] for i, ticker in enumerate(results)}

    # -----------------------------------------------------------------
    # KEY STATISTICS -- one set of metric cards per ticker, color-coded
    # to match the charts below. A percentile tells us where a value
    # ranks: the 5th percentile is a pessimistic but plausible outcome
    # (only 5% of simulations did worse); the 95th percentile is an
    # optimistic but plausible outcome (only 5% did better). Together they
    # bracket a realistic range, instead of one falsely-precise guess.
    # -----------------------------------------------------------------
    st.write("## Results")

    for ticker, data in results.items():
        last_price = data["last_price"]
        ending_prices = data["ending_prices"]
        expected_change_pct = (ending_prices.mean() / last_price - 1) * 100

        st.markdown(
            f'<h4 style="color:{ticker_color[ticker]};">● {ticker}</h4>',
            unsafe_allow_html=True,
        )
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Price", f"${last_price:,.2f}")
        col2.metric(
            "Average Simulated Price",
            f"${ending_prices.mean():,.2f}",
            delta=f"{expected_change_pct:+.1f}%",
        )
        col3.metric("Pessimistic (5th pct.)", f"${np.percentile(ending_prices, 5):,.2f}")
        col4.metric("Optimistic (95th pct.)", f"${np.percentile(ending_prices, 95):,.2f}")

    # -----------------------------------------------------------------
    # COMPARISON CHART: average simulated path per ticker, with a shaded
    # "confidence band" showing the 5th-95th percentile range at every day.
    # When comparing multiple stocks, drawing every single simulated path
    # for every stock would just be visual noise -- so instead we summarize
    # each stock down to its average trend line plus its spread, all on one
    # shared chart so they're directly comparable.
    # -----------------------------------------------------------------
    st.write("### Simulated Price Paths (Comparison)")

    fig_compare = go.Figure()
    for ticker, data in results.items():
        paths = data["simulated_paths"]
        color = ticker_color[ticker]

        # axis=1 means "across all simulations, for each day" -- so this
        # gives us one average price, and one 5th/95th percentile, per day.
        mean_path = paths.mean(axis=1)
        low_band = np.percentile(paths, 5, axis=1)
        high_band = np.percentile(paths, 95, axis=1)
        days_axis = list(range(len(mean_path)))

        # Shaded band: drawn as a filled area between the low and high
        # percentile lines, using a transparent version of the ticker's color.
        fig_compare.add_trace(
            go.Scatter(
                x=days_axis + days_axis[::-1],
                y=list(high_band) + list(low_band[::-1]),
                fill="toself",
                fillcolor=color,
                opacity=0.15,
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        # Bold average line on top of the shaded band.
        fig_compare.add_trace(
            go.Scatter(
                x=days_axis,
                y=mean_path,
                mode="lines",
                line=dict(width=3, color=color),
                name=ticker,
            )
        )

    fig_compare.update_layout(
        xaxis_title="Trading Days From Today",
        yaxis_title="Simulated Price ($)",
        height=500,
        legend_title_text="Ticker",
        hovermode="x unified",
    )
    st.plotly_chart(fig_compare, use_container_width=True)
    st.caption(
        "Bold lines show each stock's average simulated path. Shaded bands "
        "show the 5th-95th percentile range -- the realistic spread of "
        "outcomes, not just one guess."
    )

    # -----------------------------------------------------------------
    # COMPARISON HISTOGRAM: overlaid distributions of final simulated
    # prices, one semi-transparent histogram per ticker. This answers a
    # different question than the chart above: not "how did the price
    # move over time," but "out of all possible futures, how likely is
    # each final price range?"
    # -----------------------------------------------------------------
    st.write("### Distribution of Final Simulated Prices")

    fig_hist = go.Figure()
    for ticker, data in results.items():
        fig_hist.add_trace(
            go.Histogram(
                x=data["ending_prices"],
                name=ticker,
                nbinsx=50,
                marker_color=ticker_color[ticker],
                opacity=0.55,
            )
        )
    fig_hist.update_layout(
        barmode="overlay",
        xaxis_title="Simulated Ending Price ($)",
        yaxis_title="Number of Simulations",
        height=400,
        legend_title_text="Ticker",
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # -----------------------------------------------------------------
    # PER-TICKER DETAIL: the full "spaghetti" view of individual simulated
    # paths, tucked into a collapsed expander per ticker so the main page
    # stays clean, but the detail is still one click away.
    # -----------------------------------------------------------------
    st.write("### Detailed View Per Stock")
    for ticker, data in results.items():
        with st.expander(f"Show all simulated paths for {ticker}"):
            paths = data["simulated_paths"]
            num_sims_this_ticker = paths.shape[1]

            # Plotting thousands of lines can be slow in a browser, so we
            # only draw a random sample for the visual. The statistics
            # elsewhere on the page still use ALL simulations.
            max_lines_to_draw = 300
            if num_sims_this_ticker > max_lines_to_draw:
                columns_to_plot = np.random.choice(
                    num_sims_this_ticker, size=max_lines_to_draw, replace=False
                )
            else:
                columns_to_plot = range(num_sims_this_ticker)

            fig_detail = go.Figure()
            for col_index in columns_to_plot:
                fig_detail.add_trace(
                    go.Scatter(
                        y=paths[:, col_index],
                        mode="lines",
                        line=dict(width=0.5, color=ticker_color[ticker]),
                        opacity=0.25,
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
            fig_detail.add_trace(
                go.Scatter(
                    y=paths.mean(axis=1),
                    mode="lines",
                    line=dict(width=3, color="black"),
                    name="Average path",
                )
            )
            fig_detail.update_layout(
                xaxis_title="Trading Days From Today",
                yaxis_title="Simulated Price ($)",
                height=450,
            )
            st.plotly_chart(fig_detail, use_container_width=True)

    st.caption(
        "⚠️ This is an educational simulation based on historical volatility "
        "and random chance. It is NOT financial advice and should not be "
        "used to make real investment decisions. Real markets are "
        "influenced by far more than historical statistics."
    )

else:
    st.info("👈 Set your simulation options in the sidebar, then click **Run Simulation**.")
