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

st.title("📈 Monte Carlo Stock Price Simulator")
st.write(
    "This app downloads a stock's real historical prices, measures how "
    "volatile it has been, and then simulates thousands of possible future "
    "price paths using random chance. It's a way of visualizing risk and "
    "uncertainty rather than pretending we can predict exact future prices."
)

# ---------------------------------------------------------------------------
# SIDEBAR -- user inputs live here, off to the side, so they don't crowd
# the main results area.
# ---------------------------------------------------------------------------
st.sidebar.header("Simulation Settings")

ticker_symbol = st.sidebar.text_input(
    "Stock ticker symbol", value="AAPL"
).strip().upper()

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
    "Number of simulated paths",
    min_value=100,
    max_value=5000,
    value=1000,
    step=100,
    help="More simulations = smoother, more reliable statistics, but slower "
    "to compute.",
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


# ---------------------------------------------------------------------------
# MAIN LOGIC -- only runs the (slow) simulation when the user clicks the
# button, rather than on every single slider tweak.
# ---------------------------------------------------------------------------
if run_button:
    try:
        with st.spinner(f"Downloading historical data for {ticker_symbol}..."):
            historical_data = load_data(ticker_symbol, history_period)
    except ValueError as error:
        st.error(str(error))
        st.stop()

    daily_returns = calculate_daily_returns(historical_data)
    last_price = float(historical_data["Close"].iloc[-1])

    with st.spinner(f"Running {num_simulations:,} simulations..."):
        simulated_paths = run_monte_carlo_simulation(
            last_price=last_price,
            daily_returns=daily_returns,
            num_days=num_days,
            num_simulations=num_simulations,
        )

    # The final row of simulated_paths holds the ending price of every
    # simulated path -- i.e. "where did each possible future end up?"
    ending_prices = simulated_paths[-1]

    # -----------------------------------------------------------------
    # KEY STATISTICS
    # We show last price, the average simulated outcome, and a 90%
    # "confidence range" (5th to 95th percentile). In plain English: a
    # percentile tells us where a value ranks. The 5th percentile is a
    # pessimistic but plausible outcome (only 5% of simulations did worse);
    # the 95th percentile is an optimistic but plausible outcome (only 5%
    # did better). Together they bracket a realistic range, instead of one
    # falsely-precise guess.
    # -----------------------------------------------------------------
    st.subheader(f"Results for {ticker_symbol}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Price", f"${last_price:,.2f}")
    col2.metric("Average Simulated Price", f"${ending_prices.mean():,.2f}")
    col3.metric("Pessimistic (5th pct.)", f"${np.percentile(ending_prices, 5):,.2f}")
    col4.metric("Optimistic (95th pct.)", f"${np.percentile(ending_prices, 95):,.2f}")

    # -----------------------------------------------------------------
    # CHART 1: All simulated price paths over time.
    # Plotting every single path lets the user visually see the "cone of
    # uncertainty" -- narrow near today, fanning out wider the further into
    # the future we simulate (because random shocks compound over time).
    # -----------------------------------------------------------------
    st.write("### Simulated Future Price Paths")

    # Plotting thousands of lines can be slow in a browser, so if there are
    # a lot of simulations, we only draw a random sample of them for
    # performance. The statistics above still use ALL simulations -- only
    # the visual is sampled.
    max_lines_to_draw = 300
    if num_simulations > max_lines_to_draw:
        columns_to_plot = np.random.choice(
            num_simulations, size=max_lines_to_draw, replace=False
        )
    else:
        columns_to_plot = range(num_simulations)

    fig_paths = go.Figure()
    for col_index in columns_to_plot:
        fig_paths.add_trace(
            go.Scatter(
                y=simulated_paths[:, col_index],
                mode="lines",
                line=dict(width=0.5),
                opacity=0.3,
                showlegend=False,
                hoverinfo="skip",
            )
        )
    # Draw the average path on top, in a bold color, so it's easy to spot
    # the "typical" trend among all the noisy random paths.
    fig_paths.add_trace(
        go.Scatter(
            y=simulated_paths.mean(axis=1),
            mode="lines",
            line=dict(width=3, color="black"),
            name="Average path",
        )
    )
    fig_paths.update_layout(
        xaxis_title="Trading Days From Today",
        yaxis_title="Simulated Price ($)",
        height=500,
    )
    st.plotly_chart(fig_paths, use_container_width=True)

    # -----------------------------------------------------------------
    # CHART 2: Histogram of ending prices.
    # While the chart above shows paths OVER TIME, this shows just the
    # FINAL outcome of every simulation, grouped into bars. It answers a
    # different question: "out of all possible futures, how likely is
    # each final price range?"
    # -----------------------------------------------------------------
    st.write("### Distribution of Final Simulated Prices")

    fig_hist = go.Figure(
        data=[go.Histogram(x=ending_prices, nbinsx=50)]
    )
    fig_hist.add_vline(
        x=last_price,
        line_dash="dash",
        line_color="red",
        annotation_text="Current Price",
    )
    fig_hist.update_layout(
        xaxis_title="Simulated Ending Price ($)",
        yaxis_title="Number of Simulations",
        height=400,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.caption(
        "⚠️ This is an educational simulation based on historical volatility "
        "and random chance. It is NOT financial advice and should not be "
        "used to make real investment decisions. Real markets are "
        "influenced by far more than historical statistics."
    )

else:
    st.info("👈 Set your simulation options in the sidebar, then click **Run Simulation**.")
