
import time
import requests
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="Vivek AI Trader",
    page_icon="₿",
    layout="wide"
)

API = "https://api.coingecko.com/api/v3"

st.title("₿ Vivek AI Trader")
st.caption("Multi-Crypto Scanner | Spot Trading | Manual Approval")

# ---------------- SETTINGS ----------------

st.sidebar.header("Trading Settings")

capital = st.sidebar.number_input(
    "Your capital (₹)",
    min_value=500,
    max_value=1000000,
    value=4000,
    step=500
)

risk_percent = st.sidebar.slider(
    "Maximum risk per trade (%)",
    0.5, 2.0, 1.0, 0.5
)

coin_count = st.sidebar.selectbox(
    "Coins to scan",
    [5, 8, 10],
    index=1
)

min_volume = st.sidebar.number_input(
    "Minimum 24h trading volume (₹)",
    min_value=0,
    value=100000000,
    step=10000000
)

# ---------------- API ----------------

@st.cache_data(ttl=300)
def get_top_coins(count):
    response = requests.get(
        f"{API}/coins/markets",
        params={
            "vs_currency": "inr",
            "order": "volume_desc",
            "per_page": 100,
            "page": 1,
            "sparkline": "false"
        },
        timeout=25
    )
    response.raise_for_status()

    data = response.json()

    excluded = {
        "tether", "usd-coin", "dai", "true-usd",
        "first-digital-usd", "usds", "usde",
        "wrapped-bitcoin", "wrapped-steth"
    }

    coins = [
        coin for coin in data
        if coin["id"] not in excluded
        and coin.get("total_volume", 0) >= min_volume
        and coin.get("current_price", 0) > 0
    ]

    return coins[:count]


@st.cache_data(ttl=300)
def get_coin_history(coin_id):
    response = requests.get(
        f"{API}/coins/{coin_id}/market_chart",
        params={
            "vs_currency": "inr",
            "days": "1"
        },
        timeout=25
    )
    response.raise_for_status()

    data = response.json()

    df = pd.DataFrame(
        data["prices"],
        columns=["timestamp", "price"]
    )

    df["time"] = pd.to_datetime(
        df["timestamp"], unit="ms"
    )

    return df.set_index("time")[["price"]]


# ---------------- INDICATORS ----------------

def analyse_coin(coin, history):
    df = history.copy()

    if len(df) < 35:
        return None

    price = df["price"]

    df["EMA9"] = price.ewm(
        span=9, adjust=False
    ).mean()

    df["EMA21"] = price.ewm(
        span=21, adjust=False
    ).mean()

    delta = price.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1/14, adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1/14, adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["RSI"] = (
        100 - (100 / (1 + rs))
    ).fillna(50)

    ema12 = price.ewm(
        span=12, adjust=False
    ).mean()

    ema26 = price.ewm(
        span=26, adjust=False
    ).mean()

    df["MACD"] = ema12 - ema26

    df["MACD_signal"] = df["MACD"].ewm(
        span=9, adjust=False
    ).mean()

    latest = df.iloc[-1]

    current = float(latest["price"])
    rsi = float(latest["RSI"])

    score = 0
    reasons = []

    # Trend
    if latest["EMA9"] > latest["EMA21"]:
        score += 30
        reasons.append("EMA bullish")

    # MACD
    if latest["MACD"] > latest["MACD_signal"]:
        score += 25
        reasons.append("MACD bullish")

    # RSI
    if 50 <= rsi <= 68:
        score += 25
        reasons.append("RSI healthy")

    elif rsi > 75:
        reasons.append("RSI overbought")

    # Recent momentum
    momentum = (
        current / float(price.iloc[-13]) - 1
    ) * 100

    if momentum > 0:
        score += 20
        reasons.append("Positive momentum")

    # Avoid chasing overbought coins
    if rsi > 75:
        score = min(score, 45)

    # Risk calculation using recent price volatility
    returns = price.pct_change().dropna()

    volatility = float(
        returns.tail(12).std()
    )

    stop_distance = max(
        current * volatility * 2,
        current * 0.005
    )

    entry = current
    stop_loss = max(0, entry - stop_distance)
    target = entry + (stop_distance * 2)

    risk_money = capital * risk_percent / 100

    risk_per_coin = entry - stop_loss

    quantity_by_risk = (
        risk_money / risk_per_coin
        if risk_per_coin > 0 else 0
    )

    quantity_by_capital = (
        capital * 0.95 / entry
    )

    quantity = min(
        quantity_by_risk,
        quantity_by_capital
    )

    allocation = quantity * entry

    if score >= 75 and rsi < 75:
        signal = "BUY CANDIDATE"
    elif score <= 30:
        signal = "AVOID"
    else:
        signal = "WAIT"

    return {
        "Coin": coin["name"],
        "Symbol": coin["symbol"].upper(),
        "Price (INR)": current,
        "24h Change (%)": coin.get(
            "price_change_percentage_24h", 0
        ) or 0,
        "24h Volume (INR)": coin["total_volume"],
        "RSI": rsi,
        "Momentum (%)": momentum,
        "Score": score,
        "Signal": signal,
        "Entry (INR)": entry,
        "Stop Loss (INR)": stop_loss,
        "Target (INR)": target,
        "Quantity": quantity,
        "Allocation (INR)": allocation,
        "Reasons": ", ".join(reasons),
        "History": df
    }


# ---------------- SCANNER ----------------

st.subheader("Market Scanner")

if st.button("🔍 Scan Crypto Market", type="primary"):

    try:
        with st.spinner("Finding liquid cryptocurrencies..."):
            coins = get_top_coins(coin_count)

        results = []

        progress = st.progress(0)
        status = st.empty()

        for i, coin in enumerate(coins):

            status.write(
                f"Analysing {coin['name']} ({i+1}/{len(coins)})"
            )

            try:
                history = get_coin_history(coin["id"])

                result = analyse_coin(coin, history)

                if result:
                    results.append(result)

            except Exception as error:
                st.warning(
                    f"Skipped {coin['name']}: {error}"
                )

            progress.progress((i + 1) / len(coins))

            # Avoid excessive public API requests
            if i < len(coins) - 1:
                time.sleep(5)

        status.empty()
        progress.empty()

        if results:
            results.sort(
                key=lambda x: x["Score"],
                reverse=True
            )

            st.session_state["results"] = results

            st.success(
                f"Analysis complete: {len(results)} coins analysed."
            )
        else:
            st.error("No usable market data received.")

    except Exception as error:
        st.error(f"Market scanner error: {error}")


# ---------------- RESULTS ----------------

if "results" in st.session_state:

    results = st.session_state["results"]

    st.subheader("🏆 Best Trading Opportunities")

    candidates = [
        r for r in results
        if r["Signal"] == "BUY CANDIDATE"
    ]

    if candidates:
        best = candidates[0]

        st.success(
            f"Top candidate: {best['Coin']} "
            f"({best['Symbol']}) — Score {best['Score']}/100"
        )

        st.write(
            "This is the highest-ranked setup from the scanned coins, "
            "not a guaranteed profitable trade."
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Entry",
            f"₹{best['Entry (INR)']:,.4f}"
        )

        c2.metric(
            "Stop Loss",
            f"₹{best['Stop Loss (INR)']:,.4f}"
        )

        c3.metric(
            "Target",
            f"₹{best['Target (INR)']:,.4f}"
        )

        st.write(
            f"**Suggested allocation:** "
            f"₹{best['Allocation (INR)']:,.2f}"
        )

        st.write(
            f"**Estimated quantity:** {best['Quantity']:.8f}"
        )

        st.write(
            f"**Reasons:** {best['Reasons']}"
        )

    else:
        st.warning(
            "No strong BUY candidate found. "
            "The scanner recommends waiting."
        )

    st.subheader("All Scanned Coins")

    table = pd.DataFrame([
        {
            "Coin": r["Coin"],
            "Symbol": r["Symbol"],
            "Price": r["Price (INR)"],
            "24h Change %": r["24h Change (%)"],
            "RSI": r["RSI"],
            "Score": r["Score"],
            "Signal": r["Signal"]
        }
        for r in results
    ])

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True
    )

    # ---------------- SELECT COIN ----------------

    st.subheader("Detailed Coin Analysis")

    selected_symbol = st.selectbox(
        "Choose a coin",
        [r["Symbol"] for r in results]
    )

    selected = next(
        r for r in results
        if r["Symbol"] == selected_symbol
    )

    chart = selected["History"]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=chart.index,
        y=chart["price"],
        name="Price"
    ))

    fig.add_trace(go.Scatter(
        x=chart.index,
        y=chart["EMA9"],
        name="EMA 9"
    ))

    fig.add_trace(go.Scatter(
        x=chart.index,
        y=chart["EMA21"],
        name="EMA 21"
    ))

    fig.update_layout(
        title=f"{selected['Coin']} - 24 Hour Chart",
        height=400,
        template="plotly_dark"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ---------------- TELEGRAM ----------------

    st.subheader("📲 Telegram Trade Alert")

    if st.button("Send Best Candidate to Telegram"):

        if not candidates:
            st.warning("No BUY candidate to send.")
        else:
            best = candidates[0]

            try:
                token = st.secrets["TELEGRAM_BOT_TOKEN"]
                chat_id = st.secrets["TELEGRAM_CHAT_ID"]

                message = (
                    "VIVEK AI TRADER\n\n"
                    f"Coin: {best['Coin']}\n"
                    f"Signal: {best['Signal']}\n"
                    f"Score: {best['Score']}/100\n"
                    f"Entry: ₹{best['Entry (INR)']:.4f}\n"
                    f"Stop Loss: ₹{best['Stop Loss (INR)']:.4f}\n"
                    f"Target: ₹{best['Target (INR)']:.4f}\n"
                    f"Allocation: ₹{best['Allocation (INR)']:.2f}\n"
                    f"Quantity: {best['Quantity']:.8f}\n\n"
                    "Manual approval required. "
                    "Not guaranteed financial advice."
                )

                response = requests.post(
                    f"https://api.telegram.org/bot{token}/sendMessage",
                    data={
                        "chat_id": chat_id,
                        "text": message
                    },
                    timeout=15
                )

                response.raise_for_status()

                st.success("Telegram alert sent!")

            except Exception as error:
                st.error(
                    "Telegram is not connected. "
                    f"Check your Secrets settings: {error}"
                )

    st.caption(
        "Market data: CoinGecko. "
        "Indicator-based scanner, not machine-learning prediction."
    )

else:
    st.info(
        "Press Scan Crypto Market to find and rank trading opportunities."
    )

st.divider()

st.warning(
    "IMPORTANT: This scanner uses aggregated INR market data and "
    "short-term price indicators. It does not verify prices or "
    "liquidity on your particular exchange. Check the exchange order "
    "book, fees and spread before trading. No automatic orders are placed."
)
