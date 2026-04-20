import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Climate Analyzer", layout="wide")

st.title("🌍 Climate Trend Analyzer")

# ------------------ DATA GENERATION ------------------
@st.cache_data
def load_data():
    np.random.seed(42)

    dates = pd.date_range(start="2015-01-01", end="2022-12-31")

    temperature = 25 + 10*np.sin(np.linspace(0, 20, len(dates))) + np.random.normal(0, 2, len(dates))
    rainfall = np.abs(np.random.normal(5, 3, len(dates)))
    humidity = 60 + np.random.normal(0, 10, len(dates))

    df = pd.DataFrame({
        "date": dates,
        "temperature": temperature,
        "rainfall": rainfall,
        "humidity": humidity
    })

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df.set_index("date", inplace=True)

    return df

df = load_data()

# ------------------ SIDEBAR FILTER ------------------
st.sidebar.header("📅 Filters")

start_date = st.sidebar.date_input("Start Date", df.index.min().date())
end_date = st.sidebar.date_input("End Date", df.index.max().date())

start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

df_filtered = df[(df.index >= start_date) & (df.index <= end_date)]

# ------------------ FUNCTIONS ------------------

def get_trends(df):
    monthly = df.resample("M").mean(numeric_only=True)
    yearly = df.resample("Y").mean(numeric_only=True)
    return monthly, yearly

def detect_anomalies(df):
    df = df.copy()
    mean = df["temperature"].mean()
    std = df["temperature"].std()

    df["anomaly"] = (
        (df["temperature"] > mean + 2*std) |
        (df["temperature"] < mean - 2*std)
    )
    return df

def forecast(df):
    df = df.reset_index().copy()
    df["time"] = np.arange(len(df))

    X = df[["time"]]
    y = df["temperature"]

    model = LinearRegression()
    model.fit(X, y)

    future = np.arange(len(df), len(df)+30).reshape(-1, 1)
    preds = model.predict(future)

    return preds

# ------------------ TABS ------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Trends",
    "⚠️ Anomalies",
    "🔮 Forecast",
    "📋 Data"
])

# ------------------ TRENDS ------------------
with tab1:
    st.subheader("📊 Climate Trends")

    monthly, yearly = get_trends(df_filtered)

    st.write("Monthly Trends")
    st.line_chart(monthly)

    st.write("Yearly Trends")
    st.line_chart(yearly)

# ------------------ ANOMALIES ------------------
with tab2:
    st.subheader("⚠️ Anomaly Detection")

    df_anomaly = detect_anomalies(df_filtered)
    anomalies = df_anomaly[df_anomaly["anomaly"]]

    st.write(f"Total anomalies: {len(anomalies)}")
    st.dataframe(anomalies)

    fig, ax = plt.subplots()
    ax.plot(df_anomaly.index, df_anomaly["temperature"], label="Temperature")
    ax.scatter(anomalies.index, anomalies["temperature"], label="Anomaly")
    ax.legend()

    st.pyplot(fig)

# ------------------ FORECAST ------------------
with tab3:
    st.subheader("🔮 Temperature Forecast")

    preds = forecast(df_filtered)
    st.line_chart(preds)

# ------------------ DATA ------------------
with tab4:
    st.subheader("📋 Raw Data")
    st.dataframe(df_filtered)