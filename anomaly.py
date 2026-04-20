def detect_anomalies(df):
    df = df.copy()

    mean = df["temperature"].mean()
    std = df["temperature"].std()

    df["anomaly"] = (
        (df["temperature"] > mean + 2*std) |
        (df["temperature"] < mean - 2*std)
    )

    return df