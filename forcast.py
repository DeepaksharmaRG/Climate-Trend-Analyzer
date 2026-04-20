import numpy as np
from sklearn.linear_model import LinearRegression

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