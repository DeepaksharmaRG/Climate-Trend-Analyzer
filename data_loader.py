import pandas as pd
import numpy as np

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

    return df