def get_trends(df):
    monthly = df.resample("M").mean(numeric_only=True)
    yearly = df.resample("Y").mean(numeric_only=True)
    return monthly, yearly