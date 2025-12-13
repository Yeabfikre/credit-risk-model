import pandas as pd, numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xverse.transformer import WOE

# ---------- transformers ----------
class DateParts(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        X['hour']  = X['TransactionStartTime'].dt.hour
        X['day']   = X['TransactionStartTime'].dt.day
        X['month'] = X['TransactionStartTime'].dt.month
        X['year']  = X['TransactionStartTime'].dt.year
        return X

class RFM(BaseEstimator, TransformerMixin):
    def __init__(self, snapshot_date): self.snapshot_date = pd.to_datetime(snapshot_date)
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        X['days_since_last'] = (self.snapshot_date - X.groupby('CustomerId')['TransactionStartTime'].transform('max')).dt.days
        X['frequency'] = X.groupby('CustomerId')['TransactionId'].transform('count')
        X['monetary']  = X.groupby('CustomerId')['Value'].transform('sum')
        # return one row per customer
        return X[['CustomerId','days_since_last','frequency','monetary']].drop_duplicates()

class DtypeCaster(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        cats = ['ChannelId','ProductCategory','CurrencyCode','CountryCode','ProviderId']
        for c in cats: X[c] = X[c].astype('category')
        return X

# ---------- full preprocessing ----------
def build_rfm_pipeline(snapshot_date):
    return Pipeline([
        ('caster', DtypeCaster()),
        ('dates', DateParts()),
        ('rfm', RFM(snapshot_date))
    ])


if __name__ == "__main__":

    # 1. Load raw data
    df = pd.read_csv(
        "D:/kifiya AI/credit-risk-model/data/raw/data.csv",
        parse_dates=["TransactionStartTime"]
    )

    # 2. Define snapshot date
    snapshot_date = df["TransactionStartTime"].max() + pd.Timedelta(days=1)

    # 3. Build pipeline
    rfm_pipeline = build_rfm_pipeline(snapshot_date)

    # 4. Run pipeline → THIS creates the DataFrame
    rfm_df = rfm_pipeline.fit_transform(df)

    # 5. Save processed features
    rfm_df.to_parquet(
        "data/processed/rfm_features.parquet",
        index=False
    )

    print("RFM features saved successfully")


