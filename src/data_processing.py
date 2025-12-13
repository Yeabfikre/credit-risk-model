import pandas as pd, numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
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

def create_proxy_target(rfm_df, random_state=42):
    rfm = rfm_df[['days_since_last','frequency','monetary']].fillna(0)
    rfm_scaled = StandardScaler().fit_transform(rfm)
    km = KMeans(n_clusters=3, random_state=random_state, n_init=10).fit(rfm_scaled)
    rfm_df['cluster'] = km.labels_
    rfm_df['is_high_risk'] = (rfm_df['cluster'] == 0).astype(int)
    return rfm_df

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

    # 4. Run pipeline
    rfm_df = rfm_pipeline.fit_transform(df)

    # 5. Create proxy target labels
    labeled_df = create_proxy_target(rfm_df)
    print("High-risk rate:", labeled_df["is_high_risk"].mean())

    # 6. Save processed features and labeled data
    rfm_df.to_parquet(
        "D:/kifiya AI/credit-risk-model/data/processed/rfm_features.parquet",
        index=False
    )
    labeled_df.to_parquet(
        "D:/kifiya AI/credit-risk-model/data/processed/train_labeled.parquet",
        index=False
    )

    print("RFM features saved successfully")
    print("Labeled training data saved")
