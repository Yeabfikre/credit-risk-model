from src.data_processing import create_proxy_target
import pandas as pd

def test_shape():
    df = pd.read_parquet('D:/kifiya AI/credit-risk-model/data/processed/train_labeled.parquet')
    assert 'is_high_risk' in df.columns

def test_risk_rate():
    df = pd.read_parquet('D:/kifiya AI/credit-risk-model/data/processed/train_labeled.parquet')
    assert 0.2 <= df['is_high_risk'].mean() <= 0.7