import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("credit-risk-model")


DATA = 'data/processed/train_labeled.parquet'
TARGET = 'is_high_risk'


def load():
    df = pd.read_parquet(DATA)

    y = df[TARGET]
    X = df.drop(columns=[TARGET, 'cluster'], errors='ignore')

    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def make_pipe(model, X_sample):
    """
    Build pipeline dynamically based on actual columns
    """
    num_features = [
        col for col in X_sample.columns
        if col.lower() != 'customerid'
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features)
        ],
        remainder='drop'
    )

    return Pipeline(
        steps=[
            ('prep', preprocessor),
            ('clf', model)
        ]
    )


def eval(model_name, model, param_grid):
    X_tr, X_te, y_tr, y_te = load()

    pipe = make_pipe(model, X_tr)

    gs = GridSearchCV(
        pipe,
        param_grid,
        cv=3,
        scoring='roc_auc',
        n_jobs=-1,
        error_score='raise'
    )

    gs.fit(X_tr, y_tr)

    best = gs.best_estimator_
    preds = best.predict(X_te)
    prob = best.predict_proba(X_te)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_te, preds),
        'precision': precision_score(y_te, preds),
        'recall': recall_score(y_te, preds),
        'f1': f1_score(y_te, preds),
        'roc_auc': roc_auc_score(y_te, prob)
    }

    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(gs.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best, 'model')

    print(f"\n{model_name.upper()} RESULTS")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == '__main__':
    eval(
        'lr',
        LogisticRegression(max_iter=1000, class_weight='balanced'),
        {'clf__C': [0.01, 0.1, 1]}
    )

    eval(
        'xgb',
        XGBClassifier(
            eval_metric='logloss',
            scale_pos_weight=5,
            random_state=42
        ),
        {
            'clf__n_estimators': [200, 400],
            'clf__max_depth': [3, 5]
        }
    )
