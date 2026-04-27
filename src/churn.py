import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def prepare_churn_data(rfm):
    df = rfm.copy()

    # --- Realistic churn logic (multi-condition) ---
    df['Churn'] = np.where(
        (df['Recency'] > df['Recency'].quantile(0.7)) & 
        (df['Frequency'] < df['Frequency'].median()) &
        (df['Monetary'] < df['Monetary'].median()),
        1, 0
    )

    # --- Features (IMPORTANT: remove direct leakage) ---
    X = df[['Frequency', 'Monetary']]   # Removed Recency ❗
    y = df['Churn']

    return X, y, df


def train_churn_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=5,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    report = classification_report(y_test, preds, output_dict=True)

    return model, report


def predict_churn(model, X):
    probs = model.predict_proba(X)[:, 1]
    return probs