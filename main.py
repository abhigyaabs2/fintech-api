




import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sklearn.ensemble import IsolationForest
import joblib

app = FastAPI(title="Real-Time Transaction Anomaly Detection API")


# -------------------------------------------------------------------
# 1. INITIAL TRAINING (BASELINE USER BEHAVIOR MODEL)
# -------------------------------------------------------------------
np.random.seed(42)

n_samples = 200
past_amounts = np.random.normal(loc=100, scale=20, size=n_samples)
past_hours = np.random.randint(8, 23, size=n_samples)

history_df = pd.DataFrame({
    'Amount': past_amounts,
    'Hour': past_hours
})

baseline_model = IsolationForest(contamination=0.05, random_state=42)
baseline_model.fit(history_df[['Amount', 'Hour']])

joblib.dump(baseline_model, "model.pkl")


# -------------------------------------------------------------------
# 2. REQUEST SCHEMA
# -------------------------------------------------------------------
class Transaction(BaseModel):
    amount: float
    hour: int


class RequestData(BaseModel):
    history: List[Transaction]     # last 5 transactions
    current: Transaction           # current transaction to evaluate


# -------------------------------------------------------------------
# 3. PREDICT ENDPOINT
# -------------------------------------------------------------------
@app.post("/analyze")
def analyze_transactions(data: RequestData):

    # Load the model
    model = joblib.load("model.pkl")

    # Convert last 5 history items to DataFrame
    history_df = pd.DataFrame([{
        "Amount": tx.amount,
        "Hour": tx.hour
    } for tx in data.history])

    # Append current transaction
    current_df = pd.DataFrame([{
        "Amount": data.current.amount,
        "Hour": data.current.hour
    }])

    # Predict only the current transaction
    pred = model.predict(current_df)[0]
    result = "Normal" if pred == 1 else "Anomaly"

    return {
        "current_transaction": {
            "amount": data.current.amount,
            "hour": data.current.hour
        },
        "prediction": result,
        "history_used": len(history_df),
    }







