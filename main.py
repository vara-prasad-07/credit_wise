from fastapi import FastAPI
import pandas as pd
import numpy as np
import xgboost as xgb
from utility import predict
from pydantic import BaseModel

app = FastAPI()

# ---- Response schema ----
class PredictionResponse(BaseModel):
    prediction: float
    explanation: str
    details:dict

def load_model():
    model = xgb.XGBRegressor()
    model.load_model("credit_model.json")
    print("Model loaded successfully")
    return model

model = load_model()
obj = predict(model)

@app.get("/predict/{company_name}/{ticker}", response_model=PredictionResponse)
def send(company_name: str, ticker: str):
    # Step 1: Fetch features
    df = obj.get_realtime_features(company_name, ticker)

    # Step 2: Make prediction
    features = ["Debt/Equity", "Revenue Growth %", "Net Income", "Volatility", "News Sentiment"]
    result = model.predict(df[features])

    # Step 3: Generate explanation (assuming obj has method for it)
    explanation_text = obj.generate_explanation(df, features)

    # Step 4: Return structured response
    return {
        "prediction": float(result[0]),
        "explanation": explanation_text,
        "details": df.to_dict(orient="records")[0]
    }
