from fastapi import FastAPI
import pandas as pd
import numpy as np
import xgboost as xgb
from utility import predict
from pydantic import BaseModel
import os

app = FastAPI()

# ---- Response schema ----
class PredictionResponse(BaseModel):
    prediction: float
    explanation: str
    details: dict

def load_model():
    model = xgb.XGBRegressor()
    model.load_model("credit_model.json")
    print("Model loaded successfully")
    return model

model = load_model()
obj = predict(model)

@app.get("/")
def root():
    return {"message": "Credit Score Prediction API"}

@app.get("/predict/{company_name}/{ticker}", response_model=PredictionResponse)
def send(company_name: str, ticker: str):
    try:
        # Step 1: Fetch features
        df = obj.get_realtime_features(company_name, ticker)

        # Step 2: Make prediction
        features = ["Debt/Equity", "Revenue Growth %", "Net Income", "Volatility", "News Sentiment"]
        result = model.predict(df[features])

        # Step 3: Generate explanation
        explanation_text = obj.generate_explanation(df, features)

        # Step 4: Return structured response
        return {
            "prediction": float(result[0]),
            "explanation": explanation_text,
            "details": df.to_dict(orient="records")[0]
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)