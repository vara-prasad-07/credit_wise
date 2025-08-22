import pandas as pd
import numpy as np
import yfinance as yf
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import shap
import time
import os
from dotenv import load_dotenv
load_dotenv()
class predict:
    def __init__(self,model):
        self.analyzer = SentimentIntensityAnalyzer()
        self.NEWS_API_KEY=os.environ.get('NEWS_API_KEY')
        self.model = model
        self.explainer = shap.Explainer(model)
        
    def get_news_sentiment(self, company):
        company_clean = company.replace(".", "")
        queries = [
            f"{company_clean} stock",
            f"{company_clean} earnings",
            f"{company_clean} revenue",
            f"{company_clean} profit",
            f"{company_clean} loss",
            f"{company_clean} rating"
        ]

        sentiments = []
        for i, q in enumerate(queries):
            try:
                # Add delay between requests
                if i > 0:
                    time.sleep(1)
                    
                # Use /latest endpoint instead of /news
                url = "https://newsdata.io/api/1/latest"
                params = {
                    "apikey": self.NEWS_API_KEY,
                    "q": q,
                    "language": "en"
                    # Remove page parameter - not needed for initial request
                }
                
                print(f"Requesting: {q}")
                r = requests.get(url, params=params, timeout=10)
                
                print(f"Status: {r.status_code}")
                if r.status_code != 200:
                    print(f"Response: {r.text[:200]}")
                    
                r.raise_for_status()
                data = r.json()

                for article in data.get("results", []):
                    text = article.get("title", "") + ". " + (article.get("description") or "")
                    if any(k in text.lower() for k in ["stock", "revenue", "profit", "loss", "earnings"]):
                        s = self.analyzer.polarity_scores(text)["compound"]
                        print(f"News sentiment for '{text}': {s}")
                        sentiments.append(s)

            except Exception as e:
                print(f"News fetch error for '{q}': {e}")

        return np.mean(sentiments) if sentiments else 0

    
    def get_financial_data(self,ticker):
        tkr = yf.Ticker(ticker)
        info = tkr.info
        
        debt_equity = info.get("debtToEquity", None)
        revenue_growth = info.get("revenueGrowth", None)
        net_income = info.get("netIncomeToCommon", None)

        hist = tkr.history(period="1mo")
        volatility_val = hist["Close"].pct_change().std() * (252**0.5) if not hist.empty else None
        if volatility_val is None:
            volatility = 1
        elif volatility_val > 0.03:
            volatility = 2  # High
        elif volatility_val > 0.015:
            volatility = 1  # Medium
        else:
            volatility = 0  # Low

        return {
            "Debt/Equity": debt_equity,
            "Revenue Growth %": revenue_growth * 100 if revenue_growth else None,
            "Net Income": net_income,
            "Volatility": volatility
        }
        
    def get_realtime_features(self,company, ticker):
        fin = self.get_financial_data(ticker)
        sentiment = self.get_news_sentiment(company)
        row = {
            "Debt/Equity": fin["Debt/Equity"],
            "Revenue Growth %": fin["Revenue Growth %"],
            "Net Income": fin["Net Income"],
            "Volatility": fin["Volatility"],
            "News Sentiment": sentiment
        }
        return pd.DataFrame([row])    
    def generate_explanation(self, instance, feature_names):
        shap_values = self.explainer(instance)
        explanation = []
        base_value = shap_values.base_values[0]
        final_score = shap_values.values[0].sum() + base_value

        for feature, shap_val, value in zip(feature_names, shap_values.values[0], instance.values[0]):
            if abs(shap_val) > 0.01:
                if shap_val > 0:
                    explanation.append(f"{feature} = {value} increased the score by {shap_val:.2f}")
                else:
                    explanation.append(f"{feature} = {value} decreased the score by {abs(shap_val):.2f}")

        return f"Predicted Credit Score: {final_score:.2f}. Key factors: " + "; ".join(explanation)