import pandas as pd
import numpy as np
import yfinance as yf
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import shap

class predict:
    def __init__(self,model):
        self.analyzer = SentimentIntensityAnalyzer()
        self.NEWS_API_KEY="9b052ed37eea4cec8641caa87d48f853"
        self.model = model
        self.explainer = shap.Explainer(model)
        
    def get_news_sentiment(self,company):
        queries = [
            f"{company} stock",
            f"{company} earnings",
            f"{company} revenue",
            f"{company} profit",
            f"{company} loss",
            f"{company} rating"
        ]

        sentiments = []
        for q in queries:
            try:
                url = "https://newsapi.org/v2/everything"
                params = {
                    "q": q,
                    "sortBy": "publishedAt",
                    "language": "en",
                    "pageSize": 5,
                    "apiKey": self.NEWS_API_KEY
                }
                r = requests.get(url, params=params, timeout=10)
                r.raise_for_status()
                data = r.json()
                
                for article in data.get("articles", []):
                    text = article["title"] + ". " + (article.get("description") or "")
                    if any(k in text.lower() for k in ["stock", "revenue", "profit", "loss", "earnings"]):
                        s = self.analyzer.polarity_scores(text)["compound"]
                        print(f"News sentiment for '{text}': {s}")
                        sentiments.append(s)
                        
            except Exception as e:
                print("News fetch error:", e)

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