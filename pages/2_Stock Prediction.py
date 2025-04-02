import os
import logging
import warnings
import requests
import numpy as np
import pandas as pd
import datetime
import streamlit as st
import plotly.graph_objects as go
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential, load_model # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" 

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
load_dotenv()

st.set_page_config(page_title="StockUP", layout="wide", page_icon=":chart_with_upwards_trend:")
image_path = os.path.join(os.path.dirname(__file__), "../images/stockuplogo.png")
st.sidebar.image(image_path, use_container_width=True)
css = """
<style>
    [data-testid="stHeader"] { background:#000000; }
    .stApp { background: #000000; }
    [data-testid="stSidebarContent"] { background:#000000; }

"""
st.markdown(css, unsafe_allow_html=True)

#Upstox Api Fetch
def fetch_upstox_data(instrument_key, start_date, end_date):
    headers = {
        "Authorization": f"Bearer {os.getenv('ACCESS_TOKEN')}",
        "Accept": "application/json"
    }
    url = f"https://api.upstox.com/v2/historical-candle/{instrument_key}/day/{end_date}/{start_date}"
    logging.info(f"Fetching data from: {url}")
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json().get("data", {}).get("candles", [])
    if not data:
        raise ValueError("No stock data returned for the given date range.")
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", "open_interest"])
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    df.sort_values("timestamp", inplace=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].dropna()
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["day_gap"] = df["timestamp"].diff().dt.days
    return df

def forward_fill_daily(df, date_col="Date"):
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    full_dates = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
    df = df.reindex(full_dates, method="ffill")
    df.index.name = date_col
    return df.reset_index()

def create_sequences(data_arr, window=60):
    X, y = [], []
    for i in range(window, len(data_arr)):
        X.append(data_arr[i-window:i, :])
        y.append(data_arr[i, 0])
    return np.array(X), np.array(y)

#FinBERT Sentiment Functions
def finbert_analysis(text):
    if not text.strip():
        return {"Positive": 0.0, "Neutral": 0.0, "Negative": 0.0}, "Neutral"
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model_finbert(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    scores = {LABELS[i]: round(probs[0][i].item(), 4) for i in range(len(LABELS))}
    label = LABELS[torch.argmax(probs).item()]
    return scores, label

def combine_scores(scores1, scores2, weight1=0.5, weight2=0.5):
    return {label: round(scores1.get(label, 0)*weight1 + scores2.get(label, 0)*weight2, 4)
            for label in LABELS}

def analyze_sentiment(text, pdf_url=None, subject=""):
    desc_scores, _ = finbert_analysis(text)
    pdf_scores = {"Positive": 0.0, "Neutral": 0.0, "Negative": 0.0}
    combined_scores = desc_scores.copy()
    if pdf_url and not pdf_url.lower().endswith('.xml') and pdf_url != "No Attachment":
        pdf_text = "" 
        pdf_scores, _ = finbert_analysis(pdf_text)
        combined_scores = combine_scores(desc_scores, pdf_scores, weight1=0.5, weight2=0.5)
    keyword_score = 0.0
    for phrase in POSITIVE_PHRASES:
        if phrase.lower() in text.lower():
            keyword_score += 0.2
    for phrase in NEGATIVE_PHRASES:
        if phrase.lower() in text.lower():
            keyword_score -= 0.2
    for phrase in COLLABORATION_PHRASES:
        if phrase.lower() in text.lower():
            keyword_score += 0.7
    if "partnership" in subject.lower() or "collaboration" in subject.lower():
        keyword_score += 0.7
    adjusted_positive = combined_scores["Positive"] + keyword_score
    adjusted_positive = max(0, min(1, adjusted_positive))
    adjusted_scores = combined_scores.copy()
    adjusted_scores["Positive"] = adjusted_positive
    if (adjusted_scores["Positive"] > adjusted_scores["Negative"] and
        adjusted_scores["Positive"] > adjusted_scores["Neutral"]):
        final_sentiment = "Positive"
    elif adjusted_scores["Negative"] > adjusted_scores["Neutral"]:
        final_sentiment = "Negative"
    else:
        final_sentiment = "Neutral"
    return final_sentiment, adjusted_scores

def sentiment_to_numeric(sentiment):
    return {"Positive": 1, "Neutral": 0, "Negative": -1}.get(sentiment, 0)

def aggregate_daily_sentiment(announcements):
    records = []
    for ann in announcements:
        final_sentiment, _ = analyze_sentiment(ann["Details"], ann["Attachment Link"], ann["Subject"])
        numeric = sentiment_to_numeric(final_sentiment)
        try:
            dt_obj = datetime.datetime.strptime(ann["Broadcast Date"], "%d-%b-%Y %H:%M:%S").date()
        except:
            dt_obj = None
        if dt_obj:
            records.append({"date": dt_obj, "sentiment": numeric})
    if records:
        df_sent = pd.DataFrame(records)
        return df_sent.groupby("date")["sentiment"].mean().reset_index()
    else:
        return pd.DataFrame(columns=["date", "sentiment"])

def fetch_news_from_mediastack(query, from_date, to_date, limit=100):
    if not query:
        st.warning("No 'short_name' provided. Returning fallback news.")
        return [{
            "Symbol": "FALLBACK",
            "Company Name": "Sample Company",
            "Subject": "No short_name Provided",
            "Details": "No short_name was provided, so we are returning sample news.",
            "Attachment Link": "No Attachment",
            "Broadcast Date": datetime.datetime.now().strftime("%d-%b-%Y %H:%M:%S")
        }]
    MEDIASTACK_API_KEY = os.getenv("MEDIASTACK_API_KEY")
    if not MEDIASTACK_API_KEY:
        st.error("Mediastack API key not found! Returning fallback news.")
        return [{
            "Symbol": query.upper(),
            "Company Name": f"{query.upper()}",
            "Subject": "Missing API Key",
            "Details": "API key is missing.",
            "Attachment Link": "No Attachment",
            "Broadcast Date": datetime.datetime.now().strftime("%d-%b-%Y %H:%M:%S")
        }]
    try:
        from_date_iso = datetime.datetime.strptime(from_date, "%d-%m-%Y").strftime("%Y-%m-%d")
        to_date_iso = datetime.datetime.strptime(to_date, "%d-%m-%Y").strftime("%Y-%m-%d")
        base_url = "https://api.mediastack.com/v1/news"
        params = {
            "access_key": MEDIASTACK_API_KEY,
            "keywords": query,
            "countries": "in",
            "languages": "en",
            "date": f"{from_date_iso},{to_date_iso}",
            "limit": limit
        }
        response = requests.get(base_url, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()
        articles = data.get("data", [])
        if not articles:
            st.warning(f"No articles found for '{query}'. Returning fallback news.")
            return [{
                "Symbol": query.upper(),
                "Company Name": f"{query.upper()}",
                "Subject": "No Articles Found",
                "Details": "No articles found. Sample news returned.",
                "Attachment Link": "No Attachment",
                "Broadcast Date": datetime.datetime.now().strftime("%d-%b-%Y %H:%M:%S")
            }]
        results = []
        for article in articles:
            title = article.get("title", "")
            description = article.get("description", "")
            details = f"{title} {description}".strip() or "No details"
            published_at = article.get("published_at", "")
            try:
                dt = datetime.datetime.fromisoformat(published_at)
            except Exception:
                dt = datetime.datetime.now()
            broadcast_str = dt.strftime("%d-%b-%Y %H:%M:%S")
            results.append({
                "Symbol": query.upper(),
                "Company Name": f"{query.upper()}",
                "Subject": title,
                "Details": details,
                "Attachment Link": "No Attachment",
                "Broadcast Date": broadcast_str
            })
        return results
    except requests.exceptions.RequestException as e:
        st.error(f"Mediastack API request failed: {e}")
        st.warning("Returning fallback news.")
        return [{
            "Symbol": query.upper(),
            "Company Name": f"{query.upper()}",
            "Subject": "API Error",
            "Details": "Error calling Mediastack. Using fallback news.",
            "Attachment Link": "No Attachment",
            "Broadcast Date": datetime.datetime.now().strftime("%d-%b-%Y %H:%M:%S")
        }]
    except ValueError as ve:
        st.error(f"Invalid date format: {ve}")
        st.warning("Returning fallback news.")
        return [{
            "Symbol": query.upper(),
            "Company Name": f"{query.upper()}",
            "Subject": "Date Parsing Error",
            "Details": "Date parsing error. Using fallback news.",
            "Attachment Link": "No Attachment",
            "Broadcast Date": datetime.datetime.now().strftime("%d-%b-%Y %H:%M:%S")
        }]

#FinBERT Setup
MODEL_NAME = "ProsusAI/finbert"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model_finbert = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
model_finbert.eval()
LABELS = ["Negative", "Neutral", "Positive"]

POSITIVE_PHRASES = [
    "entered into an agreement", "signed a deal", "expanding operations",
    "secured funding", "growth", "positive outlook", "new partnership",
    "major retail chain", "supply contract", "revenue growth", "acquisition",
    "partners with", "collaboration", "partnership", "joint venture"
]
NEGATIVE_PHRASES = [
    "facing legal action", "terminated agreement", "filed for bankruptcy",
    "declining revenue", "regulatory issues", "losses reported"
]
COLLABORATION_PHRASES = [
    "collaboration", "partners with", "partnered with", "collaborates",
    "collaborated", "extends partnership"
]

#LSTM+FinBERT Model
def run_lstm_finbert_upstox(instrument_key, short_name, start_date, end_date, window_size=60):
    df = fetch_upstox_data(instrument_key, start_date, end_date)
    df["Date"] = df["timestamp"].dt.date
    df_price = df[["Date", "close"]].copy()
    df_price = forward_fill_daily(df_price, date_col="Date")
    df_price.sort_values("Date", inplace=True)
    df_price.reset_index(drop=True, inplace=True)
    df_price.rename(columns={"close": "Actual"}, inplace=True)
    
    newsdata_ann = fetch_news_from_mediastack(
        query=short_name,
        from_date="01-01-2020",
        to_date=datetime.date.today().strftime("%d-%m-%Y"),
        limit=100
    )
    # ----------------- Expander for News Articles -----------------
    # with st.expander("See news articles here", expanded=False):
    #     for article in newsdata_ann:
    #         final_sent, scores = analyze_sentiment(article["Details"], article["Attachment Link"], article["Subject"])
    #         st.markdown(f"**{article['Subject']}**")
    #         st.write(article["Details"])
    #         st.write(f"Sentiment: {final_sent} (scores: {scores})")
    #         st.write(f"Source: {article['Company Name']} ({article['Broadcast Date']})")
    #         st.write("---")
    
    if not newsdata_ann:
        newsdata_ann = [{
            "Symbol": instrument_key,
            "Company Name": "Sample Company",
            "Subject": "Fallback News",
            "Details": "Sample press release with positive outlook and secured funding.",
            "Attachment Link": "No Attachment",
            "Broadcast Date": "20-Feb-2025 16:00:02"
        }]
    daily_sentiment_df = aggregate_daily_sentiment(newsdata_ann)
    daily_sentiment_df["date"] = pd.to_datetime(daily_sentiment_df["date"])
    
    # Merge sentiment with price
    df_merged = pd.merge(df_price, daily_sentiment_df, left_on="Date", right_on="date", how="left")
    df_merged["sentiment"].fillna(0, inplace=True)
    
    # Prepare features of Actual price and sentiment
    features = df_merged[["Actual", "sentiment"]].values.astype(float)
    scaler_finbert = MinMaxScaler(feature_range=(0, 1))
    features[:, 0] = scaler_finbert.fit_transform(features[:, 0].reshape(-1, 1)).ravel()
    
    X_all, y_all = create_sequences(features, window_size)
    X_all = X_all.reshape((X_all.shape[0], window_size, 2))
    
    train_size = int(len(features) * 0.8)
    train_data = features[:train_size]
    test_data = features[train_size:]
    X_train, y_train = create_sequences(train_data, window_size)
    X_test, y_test = create_sequences(test_data, window_size)
    X_train = X_train.reshape((X_train.shape[0], window_size, 2))
    X_test = X_test.reshape((X_test.shape[0], window_size, 2))
    
    with st.spinner("Model is running..."):
        model = load_model("improved_lstm_finbert_model_final.keras", compile=False)
        predictions = model.predict(X_all)
    predictions = scaler_finbert.inverse_transform(predictions)
    df_merged["Predicted_FinBERT"] = np.nan
    df_merged.loc[window_size:window_size+len(predictions)-1, "Predicted_FinBERT"] = predictions.ravel()
    
    # model.save("improved_lstm_finbert_model_final.keras")
    return df_merged, scaler_finbert, model

# ----------------- LSTM-Only Prediction -----------------
def load_and_predict_lstm_only(instrument_key, start_date, end_date, window_size=60):
    df = fetch_upstox_data(instrument_key, start_date, end_date)
    df["Date"] = df["timestamp"].dt.date
    df_price = df[["Date", "close"]].copy()
    df_price = forward_fill_daily(df_price, date_col="Date")
    df_price.sort_values("Date", inplace=True)
    df_price.reset_index(drop=True, inplace=True)
    df_price.rename(columns={"close": "Actual"}, inplace=True)
    
    dataset = df_price["Actual"].values.reshape(-1, 1)
    scaler_only = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler_only.fit_transform(dataset)
    X_all, _ = create_sequences(scaled_data, window_size)
    X_all = X_all.reshape((X_all.shape[0], window_size, 1))
    
    with st.spinner("Model is running..."):
        model_only = load_model("improved_lstm_only_model_final.keras", compile=False)
        predictions = model_only.predict(X_all)
    predictions = scaler_only.inverse_transform(predictions)
    
    df_price["Predicted"] = np.nan
    df_price.loc[window_size:window_size+len(predictions)-1, "Predicted"] = predictions.ravel()
    return df_price, scaler_only, model_only

#Model Comparison & Forecasting
def compare_models(instrument_key, short_name, start_date, end_date, window_size=60):
    # LSTM-only predictions
    df_lstm_only, scaler_only, model_only = load_and_predict_lstm_only(instrument_key, start_date, end_date, window_size)
    
    # LSTM+FinBERT predictions
    df_lstm_finbert, scaler_finbert, model_finbert = run_lstm_finbert_upstox(instrument_key, short_name, start_date, end_date, window_size)
    
    df_compare = pd.merge(df_lstm_only[["Date", "Actual", "Predicted"]],
                          df_lstm_finbert[["Date", "Predicted_FinBERT"]],
                          on="Date", how="outer")
    df_compare.sort_values("Date", inplace=True)
    df_compare.reset_index(drop=True, inplace=True)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_compare["Date"],
        y=df_compare["Actual"],
        mode="lines",
        name="Actual",
        line=dict(color="blue")
    ))
    fig.add_trace(go.Scatter(
        x=df_compare["Date"],
        y=df_compare["Predicted"],
        mode="lines",
        name="Predicted (LSTM Only)",
        line=dict(color="red")
    ))
    fig.add_trace(go.Scatter(
        x=df_compare["Date"],
        y=df_compare["Predicted_FinBERT"],
        mode="lines",
        name="Predicted (LSTM+FinBERT)",
        line=dict(color="green", dash="dash")
    ))
    fig.update_layout(
        title="Model Comparison: Actual vs. Predicted Prices",
        xaxis=dict(type="date", tickangle=45),
        yaxis_title="Price",
        legend=dict(x=0, y=1, bgcolor="rgba(255,255,255,0)")
    )
    st.plotly_chart(fig, use_container_width=True)
    
    total = len(df_lstm_only)
    train_size = int(total * 0.8)
    test_start_idx = max(train_size, window_size)
    df_test_only = df_lstm_only.iloc[test_start_idx:].dropna(subset=["Predicted"])
    mae_only = mean_absolute_error(df_test_only["Actual"], df_test_only["Predicted"])
    rmse_only = np.sqrt(mean_squared_error(df_test_only["Actual"], df_test_only["Predicted"]))
    r2_only = r2_score(df_test_only["Actual"], df_test_only["Predicted"])
    
    total2 = len(df_lstm_finbert)
    train_size2 = int(total2 * 0.8)
    test_start_idx2 = max(train_size2, window_size)
    df_test_finbert = df_lstm_finbert.iloc[test_start_idx2:].dropna(subset=["Predicted_FinBERT"])
    mae_finbert = mean_absolute_error(df_test_finbert["Actual"], df_test_finbert["Predicted_FinBERT"])
    rmse_finbert = np.sqrt(mean_squared_error(df_test_finbert["Actual"], df_test_finbert["Predicted_FinBERT"]))
    r2_finbert = r2_score(df_test_finbert["Actual"], df_test_finbert["Predicted_FinBERT"])
    
    st.markdown("<h3 style='text-align: center;'>Model Metrics</h3>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;'>LSTM Only</h5>", unsafe_allow_html=True)
    col_spacer1, col1, col2, col3, col_spacer2 = st.columns([1,1,1,1,1])
    with col1:
        st.metric("MAE", f"{mae_only:.2f}")
    with col2:
        st.metric("RMSE", f"{rmse_only:.2f}")
    with col3:
        st.metric("Accuracy", f"{r2_only*100:.2f}%")
    st.write("---")
    st.markdown("<h5 style='text-align: center;'>LSTM+FinBERT</h5>", unsafe_allow_html=True)
    col_spacerA, col4, col5, col6, col_spacerB = st.columns([1,1,1,1,1])
    with col4:
        st.metric("MAE", f"{mae_finbert:.2f}")
    with col5:
        st.metric("RMSE", f"{rmse_finbert:.2f}")
    with col6:
        st.metric("Accuracy", f"{r2_finbert*100:.2f}%")
    
    # Choose the better model based on RMSE (lower is better)
    if rmse_finbert < rmse_only:
        chosen_model_name = "LSTM+FinBERT"
        chosen_model = model_finbert
        chosen_scaler = scaler_finbert
        df_temp = fetch_upstox_data(instrument_key, start_date, end_date)
        df_temp["Date"] = df_temp["timestamp"].dt.date
        df_price_temp = df_temp[["Date", "close"]].copy()
        df_price_temp = forward_fill_daily(df_price_temp, date_col="Date")
        df_price_temp.rename(columns={"close": "Actual"}, inplace=True)
        newsdata_temp = fetch_news_from_mediastack(
            query=short_name,
            from_date="01-01-2020",
            to_date=datetime.date.today().strftime("%d-%m-%Y"),
            limit=100
        )
        daily_sentiment_temp = aggregate_daily_sentiment(newsdata_temp)
        daily_sentiment_temp["date"] = pd.to_datetime(daily_sentiment_temp["date"])
        df_merged_full = pd.merge(df_price_temp, daily_sentiment_temp, left_on="Date", right_on="date", how="left")
        df_merged_full["sentiment"].fillna(0, inplace=True)
        features = df_merged_full[["Actual", "sentiment"]].values.astype(float)
        features[:, 0] = chosen_scaler.transform(features[:, 0].reshape(-1, 1)).ravel()
        last_window = features[-window_size:]
        X_next = last_window.reshape((1, window_size, 2))
    else:
        chosen_model_name = "LSTM Only"
        chosen_model = load_model("improved_lstm_only_model_final.keras", compile=False)
        df_temp = fetch_upstox_data(instrument_key, start_date, end_date)
        df_temp["Date"] = df_temp["timestamp"].dt.date
        df_price_temp = df_temp[["Date", "close"]].copy()
        df_price_temp = forward_fill_daily(df_price_temp, date_col="Date")
        df_price_temp.rename(columns={"close": "Actual"}, inplace=True)
        scaler_only = MinMaxScaler(feature_range=(0, 1))
        dataset_temp = df_price_temp["Actual"].values.reshape(-1, 1)
        scaler_only.fit(dataset_temp)
        last_window = scaler_only.transform(df_price_temp["Actual"].values[-window_size:].reshape(-1, 1))
        X_next = last_window.reshape((1, window_size, 1))
        chosen_scaler = scaler_only
    
    st.write(f"## Chosen Model for Next-Day Forecast: {chosen_model_name}")
    with st.spinner("Model is running..."):
        next_day_scaled = chosen_model.predict(X_next)
    next_day_price = chosen_scaler.inverse_transform(next_day_scaled)[0][0]
    st.write(f"## Predicted Price for {short_name} on the Next Day: **₹{next_day_price:.2f}**")
    
    return df_compare

#Main Function & Streamlit UI
def main():
    st.title("Stock Prediction Model Comparison")
    st.write("""
    This app compares two models on Upstox data:
    - LSTM Only (using only price data)
    - LSTM+FinBERT (using price data and sentiment from news)
    """)
    
    try:
        response = requests.get("https://energisense-server.onrender.com/nse-json")
        response.raise_for_status()
        json_data = response.json()
        if isinstance(json_data, dict):
            json_data = [json_data]
        equities = {
            item.get("name"): {
                "instrument_key": item.get("instrument_key"),
                "short_name": item.get("short_name")
            }
            for item in json_data
            if item.get("instrument_type") == "EQ" and item.get("name")
        }
    except Exception as e:
        st.error("Error fetching NSE JSON data:")
        st.exception(e)
        return
    
    if not equities:
        st.error("No valid symbols found in the JSON data!")
        return
    
    selected_symbol = st.selectbox(
        label="",
        options=list(equities.keys()),
        index=None,
        label_visibility="collapsed",
        placeholder="Please choose an NSE listed stock"
    )
    if not selected_symbol:
        st.info("Please select a stock to proceed.")
        st.stop()
    
    instrument_key = equities[selected_symbol]["instrument_key"]
    short_name = equities[selected_symbol]["short_name"]
    if instrument_key is None:
        st.error("Instrument key not found for selected stock!")
        return
    if isinstance(instrument_key, np.ndarray):
        instrument_key = instrument_key.item()
    
    result = compare_models(instrument_key, short_name, "2018-01-01", datetime.date.today().strftime("%Y-%m-%d"), window_size=60)
    if result is not None:
        pass
        # st.write("Model comparison completed.")

if __name__ == "__main__":
    main()
