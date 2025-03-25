import streamlit as st
import pandas as pd
import numpy as np
import datetime, warnings, logging, os, torch, requests
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="StockUP", layout="wide", page_icon=":chart_with_upwards_trend:")

def price_within_threshold_accuracy(actual, predicted, threshold=0.05):
    if len(actual) == 0:
        return 0.0
    correct = 0
    for i in range(len(actual)):
        if actual[i] == 0:
            continue
        error_pct = abs(predicted[i] - actual[i]) / abs(actual[i])
        if error_pct <= threshold:
            correct += 1
    return (correct / len(actual)) * 100

def run_lstm_finbert_pipeline(stock_symbol, instrument_key, short_name):
    # 1. GET CONFIG
    def get_config_local(stock_symbol, instrument_key, short_name):
        current_date = datetime.date.today()
        return {
            "instrument_key": instrument_key,
            "stock_symbol": stock_symbol,
            "short_name": short_name,
            "start_date_stock": "2020-01-01",
            "end_date_stock": current_date.strftime("%Y-%m-%d"),
            "start_date_ann": "01-01-2020",
            "end_date_ann": current_date.strftime("%d-%m-%Y"),
            "window_size": 25,
            "epochs": 50,
            "batch_size": 16,
            "learning_rate": 1e-4
        }
    config_local = get_config_local(stock_symbol, instrument_key, short_name)
    # 2. SET UP FINBERT
    # ---------------------------
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

    # ---------------------------
    # 3. MEDIASTACK NEWS FUNCTION
    # ---------------------------
    def fetch_news_from_mediastack(query, from_date, to_date, limit=100):
        if not query:
            st.warning("No 'short_name' found. Returning sample fallback news.")
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
            st.error("Mediastack API key not found! Returning sample fallback news.")
            return [{
                "Symbol": query.upper(),
                "Company Name": f"{query.upper()}",
                "Subject": "Missing API Key",
                "Details": "Could not fetch news because the API key is missing.",
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
                st.warning(f"No articles found for '{query}'. Returning sample fallback news.")
                return [{
                    "Symbol": query.upper(),
                    "Company Name": f"{query.upper()}",
                    "Subject": "No Articles Found",
                    "Details": "No articles were found for this query. Sample news returned.",
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
            st.warning("Returning sample fallback news instead of throwing error.")
            return [{
                "Symbol": query.upper(),
                "Company Name": f"{query.upper()}",
                "Subject": "API Error",
                "Details": "An error occurred calling Mediastack. Using fallback news.",
                "Attachment Link": "No Attachment",
                "Broadcast Date": datetime.datetime.now().strftime("%d-%b-%Y %H:%M:%S")
            }]
        except ValueError as ve:
            st.error(f"Invalid date format or other ValueError: {ve}")
            st.warning("Returning sample fallback news.")
            return [{
                "Symbol": query.upper(),
                "Company Name": f"{query.upper()}",
                "Subject": "Date Parsing Error",
                "Details": "Invalid date format or parsing error occurred. Using fallback news.",
                "Attachment Link": "No Attachment",
                "Broadcast Date": datetime.datetime.now().strftime("%d-%b-%Y %H:%M:%S")
            }]

    # ---------------------------
    # 4. SENTIMENT ANALYSIS
    # ---------------------------
    def sentiment_to_numeric(sentiment):
        return {"Positive": 1, "Neutral": 0, "Negative": -1}.get(sentiment, 0)

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
            pdf_text = ""  # Not actually extracting PDF
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

    # -------------- FETCH NEWS VIA MEDIASTACK --------------
    newsdata_ann = fetch_news_from_mediastack(
        query=config_local["short_name"],
        from_date=config_local["start_date_ann"],
        to_date=config_local["end_date_ann"],
        limit=100
    )
    if not newsdata_ann:
        newsdata_ann = [{
            "Symbol": config_local["short_name"],
            "Company Name": "Sample Company",
            "Subject": "Fallback News",
            "Details": "Sample press release with positive outlook and secured funding.",
            "Attachment Link": "No Attachment",
            "Broadcast Date": "20-Feb-2025 16:00:02"
        }]
    daily_sentiment_df = aggregate_daily_sentiment(newsdata_ann)
    daily_sentiment_df["date"] = pd.to_datetime(daily_sentiment_df["date"])

    # -------------- FETCH STOCK DATA FROM UPSTOX --------------
    def fetch_upstox_data():
        headers_stock = {
            "Authorization": f"Bearer {os.getenv('ACCESS_TOKEN')}",
            "Accept": "application/json"
        }
        ohlc_url = (
            f"https://api.upstox.com/v2/historical-candle/{config_local['instrument_key']}/day/"
            f"{config_local['end_date_stock']}/{config_local['start_date_stock']}"
        )
        logging.info(f"Upstox URL: {ohlc_url}")
        response = requests.get(ohlc_url, headers=headers_stock)
        if response.status_code == 200:
            data = response.json().get("data", {}).get("candles", [])
            if data:
                df_stock = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", "open_interest"])
                df_stock["timestamp"] = pd.to_datetime(df_stock["timestamp"]).dt.tz_localize(None)
                df_stock.sort_values("timestamp", ascending=True, inplace=True)
                return df_stock
            else:
                raise ValueError("No stock data available for the given date range.")
        else:
            raise ValueError(f"Error fetching stock data: {response.status_code} {response.text}")

    df_stock = fetch_upstox_data()
    df_stock["date"] = pd.to_datetime(df_stock["timestamp"].dt.date)

    # Merge with daily sentiment
    df_merged = pd.merge(df_stock, daily_sentiment_df, how="left", on="date")
    df_merged["sentiment"].fillna(0, inplace=True)
    df_merged["news_flag"] = np.where(df_merged["sentiment"] != 0, 1, 0)
    df_merged.sort_values("timestamp", ascending=True, inplace=True)
    df_merged["hl_range"] = (df_merged["high"] - df_merged["low"]) / df_merged["close"]
    df_merged["log_return"] = np.log(df_merged["close"] / df_merged["close"].shift(1)).fillna(0)
    df_merged["ma5"] = df_merged["close"].rolling(window=5).mean().fillna(method="bfill")
    df_merged["volatility"] = df_merged["log_return"].rolling(window=5).std().fillna(0)
    df_merged["sent_mom"] = df_merged["sentiment"].rolling(window=3).mean().fillna(0)

    from sklearn.preprocessing import MinMaxScaler
    scalers = {
        "close": MinMaxScaler(),
        "volume": MinMaxScaler(),
        "hl_range": MinMaxScaler(),
        "ma5": MinMaxScaler(),
        "volatility": MinMaxScaler()
    }
    for col, scaler in scalers.items():
        df_merged[f"{col}_scaled"] = scaler.fit_transform(df_merged[[col]])

    feature_cols = [
        "close_scaled",
        "sentiment",
        "news_flag",
        "hl_range_scaled",
        "volume_scaled",
        "ma5_scaled",
        "volatility_scaled",
        "sent_mom",
        "log_return"
    ]
    df_merged.reset_index(drop=True, inplace=True)
    all_dates = df_merged["timestamp"].values
    data_features = df_merged[feature_cols].values

    def create_sequences(data, window_size, dates):
        X, y, label_dates = [], [], []
        for i in range(len(data) - window_size):
            X.append(data[i : i + window_size])
            y.append(data[i + window_size, 0])  # next day's close_scaled
            label_dates.append(dates[i + window_size])
        return np.array(X), np.array(y), np.array(label_dates)

    X_seq, y_seq, label_dates = create_sequences(data_features, config_local["window_size"], all_dates)
    if len(X_seq) == 0:
        st.error("Not enough data to form sequences. Please check your data or date ranges.")
        # Stop execution within this function to avoid calling model.fit with empty data
        st.stop()

    # Train, val, test split
    total_samples = len(X_seq)
    train_size = int(total_samples * 0.70)
    val_size = int(total_samples * 0.15)
    X_train, y_train = X_seq[:train_size], y_seq[:train_size]
    X_val, y_val = X_seq[train_size:train_size + val_size], y_seq[train_size:train_size + val_size]
    X_test, y_test = X_seq[train_size + val_size:], y_seq[train_size + val_size:]
    test_label_dates = label_dates[train_size + val_size:]

    if X_train.shape[0] == 0 or X_val.shape[0] == 0 or X_test.shape[0] == 0:
        st.error("Train/Val/Test sets have zero samples. Cannot train the model.")
        st.stop()

    # -------------- BUILD AND TRAIN THE MODEL --------------
    # Comment out the model building and training code:
    
    # model = Sequential([
    #     Bidirectional(LSTM(128, return_sequences=True, input_shape=(config_local["window_size"], len(feature_cols)))),
    #     Dropout(0.2),
    #     GRU(64, return_sequences=False),
    #     Dropout(0.2),
    #     Dense(64, activation='relu'),
    #     Dense(1)
    # ])

    # optimizer = Adam(learning_rate=1e-5)
    # model.compile(
    #     optimizer=optimizer,
    #     loss='mean_squared_error',
    #     metrics=['mean_absolute_percentage_error']
    # )

    # early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    # history = model.fit(
    #     X_train, y_train,
    #     epochs=100,
    #     batch_size=32,
    #     validation_data=(X_val, y_val),
    #     callbacks=[early_stop],
    #     verbose=1
    # )
    
    # Load the saved model instead
    model = load_model("D:\\Clg coding\\My Projects\\StockUP Project\\improved_lstm_finbert_model_final.keras")
    print("Model loaded successfully!")

    # Comment out training history plot as 'history' is not available
    # fig_loss, ax_loss = plt.subplots(figsize=(10, 5))
    # ax_loss.plot(history.history['loss'], label='Train Loss')
    # ax_loss.plot(history.history['val_loss'], label='Validation Loss')
    # ax_loss.set_title('Training vs Validation Loss')
    # ax_loss.set_xlabel('Epoch')
    # ax_loss.set_ylabel('Loss')
    # ax_loss.legend()
    # st.pyplot(fig_loss)


    # Evaluate the model
    test_loss, test_mape = model.evaluate(X_test, y_test, verbose=0)
    predictions_scaled = model.predict(X_test).reshape(-1, 1)

    # Inverse transform predictions and actual values
    predicted_prices = scalers["close"].inverse_transform(predictions_scaled)
    actual_prices = scalers["close"].inverse_transform(y_test.reshape(-1, 1))

    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
    mae = mean_absolute_error(actual_prices, predicted_prices)
    mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100

    st.write("### Model Performance Metrics")
    st.write(f"RMSE: {rmse:.4f}")
    st.write(f"MAE: {mae:.4f}")
    st.write(f"MAPE: {mape:.2f}%")

    df_compare = pd.DataFrame({
        "Date": test_label_dates,
        "Actual": actual_prices.flatten(),
        "Predicted": predicted_prices.flatten()
    })
    df_compare["Difference"] = df_compare["Predicted"] - df_compare["Actual"]
    df_compare["Diff %"] = (df_compare["Difference"] / df_compare["Actual"]) * 100
    df_compare["Date"] = pd.to_datetime(df_compare["Date"])
    df_compare.sort_values("Date", inplace=True)
    # st.write("#### Last 10 Predictions with Difference:")
    # st.write(df_compare.tail(10))

    fig_pred, ax_pred = plt.subplots(figsize=(10, 5))
    ax_pred.plot(df_compare["Date"], df_compare["Actual"], label="Actual Prices", color="blue")
    ax_pred.plot(df_compare["Date"], df_compare["Predicted"], label="Predicted Prices", color="red")
    ax_pred.set_title("Test Set: Actual vs Predicted Prices")
    ax_pred.set_xlabel("Date")
    ax_pred.set_ylabel("Price (INR)")
    ax_pred.legend()
    plt.xticks(rotation=45)
    from matplotlib.dates import DateFormatter
    ax_pred.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    plt.tight_layout()
    st.pyplot(fig_pred)

    # Optionally, comment out model saving if not needed:
    # model.save("improved_lstm_finbert_model_final.keras")

    # Forecast next day using the saved model
    last_sequence = data_features[-config_local["window_size"]:]
    last_sequence = np.expand_dims(last_sequence, axis=0)
    tomorrow_pred_scaled = model.predict(last_sequence)
    predicted_tomorrow = scalers["close"].inverse_transform(tomorrow_pred_scaled)
    forecast_date = pd.to_datetime(df_stock["timestamp"].iloc[-1]).date() + datetime.timedelta(days=1)

    return forecast_date.strftime('%Y-%m-%d'), predicted_tomorrow[0][0]

def main():
    st.title("Stock Prediction Model")

    url = "https://energisense-server.onrender.com/nse-json"
    try:
        response = requests.get(url)
        response.raise_for_status()
        json_data = response.json()
        if isinstance(json_data, dict):
            json_data = [json_data]

        # Build a dictionary mapping name to instrument key & short_name
        equities = {
            item.get("name"): {
                "instrument_key": item.get("instrument_key"),
                "short_name": item.get("short_name")
            }
            for item in json_data
            if item.get("instrument_type") == "EQ" and item.get("name")
        }

        if not equities:
            st.error("No valid symbols found in the JSON data!")
            return

        selected_symbol = st.selectbox(
            "Choose an option",
            options=list(equities.keys()),
            placeholder='Enter or Choose a NSE listed Stock'
        )
        if not selected_symbol:
            st.info("Please select a stock from the dropdown.")
            return

        instrument_key = equities[selected_symbol]["instrument_key"]
        short_name = equities[selected_symbol]["short_name"]
        if instrument_key is None:
            st.error("Instrument key not found for selected stock!")
            return
        if isinstance(instrument_key, np.ndarray):
            instrument_key = instrument_key.item()

        # Run the pipeline
        result = run_lstm_finbert_pipeline(selected_symbol, instrument_key, short_name)
        if result is not None:
            forecast_date, predicted_price = result
            st.write(f"### Predicted closing price for {forecast_date}: {predicted_price:.2f}")

    except requests.exceptions.RequestException as e:
        st.error("Error fetching NSE JSON data:")
        st.exception(e)

if __name__ == "__main__":
    main()

