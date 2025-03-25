import streamlit as st

st.set_page_config(page_title="StockUP", page_icon=":chart_with_upwards_trend:",layout="centered", initial_sidebar_state="collapsed")
 
st.title("Welcome to the Stock Prediction App")
st.write(
    """
    This application demonstrates:
    - Fetching & analyzing corporate announcements using FinBERT for sentiment.
    - Building an LSTM + GRU model to predict stock prices using both price data and sentiment.
    - Visualizing the correlation matrix and model performance metrics.
    """
)
