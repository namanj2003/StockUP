import streamlit as st
import yfinance as yf
import datetime as dt
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import os

st.set_page_config(page_title="StockUP", layout="wide", page_icon=":chart_with_upwards_trend:")
image_path = os.path.join(os.path.dirname(__file__), "../images/stockuplogo.png")
st.sidebar.image(image_path, use_container_width=True)
st.title('Fundamental Information')
a = '''
<style>
[data-testid="stSidebarContent"]{
        background:#000000;
    }
.stApp {
    background: #000000;
    }
[data-testid="stHeader"]{
    background:#000000;
    }
[data-testid="StyledLinkIconContainer"]{
    text-align: center;
}
[id='fundamental-information']{
    padding:0px;
}
[data-testid="stAppViewBlockContainer"]{
    padding-top:60px;
}
</style>
'''
st.markdown(a, unsafe_allow_html=True)

# Load symbols from CSV
csv = pd.read_csv('Files/symbols.csv')
symbol = csv['Symbol'].tolist()
for i in range(len(symbol)):
    symbol[i] = symbol[i] + ".NS"

ticker = st.selectbox(
    'Enter or Choose NSE listed Stock Symbol',
    symbol, placeholder='Enter or Choose a NSE listed Stock Symbol', index=None, label_visibility='hidden'
)

if ticker is None:
    st.warning("Please select a stock to continue.")
else:
    stock = yf.Ticker(ticker)
    try:
        info = stock.info
    except requests.exceptions.ConnectionError as ce:
        st.error(f"Error retrieving stock information: {ce}\nPlease check your network connection or try again later.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.stop()
    
    st.subheader(info.get('longName', 'No Name Available'))
    st.write(f"****Sector****: {info.get('sector', 'Not available')}")
    st.write(f"****Industry****: {info.get('industry', 'Not available')}")
    st.write(f"****Phone****: {info.get('phone', 'Not available')}")
    
    address1 = info.get('address1')
    city = info.get('city')
    zip_code = info.get('zip')
    country = info.get('country')
    if address1 and city and zip_code and country:
        address = f"{address1}, {city}, {zip_code}, {country}"
        st.write(f"****Address****: {address}")
    else:
        st.write("****Address****: Not available")
    
    st.write(f"****Website****: {info.get('website', 'Not available')}")
    with st.expander('See detailed business summary'):
        st.write(info.get('longBusinessSummary', 'Not available'))

    # Date inputs for historical prices
    min_value = dt.datetime.today() - dt.timedelta(10 * 365)
    max_value = dt.datetime.today()
    start_input = st.date_input(
        'Enter starting date',
        value=dt.datetime.today() - dt.timedelta(90),
        min_value=min_value, max_value=max_value,
        help='Enter the starting date from which you have to look the price'
    )
    end_input = st.date_input(
        'Enter last date',
        value=dt.datetime.today(),
        min_value=min_value, max_value=max_value,
        help='Enter the last date till which you have to look the price'
    )

    # Download historical price data
    hist_price = yf.download(ticker, start_input, end_input)
    hist_price = hist_price.reset_index()
    # Flatten any multi-index columns (if any)
    hist_price.columns = [col if not isinstance(col, tuple) else col[0] for col in hist_price.columns]
    # Ensure Date is in datetime format
    hist_price['Date'] = pd.to_datetime(hist_price['Date'])

    @st.cache_data
    def convert_data(df):
        return df.to_csv(index=False).encode('utf-8')
    historical_csv = convert_data(hist_price)
    st.download_button(
        label="Download historical data as CSV",
        data=historical_csv,
        file_name='historical_df.csv',
        mime='text/csv',
    )

    # Chart selection: Candlestick or Line Chart
    chart = st.radio(
        "Choose Style",
        ('Candlestick', 'Line Chart')
    )
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
    if chart == 'Line Chart':
        fig = px.line(
            hist_price, 
            x="Date", 
            y="Close", 
            title="Stock Prices of " + ticker,
            render_mode='svg'
        )
        fig.update_layout(title_x=0.5, height=600, template='plotly_white')
        fig.update_yaxes(tickprefix='₹')
        st.plotly_chart(fig, use_container_width=True)
    if chart == 'Candlestick':
        fig = go.Figure(data=[go.Candlestick(
            x=hist_price['Date'],
            open=hist_price['Open'],
            high=hist_price['High'],
            low=hist_price['Low'],
            close=hist_price['Close']
        )])
        fig.update_layout(title={'text': 'Stock Prices of ' + ticker, 'x': 0.5, 'xanchor': 'center'},
                          height=600, template='gridon')
        fig.update_yaxes(tickprefix='₹')
        st.plotly_chart(fig, use_container_width=True)

    # Updated financial tabs using new yfinance variables
    # Here, we omit the Annual Income Statement tab for brevity.
    tab1, tab3, tab4, tab5 = st.tabs([
        "Quarterly Income Statement", 
        "Balance Sheet", 
        "Cash Flow", 
        "Splits & Dividends"
    ])

    # Helper function for formatting numbers with commas
    def format_numbers(val):
        if pd.isnull(val):
            return ""
        try:
            return f"{float(val):,.0f}"
        except Exception:
            return val

    # Tab 1: Quarterly Income Statement (new variable)
    with tab1:
        st.subheader('Quarterly Income Statement')
        st.write("This displays selected key items from the quarterly income statement.")
        try:
            quarterly_results = stock.quarterly_income_stmt
            if quarterly_results.empty:
                st.write("No Quarterly Income Statement data available.")
            else:
                # Define allowed rows (order can be adjusted as needed; here, we force "Net Income" first)
                allowed_rows = [
                    "Net Income",
                    "EBITDA",
                    "Net Interest Income",
                    "Total Expenses",
                    "Basic Average Shares",
                    "Basic EPS",
                    "Pretax Income",
                    "Operating Income",
                    "Operating Expense",
                    "Gross Profit",
                    "Cost Of Revenue",
                    "Total Revenue",
                    "Operating Revenue"
                ]
                # Filter to only keep rows in allowed_rows
                quarterly_results = quarterly_results.loc[quarterly_results.index.intersection(allowed_rows)]
                # Reorder the DataFrame so that allowed_rows order is preserved
                quarterly_results = quarterly_results.reindex([r for r in allowed_rows if r in quarterly_results.index])
                # If columns are a DatetimeIndex, convert them to strings for display
                if isinstance(quarterly_results.columns, pd.DatetimeIndex):
                    quarterly_results.columns = [c.strftime('%Y-%m-%d') for c in quarterly_results.columns]
                # Drop rows that are completely NaN
                quarterly_results.dropna(axis=0, how='all', inplace=True)
                # Format the DataFrame cell-by-cell
                formatted_df = quarterly_results.applymap(format_numbers)
                st.dataframe(formatted_df.style.highlight_max(axis=1, color='green'), width=1000)
        except Exception as e:
            st.write("Error retrieving Quarterly Income Statement:", e)

    # Tab 3: Balance Sheet (using new variable balance_sheet)
    with tab3:
        st.subheader('Balance Sheet')
        st.write("This displays key balance sheet metrics.")
        try:
            balance = stock.balance_sheet
            if balance.empty:
                st.write("Balance Sheet data is not available.")
            else:
                # Flatten multi-index columns if they exist
                if isinstance(balance.columns, pd.MultiIndex):
                    balance.columns = [col[0] for col in balance.columns]
                # If columns are a DatetimeIndex, convert them to strings for display
                if isinstance(balance.columns, pd.DatetimeIndex):
                    balance.columns = [c.strftime('%Y-%m-%d') for c in balance.columns]
                balance.dropna(axis=0, how='all', inplace=True)
                formatted_balance = balance.applymap(format_numbers)
                st.dataframe(formatted_balance.style.highlight_max(axis=1, color='green'), width=1000)
        except Exception as e:
            st.write("Error retrieving Balance Sheet:", e)

    # Tab 4: Cash Flow (using new variable cashflow)
    with tab4:
        st.subheader('Cash Flow')
        st.write("This displays key cash flow metrics.")
        try:
            cf = stock.cashflow
            if cf.empty:
                st.write("Cash Flow data is not available.")
            else:
                if isinstance(cf.columns, pd.DatetimeIndex):
                    cf.columns = [c.strftime('%Y-%m-%d') for c in cf.columns]
                cf.dropna(axis=0, how='all', inplace=True)
                formatted_cf = cf.applymap(format_numbers)
                st.dataframe(formatted_cf.style.highlight_max(axis=1, color='green'), width=1000)
        except Exception as e:
            st.write("Error retrieving Cash Flow:", e)

    # Tab 5: Splits & Dividends (using new variable actions)
    with tab5:
        st.subheader('Splits & Dividends')
        st.write("This displays both splits and dividend history.")
        try:
            actions = stock.actions  # New variable containing splits and dividends
            if actions.empty:
                st.write("Splits & Dividends data is not available.")
            else:
                actions = actions.reset_index()
                # Rename columns for clarity; expected order: Date, Dividends, Splits
                # (Depending on the ticker, one of these might be NaN)
                actions.columns = ['Date', 'Dividends', 'Splits']
                actions['Date'] = pd.to_datetime(actions['Date']).dt.strftime('%Y-%m-%d')
                st.dataframe(actions, width=1000)
        except Exception as e:
            st.write("Error retrieving Splits & Dividends:", e)
