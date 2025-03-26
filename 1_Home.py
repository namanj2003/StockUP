import streamlit as st

st.set_page_config(
    page_title="StockUP",
    page_icon="📈",
    layout="wide", 
    initial_sidebar_state="collapsed"
)
st.sidebar.image("D:\Clg coding\My Projects\StockUP Project\images\stockuplogo.png", use_container_width=True)

css = """
<style>
    [data-testid="stHeader"]{
        background:#000000;
    }
    .stApp {
        background: rgb(7,200,175);
        background: linear-gradient(153deg, rgba(7,200,175,1) 16%, rgba(35,11,211,1) 56%, rgba(17,11,85,1) 71%, rgba(0,0,0,1) 100%);
    }
    [data-testid="stSidebarContent"]{
        background:#000000;
    }
</style>
"""

st.markdown(css, unsafe_allow_html=True)

st.image(
    "D:\Clg coding\My Projects\StockUP Project\images\\2.png",
    use_container_width=True
)
st.image(
    "D:\Clg coding\My Projects\StockUP Project\images\FINBERTTT.png",
    use_container_width=True
)
