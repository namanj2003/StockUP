import streamlit as st
import os

st.set_page_config(
    page_title="StockUP",
    page_icon="📈",
    layout="wide", 
    initial_sidebar_state="collapsed"
)
image_path = os.path.join(os.path.dirname(__file__), "./images/stockuplogo.png")
st.sidebar.image(image_path, use_container_width=True)

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

base_dir = os.path.dirname(__file__)  
image1_path = os.path.join(base_dir, "./images/2.png")
image2_path = os.path.join(base_dir, "./images/FINBERTTT.png")
st.image(image1_path, use_container_width=True)
st.image(image2_path, use_container_width=True)