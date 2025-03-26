import streamlit as st
import requests
from bs4 import BeautifulSoup
from dateutil import parser

st.set_page_config(page_title="StockUP", layout="wide", page_icon=":chart_with_upwards_trend:")
st.sidebar.image("D:\Clg coding\My Projects\StockUP Project\images\stockuplogo.png", use_container_width=True)
# Simplified CSS
css = """
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
[data-testid="stAppViewContainer"] {
    padding: 0 !important;
}

/* Add top padding to the page */
[data-testid="stAppViewBlockContainer"] {
    padding-top: 60px !important;
}

/* Center any link icons */
[data-testid="StyledLinkIconContainer"] {
    text-align: center;
}

/* Hide horizontal overflow if needed */
body, .main {
    overflow-x: hidden;
}

/* Make images responsive and match the height of text container */
.article-block {
    display: flex;
    align-items: center;
    margin-bottom: 2rem;
}

.article-block img {
    width: 100%;
    height: auto;
    max-height: 250px; /* Adjust max height to match the text block */
    object-fit: cover;
    border-radius: 10px;
}

/* Flexbox to align text and image */
.article-content {
    display: flex;
    align-items: center;
    justify-content: center;
}

</style>
"""

st.markdown(css, unsafe_allow_html=True)

st.title("News & Announcements")
st.header(":zap: Buzzing Stocks News")
st.divider()

# Fetch RSS feed from Economic Times
news = requests.get('https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms')
soup = BeautifulSoup(news.text, "lxml-xml")

titles = soup.find_all("title")
descriptions = soup.find_all("description")
links = soup.find_all("link")
image_urls = [enclosure.get('url') for enclosure in soup.find_all("enclosure")]
dates = soup.find_all("pubDate")

for title, description, link, image_url, date_str in zip(titles[2:], descriptions[1:], links[3:], image_urls, dates):
    date = parser.parse(date_str.text)
    
    # Start a container for each article block
    with st.container():
        st.markdown('<div class="article-block">', unsafe_allow_html=True)
        
        # Use columns with a small ratio so the image stays smaller
        col1, col2 = st.columns([1, 3])
        
        with col1:
            if image_url:
                # Set a fixed width to prevent overly large images
                st.image(image_url, width=300)
            else:
                st.write(f"Date: {date_str.text}")
                st.divider()
        
        with col2:
            st.subheader(title.text)
            st.write(description.text)
            st.link_button("Read More", link.text)  # Custom link button method
            st.write(date.strftime('%Y-%m-%d %H:%M:%S'))
            st.divider()
        
        st.markdown('</div>', unsafe_allow_html=True)
