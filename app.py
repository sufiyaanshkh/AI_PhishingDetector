import streamlit as st
import pandas as pd
import joblib
from urllib.parse import urlparse
import re

# ---------------------------
# LOAD MODEL
# ---------------------------
url_model = joblib.load("url_model.pkl")

# ---------------------------
# URL FEATURE EXTRACTION
# ---------------------------
def extract_url_features(url):
    if not url:
        url = ''
    parsed_url = urlparse(url)
    features = {
        'length': len(url),
        'num_dots': url.count('.'),
        'num_slashes': url.count('/'),
        'has_ip': bool(re.match(r'\d+\.\d+\.\d+\.\d+', parsed_url.netloc)),
        'has_https': url.startswith('https'),
        'num_subdomains': len(parsed_url.netloc.split('.')) - 1 if parsed_url.netloc else 0,
        'has_at': '@' in url,
        'has_dash': '-' in parsed_url.netloc,
        'has_underscore': '_' in parsed_url.netloc,
        'suspicious_word': int(any(word in parsed_url.netloc.lower() for word in ['login','secure','update','verify','account','bank']))
    }
    return pd.Series(features)

# ---------------------------
# PREDICTION FUNCTION
# ---------------------------
def predict_url(model, url):
    features = extract_url_features(url)
    return model.predict([list(features)])[0]

# ---------------------------
# STREAMLIT APP
# ---------------------------
st.title("🔍 Phishing Detector: Check Website Safety")

user_input = st.text_input("Enter the URL to check:")

if st.button("Check Safety"):
    if user_input.strip():
        prediction = predict_url(url_model, user_input.strip())
        if prediction == 1:
            st.error("⚠️ This URL appears to be PHISHING (Unsafe)!")
        else:
            st.success("✅ This URL appears to be SAFE.")
    else:
        st.warning("Please enter a URL.")
