import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from urllib.parse import urlparse
import re
import joblib

# ---------------------------
# LOAD DATASETS
# ---------------------------
phishing_sites_csv = pd.read_csv("datasets/phishing_sites.csv")
top_1m_csv = pd.read_csv("datasets/top-1m.csv")

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
# TRAIN URL MODEL
# ---------------------------
phishing_df = phishing_sites_csv.copy()
phishing_df['label'] = 1  # Phishing

legit_df = top_1m_csv.copy()
legit_df = legit_df.sample(min(1000, len(legit_df)))
legit_df['url'] = 'https://' + legit_df.iloc[:, 1].astype(str)  # assuming 2nd column has URLs
legit_df['label'] = 0  # Safe

url_df = pd.concat([phishing_df[['url','label']], legit_df[['url','label']]], ignore_index=True)
url_df['url'] = url_df['url'].fillna('').astype(str)

features_df = url_df['url'].apply(extract_url_features)
X = features_df
y = url_df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42, n_estimators=100, n_jobs=-1)
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))
print(f"Model Accuracy: {acc:.2f}")

# Save the model
joblib.dump(model, "url_model.pkl")
print("Model saved as url_model.pkl")
