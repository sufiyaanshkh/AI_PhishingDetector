# 🔍 AI Phishing Detector: URL Safety Checker

## Overview
The **AI Phishing Detector** is a web-based application built with **Streamlit** that allows users to check whether a URL is phishing (malicious) or legitimate (safe). Using a **Random Forest Classifier**, the application analyzes URLs based on multiple features like domain structure, suspicious words, special characters, and more. The model is trained on verified phishing URLs and top legitimate websites to provide accurate detection in real-time.

---

## Features
- **Real-Time URL Analysis:** Enter any website URL and instantly check its safety.
- **High Accuracy:** Uses a trained Random Forest model for phishing detection.
- **URL Feature Extraction:**
  - URL length
  - Number of dots (`.`) and slashes (`/`)
  - Presence of IP address in the domain
  - HTTPS usage
  - Number of subdomains
  - Special characters (`@`, `-`, `_`)
  - Suspicious keywords (`login`, `secure`, `update`, `verify`, `account`, `bank`)
- **User-Friendly Interface:** Built with Streamlit for easy interaction.

---

## Datasets
1. **Phishing Sites (`phishing_sites.csv`)** – Verified phishing URLs.
2. **Top 1 Million Sites (`top-1m.csv`)** – Legitimate high-ranking websites.

---

## Tech Stack
- Python
- Streamlit
- Scikit-learn (Random Forest Classifier)
- Pandas, NumPy
- URL Parsing and feature extraction using `urllib` and `regex`

---

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/sufiyaanshkh/AI_PhishingDetector.git
cd AI_PhishingDetector

## PROJECT STRUCTURE 
AI_PhishingDetector/
│
├── app.py                 # Streamlit app
├── train_url_model.py     # Script to train phishing URL model
├── url_model.pkl          # Saved Random Forest model
├── datasets/
│   ├── phishing_sites.csv
│   └── top-1m.csv
├── README.md              # Project documentation
└── requirements.txt       # Python dependencie