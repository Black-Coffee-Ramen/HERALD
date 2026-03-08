import streamlit as st
import requests
import os
import pandas as pd

st.set_page_config(page_title="Phishing Detection Dashboard", layout="wide")

API_URL = os.getenv("API_URL", "http://localhost:8000")
API_TOKEN = os.getenv("API_TOKEN", "default_secret")

def check_auth():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.markdown("## 🔒 Authentication Required")
        token_input = st.text_input("Enter API Token:", type="password")
        if st.button("Login"):
            if token_input == API_TOKEN:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Invalid Token")
        st.stop()
    else:
        if st.sidebar.button("Logout"):
            st.session_state.authenticated = False
            st.rerun()

check_auth()

st.title("🛡️ Phishing Detection Dashboard")

def get_label_color(label):
    if label == "Phishing":
        return "🔴 Phishing"
    elif label == "Suspected":
        return "🟠 Suspected"
    elif label == "Clean":
        return "🟢 Clean"
    else:
        return "⚪ Unknown"

col1, col2 = st.columns([0.8, 0.2])
with col1:
    st.markdown("### Recent Detections")
with col2:
    if st.button("🔄 Refresh Data"):
        st.rerun()

headers = {"X-API-Token": API_TOKEN}

try:
    res_det = requests.get(f"{API_URL}/api/detections", headers=headers)
    if res_det.status_code == 200:
        det_data = res_det.json()
        if det_data:
            st.markdown(f"**{len(det_data)} domains detected recently**")
            df_det = pd.DataFrame(det_data)
            df_det['Label'] = df_det['label'].apply(get_label_color)
            df_det = df_det.rename(columns={
                'domain': 'Domain',
                'confidence': 'Confidence',
                'target_cse': 'Target CSE',
                'scan_date': 'Detected At'
            })
            st.dataframe(df_det[['Domain', 'Label', 'Confidence', 'Target CSE', 'Detected At']], use_container_width=True)
        else:
            st.info("No recent detections found.")
    else:
        st.error(f"Failed to fetch detections. API Status: {res_det.status_code}")
except Exception as e:
    st.error(f"Error connecting to API: {e}")

st.markdown("---")

st.markdown("### Suspected Domains")


try:
    response = requests.get(f"{API_URL}/api/suspected", headers=headers)
    if response.status_code == 200:
        data = response.json()
        if data:
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No suspected domains found in the database.")
    else:
        st.error(f"Failed to fetch data. API Status: {response.status_code}")
except Exception as e:
    st.error(f"Error connecting to API: {e}")

st.markdown("---")
st.markdown("### Manual Domain Scan")
with st.form("manual_scan"):
    domain_to_scan = st.text_input("Domain to Scan:")
    target_cse = st.text_input("Target CSE (Optional):", value="Unknown")
    submitted = st.form_submit_button("Queue Scan")
    if submitted and domain_to_scan:
        try:
            res = requests.post(
                f"{API_URL}/api/scan",
                json={"domain": domain_to_scan, "target_cse": target_cse},
                headers=headers
            )
            if res.status_code == 200:
                st.success(f"Domain '{domain_to_scan}' queued successfully.")
            else:
                st.error(f"Failed to queue domain. API Error: {res.text}")
        except Exception as e:
            st.error(f"Error contacting API: {e}")
