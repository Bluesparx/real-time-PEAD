# dashboard.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pymongo import MongoClient
from datetime import datetime, timedelta
import requests
from typing import Optional

# Page config
st.set_page_config(
    page_title="BSE AI Analysis Dashboard",
    page_icon="📈",
    layout="wide"
)

# --- Configuration ---
API_URL = "http://localhost:8000"

# Title
st.title("📈 BSE Real-Time AI Analysis Dashboard")
st.markdown("---")

# --- Helper Functions ---

# Function to fetch data from the FastAPI backend
def fetch_api_data(endpoint: str, company: Optional[str] = None):
    try:
        url = f"{API_URL}{endpoint}"
        if company:
            url += f"/{company}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to API ({endpoint}): {e}")
        return None
    
# --- Sidebar ---
st.sidebar.header("Filters")

# Get companies
companies_data = fetch_api_data("/companies")
companies = companies_data.get("companies", []) if companies_data else []

selected_company = st.sidebar.selectbox("Select Company", ["All"] + companies)

date_range = st.sidebar.date_input(
    "Analysis Window",
    value=(datetime.now() - timedelta(days=90), datetime.now())
)

# Refresh data
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.header("Agent Status")

# --- Main Dashboard ---

# 1. KPI Cards (Stats fetched from /dashboard/summary)
summary_data = fetch_api_data("/dashboard/summary")

if summary_data:
    stats = summary_data.get("stats", {})
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Announcements", f"{stats.get('total_announcements', 0):,}")
    col2.metric("PDFs Processed", f"{stats.get('pdfs_processed', 0):,}")
    col3.metric("Insights Generated", f"{stats.get('insights_generated', 0):,}")
    col4.metric("Predictions Made", f"{stats.get('predictions_made', 0):,}")
    
st.markdown("---")

# 2. Company-Specific Analysis (Prediction + Insights)
if selected_company != "All":
    st.header(f"Company Profile: {selected_company}")

    # --- Prediction ---
    prediction_data = fetch_api_data("/predictions", company=selected_company)
    
    if prediction_data:
        pred_col1, pred_col2, pred_col3 = st.columns([1, 1, 2])
        
        pred_col1.subheader("Prediction")
        pred_col1.metric(
            "7-Day Change", 
            f"{prediction_data.get('predicted_change_pct', 0):.2f}%",
            delta=prediction_data.get('predicted_direction', '')
        )
        pred_col2.metric("Confidence Score", f"{prediction_data.get('confidence', 0):.2f}")
        pred_col3.info(
            f"**Analysis Date:** {prediction_data.get('prediction_date').split('T')[0]}"
        )
    else:
        st.warning(f"No predictions available for {selected_company}.")
    
    st.markdown("---")

    # --- Insights History ---
    st.subheader("Historical AI Insights")
    insights_list = fetch_api_data("/insights", company=selected_company)
    
    if insights_list:
        # Convert to DataFrame for display and plotting
        insights_df = pd.DataFrame(insights_list)
        
        # 2a. Sentiment Over Time (Using Plotly)
        sentiment_scores = insights_df['metrics'].apply(
            lambda x: 1 if x.get('sentiment') == 'positive' else (-1 if x.get('sentiment') == 'negative' else 0)
        )
        
        fig = px.bar(
            x=insights_df['date'], 
            y=sentiment_scores, 
            title="Sentiment Trend of Announcements",
            labels={'x': 'Date', 'y': 'Sentiment Score (-1 to 1)'}
        )
        st.plotly_chart(fig, use_container_width=True)

        # 2b. Detailed Metrics Table
        st.markdown("#### Key Financial Metrics Extracted by LLM")
        
        # Flatten metrics for table view
        metrics_flat = insights_df.apply(
            lambda row: {
                'Date': row['date'],
                'Revenue (M)': row['metrics'].get('revenue'),
                'Profit (M)': row['metrics'].get('profit'),
                'Sentiment': row['metrics'].get('sentiment'),
                'Key Highlights': ', '.join(row['metrics'].get('key_highlights', []))[:80] + '...'
            }, axis=1
        )
        st.dataframe(metrics_flat, use_container_width=True)
        
else:
    # 3. Global Summary (If 'All' is selected)
    st.header("Global Market Overview")
    
    if summary_data:
        # Top Predictions Table
        st.subheader("Top Predicted Stock Moves")
        top_predictions_df = pd.DataFrame(summary_data.get("top_predictions", []))
        if not top_predictions_df.empty:
            st.dataframe(
                top_predictions_df[['company', 'predicted_change_pct', 'predicted_direction', 'confidence']],
                use_container_width=True
            )
        else:
            st.info("No sufficient data to generate top predictions yet.")
        
        # Positive News Feed
        st.subheader("Latest Positive News (LLM Classified)")
        positive_df = pd.DataFrame(summary_data.get("positive_news", []))
        if not positive_df.empty:
            positive_df['title'] = positive_df['metrics'].apply(
                lambda x: x.get('key_highlights', [''])[0]
            )
            st.dataframe(positive_df[['date', 'company', 'category', 'title']], use_container_width=True)
        else:
            st.info("No recent positive news classified by the LLM.")