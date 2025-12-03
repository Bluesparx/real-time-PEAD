import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
from typing import Optional, Dict, List, Any

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Qualitative Event Ranking Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Replace with your actual API URL
API_URL = "http://localhost:8000"

# --- ENHANCED CSS ---
st.markdown("""
    <style>
    .stMetric {
        background-color: #e0e0e0 ;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Qualitative Event Ranking Dashboard")
st.caption("Qualitative Analysis and Ranking for Earnings Events (Fundamentals + Sentiment + Price Reaction)")

# --- API HELPER ---
def fetch_api_data(endpoint: str, params: Optional[Dict] = None):
    try:
        url = f"{API_URL}{endpoint}"
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"API Error at {endpoint}: {e}. Ensure the FastAPI server is running at {API_URL}.")
        return None

# --- SIDEBAR FILTERS (Simplified) ---
with st.sidebar:
    st.header("Filters")
    
    today = datetime.now().date()
    
    # 1. Time Period
    st.subheader("Time Period")
    time_period = st.selectbox(
        "Date Range",
        ["Last 3 Days", "Last 7 Days", "Last 30 Days"],
        index=1
    )
    
    if time_period == "Last 3 Days":
        start_date = today - timedelta(days=3)
    elif time_period == "Last 30 Days":
        start_date = today - timedelta(days=30)
    else: # Default: Last 7 Days
        start_date = today - timedelta(days=7)
    end_date = today
    
    st.divider()
    
    # 2. Trading Signals (Simplified multiselect)
    st.subheader("Trading Signal")
    signal_filter = st.multiselect(
        "Include Signals",
        ["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"],
        default=["Strong Buy", "Buy"] # Focus on opportunities
    )
    
    st.divider()
    
    # Refresh button
    if st.button("Refresh Data", width='stretch'):
        st.cache_data.clear()
        st.rerun()
    
    st.caption("Data is currently filtered from {} to {}.".format(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")))

# --- HELPER FUNCTIONS ---
def get_trading_signal(row):
    """Trading signal logic based ONLY on the single ranking_score."""
    ranking_score = row.get('ranking_score', 0)
    
    if ranking_score > 0.4:
        # High ranking score indicates strong qualitative assessment and positive price reaction
        return "Strong Buy", "Strong Buy", "#00c853"
    elif ranking_score > 0.1:
        return "Buy", "Buy", "#ffd600"
    elif ranking_score < -0.1:
        return "Sell", "Sell", "#ff5252"
    elif ranking_score < -0.4:
        return "Strong Sell", "Strong Sell", "#424242"
    else:
        return "Hold", "Hold", "#9e9e9e"

def get_sentiment_label(score):
    if score is None:
        return "Neutral"
    if score > 0.3:
        return "Positive"
    elif score < -0.3:
        return "Negative"
    return "Neutral"

def format_currency(value):
    if value is None or pd.isna(value):
        return "N/A"
    if value >= 10000000:  # Crores
        return f"₹{value/10000000:.2f}Cr"
    elif value >= 100000:  # Lakhs
        return f"₹{value/100000:.2f}L"
    return f"₹{value:.2f}"

def format_percentage(value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.2f}%"

# --- DATA FETCHERS ---
@st.cache_data(ttl=300)
def get_rankings_data(start_date, end_date):
    params = {
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d")
    }
    return fetch_api_data("/rankings", params=params)

@st.cache_data(ttl=3600)
def get_calendar_data():
    """Fetches the forthcoming results calendar data."""
    calendar_response = fetch_api_data("/calendar") 
    return calendar_response.get('calendar', []) if calendar_response else []


# --- MAIN DATA LOADING ---
with st.spinner("Loading rankings data..."):
    rankings_data = get_rankings_data(start_date, end_date)

# --- RANKINGS PROCESSING AND FILTERING ---
if rankings_data and rankings_data.get('rankings'):
    df = pd.DataFrame(rankings_data['rankings'])
    
    # --- DATA PROCESSING ---
    df['signal'], df['signal_label'], df['signal_color'] = zip(*df.apply(get_trading_signal, axis=1))
    df['sentiment_label'] = df['sentiment'].apply(get_sentiment_label)
    
    # Robust date parsing
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Handle NaN values for columns used in Plotly 'size' argument
    if 'revenue_current' in df.columns:
        median_rev = df['revenue_current'].median() if df['revenue_current'].notna().any() else 1
        df['revenue_current_plot'] = df['revenue_current'].fillna(median_rev)
    else:
        df['revenue_current_plot'] = 1 
    
    # Apply filters 
    filtered_df = df[
        (df['signal'].isin(signal_filter)) 
    ].copy()
    
    # Apply implicit revenue growth filter (can be adjusted/removed)
    if 'revenue_yoy_change' in filtered_df.columns:
        min_revenue_growth = -20.0 
        filtered_df = filtered_df[
            (filtered_df['revenue_yoy_change'].isna()) | 
            (filtered_df['revenue_yoy_change'] >= min_revenue_growth)
        ]
    
    # Final sorting based on the ranking score
    filtered_df = filtered_df.sort_values('ranking_score', ascending=False)
    
    # ADDED: Calculate rank on the frontend after final sorting
    filtered_df['rank'] = range(1, len(filtered_df) + 1) 
    
    # --- KEY METRICS ---
    st.divider()
    
    # --- MAIN CONTENT TABS ---
    tab1, tab2, tab3 = st.tabs(["Top Rankings", "Deep Dive", "Results Calendar"])
    
    with tab1:
        st.subheader(f"Top {len(filtered_df)} Ranked Events")
        
        # Prepare clean table
        display_df = filtered_df.head(20).copy()
        
        table_data = []
        for _, row in display_df.iterrows():
            table_data.append({
                'Rank': row['rank'], # Now safely accessible
                'Signal': row['signal'],
                'Company': row['company'],
                'Ticker': row.get('ticker', 'N/A'),
                'Date': row['date'].strftime('%d-%b'),
                'Ranking Score': f"{row['ranking_score']:.4f}",
                'Sentiment': row['sentiment_label'],
                'Revenue': format_currency(row.get('revenue_current')),
                'Rev YoY': format_percentage(row.get('revenue_yoy_change')),
            })
        
        st.dataframe(
            pd.DataFrame(table_data),
            width='stretch',
            hide_index=True,
            height=500
        )
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download Full Data (CSV)",
            data=csv,
            file_name=f"event_rankings_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with tab2:
        st.subheader("Detailed Company Analysis")
        
        # Company selector
        selected_company = st.selectbox(
            "Select a company for detailed analysis",
            filtered_df['company'].unique()
        )
        
        if selected_company:
            company_data = filtered_df[filtered_df['company'] == selected_company].iloc[0]
            
            # Overview
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Signal", company_data['signal'])
            with col2:
                st.metric("Ranking Score", f"{company_data['ranking_score']:.4f}")
            with col3:
                st.metric("Sentiment Score", f"{company_data.get('sentiment', 0):.3f}")
            
            st.divider()
            
            # Financials
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Financial Metrics")
                st.write(f"**Quarter:** {company_data.get('quarter', 'N/A')}")
                st.write(f"**Date:** {company_data['date'].strftime('%Y-%m-%d')}")
                st.write(f"**Revenue:** {format_currency(company_data.get('revenue_current'))}")
                st.write(f"**PAT:** {format_currency(company_data.get('pat_current'))}")
                st.write(f"**EBITDA:** {format_currency(company_data.get('ebitda_current'))}")
            
            with col2:
                st.markdown("### Growth & Ranking")
                st.write(f"**Rank:** {company_data['rank']}") # Now safely accessible
                st.write(f"**Rev YoY Change:** {format_percentage(company_data.get('revenue_yoy_change'))}")
                st.write(f"**Sentiment:** {company_data['sentiment_label']}")
            
            # Key highlights
            if company_data.get('key_highlights'):
                st.markdown("### Key Highlights")
                for highlight in company_data['key_highlights']:
                    st.markdown(f"* {highlight}")
        else:
            st.info("No company selected for detailed analysis.")
            
    with tab3:
        st.subheader("Forthcoming Results Calendar (Next 30 Days)")
        
        with st.spinner("Fetching calendar data..."):
            calendar_records = get_calendar_data()
        
        if calendar_records:
            cal_df = pd.DataFrame(calendar_records)
            
            # Select and rename columns for display
            display_cal_df = cal_df.rename(columns={
                'company_name': 'Company Name',
                'TICKER': 'Ticker',
                'meeting_date_standard': 'Meeting Date',
                'meeting_date_raw': 'Raw Date',
                'scrip_code': 'Scrip Code',
            })[['Company Name', 'Ticker', 'Meeting Date', 'Raw Date', 'Scrip Code']]
            
            # Format date for better sorting/display
            display_cal_df['Meeting Date'] = pd.to_datetime(display_cal_df['Meeting Date'], errors='coerce').dt.strftime('%Y-%m-%d')
            display_cal_df = display_cal_df.sort_values(by='Meeting Date', ascending=True)

            st.dataframe(
                display_cal_df,
                width='stretch',
                hide_index=True,
                height=600
            )
            
            st.caption("Data scraped from BSE India, showing forthcoming results scheduled in the next 30 days.")
        else:
            st.warning("Could not retrieve forthcoming results calendar. Ensure your backend API is running and exposes the `/calendar` endpoint.")


else:
    # If rankings_data is None (API failure) or rankings_data['rankings'] is empty (DB empty/query failed)
    st.warning(f"No ranking data available for the selected period from the API ({API_URL}/rankings).")

# Footer
st.divider()
st.caption("This dashboard is for informational purposes only. Not financial advice.")