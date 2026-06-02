import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Big Data Weather Analysis Dashboard", layout="wide")

st.title("🌧️ Big Data Weather Analysis & Predictive Dashboard")
st.markdown("Created by Oliia | Phase 4 Deployment")

try:
    loc_df = pd.read_csv('data/dashboard_location_summary.csv')
    ml_df = pd.read_csv('data/dashboard_ml_predictions.csv')
    
    # Setup interactive filter
    st.sidebar.header("Dashboard Controls")
    selected_locations = st.sidebar.multiselect(
        "Filter Locations:", 
        options=list(loc_df['Location'].unique()), 
        default=list(loc_df['Location'].unique())[:5]
    )
    
    filtered_loc_df = loc_df[loc_df['Location'].isin(selected_locations)]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📍 Regional Weather Metrics Aggregation")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(data=filtered_loc_df, x='Location', y='Avg_Rainfall', ax=ax, palette='Blues_r')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
    with col2:
        st.subheader("🔮 Machine Learning Model Evaluation")
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        sns.scatterplot(data=ml_df, x='label', y='prediction', alpha=0.4, color='teal', ax=ax2)
        # Perfect prediction line
        min_val = min(ml_df['label'].min(), ml_df['prediction'].min())
        max_val = max(ml_df['label'].max(), ml_df['prediction'].max())
        ax2.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
        ax2.set_xlabel("Actual Temperature Range")
        ax2.set_ylabel("Predicted Temperature Range")
        st.pyplot(fig2)
        
    st.success("Dashboard components successfully compiled from pipeline data!")
    
except Exception as e:
    st.error(f"Waiting for pipeline files... Error: {e}")