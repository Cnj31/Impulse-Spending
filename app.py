
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")
st.title("📊 Impulse Spending Insights & Prediction Dashboard")

@st.cache_resource
def load_model():
    return joblib.load("final_catboost_model.pkl")

model = load_model()

uploaded_file = st.file_uploader("Upload transaction dataset (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    st.subheader("🔍 Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("🤖 Predict Impulse Purchases")
    features = ['Time_Since_Last_Transaction','Is_First_Transaction','Avg_Session_Duration_User',
                'Days_Until_Salary','Purchase_After_Social_Min','Ad_Same_Hour']

    X = df[features].fillna(0)
    y_probs = model.predict_proba(X)[:, 1]
    df['Impulse_Probability'] = y_probs
    df['Predicted_Impulse'] = (y_probs >= 0.60).astype(int)

    st.success("✅ Predictions complete.")
    st.dataframe(df[['User_ID', 'Impulse_Probability', 'Predicted_Impulse']].head())

    st.subheader("📈 Key Performance Indicators")
    col1, col2, col3 = st.columns(3)

    total_txns = len(df)
    impulse_txns = df['Predicted_Impulse'].sum()
    impulse_rate = round((impulse_txns / total_txns) * 100, 2)
    avg_session = round(df['Avg_Session_Duration_User'].mean(), 2)
    avg_delay = round(df['Purchase_After_Social_Min'].mean(), 2)

    col1.metric("Total Transactions", f"{total_txns:,}")
    col2.metric("Impulse Rate (%)", f"{impulse_rate}%")
    col3.metric("Avg. Session Duration", f"{avg_session} min")

    st.subheader("📊 Behavioral Visualizations")
    tab1, tab2, tab3 = st.tabs(["📅 Impulse by Day", "🎯 Probability Curve", "⏱️ Session Duration vs Impulse"])

    with tab1:
        if 'Transaction_DateTime' in df.columns:
            df['Transaction_DateTime'] = pd.to_datetime(df['Transaction_DateTime'], errors='coerce')
            df['DayOfWeek'] = df['Transaction_DateTime'].dt.day_name()
            chart_data = df.groupby('DayOfWeek')['Predicted_Impulse'].mean().sort_values()
            st.bar_chart(chart_data)
        else:
            st.warning("No 'Transaction_DateTime' column found.")

    with tab2:
        fig, ax = plt.subplots()
        sns.histplot(df['Impulse_Probability'], bins=30, kde=True, ax=ax)
        ax.set_title("Impulse Probability Distribution")
        st.pyplot(fig)

    with tab3:
        fig, ax = plt.subplots()
        sns.boxplot(x='Predicted_Impulse', y='Avg_Session_Duration_User', data=df, ax=ax)
        ax.set_xticklabels(['Not Impulse', 'Impulse'])
        ax.set_title("Session Duration vs Impulse")
        st.pyplot(fig)

    st.subheader("📌 Strategic Business Recommendations")
    st.markdown("""
    - **Targeted Ads:** Deliver ads within 1 hour of high-activity periods.
    - **Salary Cycles:** Promote offers post-salary (day 25–30 of cycle).
    - **High-Risk Categories:** Focus on users with longer session times and social ad overlap.
    - **Personalization:** Use behavioral indicators to customize promotions.
    """)

    st.subheader("📥 Download Predictions")
    st.download_button("Download Results as CSV", df.to_csv(index=False), file_name="impulse_predictions.csv")
else:
    st.info("👆 Upload a CSV file to begin analysis.")
