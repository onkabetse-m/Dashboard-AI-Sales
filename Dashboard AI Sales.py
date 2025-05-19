import streamlit as st
import pandas as pd
import plotly.express as px
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# ✅ Load all models
reg_model = joblib.load("demo_regression_model.pkl")
clf_model = joblib.load("interaction_classifier.pkl")  # <--- This was missing
binary_model = joblib.load("binary_interaction_classifier.pkl")



st.set_page_config(page_title="AI Sales Insights Dashboard", layout="wide")
 
# Load and cache data
st.title("📊 AI Sales Insights Dashboard")
def load_data():
    df= pd.read_csv("AI_Sales_Insights_.csv")

try:
    df = pd.read_csv("AI_Sales_Insights_.csv")
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format="%H:%M:%S", errors='coerce')
    st.session_state["df"] = df
except FileNotFoundError:
    st.error(f"🚫 Could not find the file '{"AI_Sales_Insights_.csv"}'. Please make sure it's placed in the app folder.")
    st.stop()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Home",
    "Visitor & Demo Insights",
    "Job & AI Interaction Insights",
    "Sales & Geography",
    "Session & Campaign Insights",
    "AI Predictions",
    "AI Binary Prediction"  
])
 
# Sidebar actor and team filters
actor = st.sidebar.selectbox("View As", ["Executive", "Sales Team", "Marketing"])
df = st.session_state["df"]
 
# Sales Team filter
all_teams = df["Sales Team Name"].dropna().unique()
selected_teams = st.sidebar.multiselect("View by Sales Team", all_teams)
 
# Apply actor-specific filter
# Apply actor-specific filter
if actor == "Sales Team":
    if selected_teams:
        df = df[df["Sales Team Name"].isin(selected_teams)]  # filter selected teams
    # else: show all sales teams (no filter)
elif actor == "Marketing":
    df = df[df["Campaign Source"].isin(["Twitter", "Google Ads"])]
    if selected_teams:
        df = df[df["Sales Team Name"].isin(selected_teams)]
elif selected_teams:
    df = df[df["Sales Team Name"].isin(selected_teams)]
 
# Page Logic
if page == "Home":
    st.subheader(f"Welcome, {actor}!")
    st.write("Use the sidebar to navigate insights tailored to your role.")
    st.dataframe(df.head())
 
elif page == "Visitor & Demo Insights":
    col1, col2 = st.columns(2)
 
    with col1:
        st.subheader("👥 Daily Unique Visitors")
        daily_visitors = df.groupby("Date")["IP Address"].nunique().reset_index()
        fig1 = px.bar(daily_visitors, x="Date", y="IP Address", labels={"IP Address": "Unique Visitors"})
        st.plotly_chart(fig1, use_container_width=True)
 
    with col2:
        st.subheader("🎯 Demo Request Conversion Rate")
        demo_requests = df[df["Resource Accessed"] == "/scheduledemo.php"]
        demo_conversions = demo_requests[demo_requests["Sale Made"] == "Yes"]
        demo_rate = (len(demo_conversions) / len(demo_requests)) * 100 if len(demo_requests) > 0 else 0
        st.metric("Demo Conversion Rate", f"{demo_rate:.2f}%")
 
elif page == "Job & AI Interaction Insights":
    col1, col2 = st.columns(2)
 
    with col1:
        st.subheader("📈 Job Placement Volume")
        job_requests = df[df["Resource Accessed"] == "/jobportal.php"]
        job_volume = job_requests.groupby("Date").size().reset_index(name="Job Requests")
        fig2 = px.bar(job_volume, x="Date", y="Job Requests")
        st.plotly_chart(fig2, use_container_width=True)
 
    with col2:
        st.subheader("🤖 AI Assistant Interactions")
        ai_data = df[df["Resource Accessed"] == "/aiassistant.php"]
        ai_volume = ai_data.groupby("Date").size().reset_index(name="Interactions")
        fig3 = px.line(ai_volume, x="Date", y="Interactions")
        st.plotly_chart(fig3, use_container_width=True)
 
elif page == "Sales & Geography":
    col1, col2 = st.columns(2)
 
    with col1:
        st.subheader("🌍 Sales Distribution by Country")
        country_sales = df[df["Sale Made"] == "Yes"].groupby("Country")["Sale Price"].sum().reset_index()
        fig4 = px.pie(country_sales, values="Sale Price", names="Country", hole=0.4)
        st.plotly_chart(fig4, use_container_width=True)
 
    with col2:
        st.subheader("💼 Sales by Team")
        team_sales = df[df["Sale Made"] == "Yes"].groupby("Sales Team Name")["Sale Price"].sum().reset_index()
        fig5 = px.bar(team_sales, x="Sales Team Name", y="Sale Price")
        st.plotly_chart(fig5, use_container_width=True)
 
elif page == "Session & Campaign Insights":
    col1, col2 = st.columns(2)
 
    with col1:
        st.subheader("⏱ Session Metrics")
        if "Session Duration" in df.columns:
            avg_time = df["Session Duration"].mean()
            st.metric("Avg. Session Time (sec)", f"{avg_time:.2f}")
        else:
            st.warning("Session Duration column is missing.")
 
        if "Session ID" in df.columns:
            pages_per_visit = df.groupby("Session ID")["Resource Accessed"].count().mean()
            st.metric("Pages per Visit", f"{pages_per_visit:.2f}")
        else:
            st.warning("Session ID column is missing.")
 
    with col2:
        st.subheader("📢 Campaign Conversion Rate")
        if "Sale Made" in df.columns:
            campaign_conv = df.groupby("Campaign Source")["Sale Made"].apply(lambda x: (x == "Yes").mean() * 100).reset_index()
            campaign_conv.columns = ["Campaign Source", "Conversion Rate (%)"]
            fig6 = px.bar(campaign_conv, x="Campaign Source", y="Conversion Rate (%)")
            st.plotly_chart(fig6, use_container_width=True)
        else:
            st.warning("Sale Made column is missing or misformatted.")
 
elif page == "AI Predictions":
    st.subheader("🧠 AI Model Predictions")

    col1, col2 = st.columns(2)

    # 👉 REGRESSION: Predict demo requests
    with col1:
        st.markdown("### 📈 Predict Demo Requests")
        with st.form("regression_form"):
            day = st.number_input("Day", min_value=1, max_value=31, value=15)
            month = st.number_input("Month", min_value=1, max_value=12, value=5)
            year = st.number_input("Year", min_value=2023, max_value=2025, value=2025)
            dow = st.number_input("Day of Week (0 = Mon, 6 = Sun)", min_value=0, max_value=6, value=2)
            reg_submit = st.form_submit_button("Predict Demo Volume")

        if reg_submit:
            demo_pred = reg_model.predict([[day, month, year, dow]])
            st.success(f"📊 Predicted Demo Requests: {int(demo_pred[0])}")

    # 👉 CLASSIFICATION: Predict interaction type
    with col2:
        st.markdown("### 🧠 Predict Interaction Type")
        with st.form("classification_form"):
            c_day = st.number_input("Day", min_value=1, max_value=31, value=15, key="c1")
            c_month = st.number_input("Month", min_value=1, max_value=12, value=5, key="c2")
            c_year = st.number_input("Year", min_value=2023, max_value=2025, value=2025, key="c3")
            c_dow = st.number_input("Day of Week (0 = Mon, 6 = Sun)", min_value=0, max_value=6, value=2, key="c4")
            clf_submit = st.form_submit_button("Predict Type")

        if clf_submit:
            interaction_pred = clf_model.predict([[c_day, c_month, c_year, c_dow]])
            label = clf_model.classes_[interaction_pred[0]]
            st.success(f"🔍 Predicted Interaction: {label}")
