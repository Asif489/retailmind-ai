import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import load_data, clean_data
from src.rfm import create_rfm
from src.segmentation import segment_customers
from src.forecasting import prepare_sales, train_model
from src.recommendation import create_basket, generate_rules
from src.lstm_model import prepare_lstm, build_lstm
from src.churn import prepare_churn_data, train_churn_model, predict_churn
from src.insights import generate_insights
from src.explainability import (
    explain,
    plot_summary,
    plot_feature_importance,
    plot_waterfall,
    get_top_features
)

# =====================================================
# CUSTOM CSS
# =====================================================
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
.block-container {
    padding-top: 2rem;
}
.card {
    background: #1c1f26;
    padding: 18px;
    border-radius: 14px;
    text-align: center;
    box-shadow: 0 4px 10px rgba(0,0,0,.25);
}
.big {
    font-size: 28px;
    font-weight: bold;
    color: #00d4ff;
}
.small {
    font-size: 14px;
    color: #cccccc;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# CACHE FUNCTIONS (FAST APP)
# =====================================================
@st.cache_data
def get_data():
    df = load_data("data/Online Retail.xlsx")
    return clean_data(df)

@st.cache_data
def get_rfm(df):
    return create_rfm(df)

@st.cache_data
def get_sales(df):
    return prepare_sales(df)

@st.cache_resource
def get_segmentation_model(rfm):
    return segment_customers(rfm)

@st.cache_resource
def get_churn_model(X, y):
    return train_churn_model(X, y)

with st.spinner("Loading data..."):
    df = get_data()

# =====================================================
# LOAD DATA
# =====================================================
df = get_data()

# =====================================================
# TITLE
# =====================================================
st.title("AI-Powered Customer Intelligence Platform")
st.caption("Advanced Analytics • Forecasting • Churn • XAI")

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("📂 Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "Segmentation",
        "Forecast",
        "Recommendation",
        "Churn",
        "Explainability"
    ]
)

# =====================================================
# KPI SECTION
# =====================================================
def show_kpi(df):
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
        <div class="card">
            <div class="small">Revenue</div>
            <div class="big">${df['SALES'].sum():,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="card">
            <div class="small">Customers</div>
            <div class="big">{df['CUSTOMERNAME'].nunique()}</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="card">
            <div class="small">Orders</div>
            <div class="big">{df['ORDERNUMBER'].nunique()}</div>
        </div>
        """, unsafe_allow_html=True)

# =====================================================
# OVERVIEW
# =====================================================
if page == "Overview":

    st.header("📊 Business Overview")
    show_kpi(df)

    sales = get_sales(df)

    st.subheader("Monthly Sales Trend")
    st.line_chart(sales["SALES"])

    rfm = get_rfm(df)

    st.subheader("🤖 AI Insight")
    insight = generate_insights(df, rfm, sales)
    st.success(insight)

# =====================================================
# SEGMENTATION
# =====================================================
elif page == "Segmentation":

    st.header("📊 Customer Segmentation")

    rfm = get_rfm(df)

    if len(rfm) >= 3:

        rfm, model_seg, score = get_segmentation_model(rfm)

        st.metric("Silhouette Score", f"{score:.3f}")

        fig, ax = plt.subplots(figsize=(7,4))

        sns.scatterplot(
            data=rfm,
            x="Recency",
            y="Monetary",
            hue="Cluster",
            palette="Set2",
            s=60,
            ax=ax
        )

        ax.set_title("Customer Segments")
        st.pyplot(fig)

        st.dataframe(rfm.head())

    else:
        st.warning("Not enough customers for clustering.")

# =====================================================
# FORECAST
# =====================================================
elif page == "Forecast":

    st.header("📈 Sales Forecast")

    sales = get_sales(df)

    model_fc, mae = train_model(sales)

    st.metric("Model MAE", f"{mae:.2f}")

    # LSTM
    X_lstm, y_lstm, scaler = prepare_lstm(sales)

    lstm = build_lstm((X_lstm.shape[1], X_lstm.shape[2]))
    lstm.fit(X_lstm, y_lstm, epochs=5, verbose=0)

    pred = lstm.predict(
        X_lstm[-1].reshape(1, X_lstm.shape[1], X_lstm.shape[2]),
        verbose=0
    )

    pred = scaler.inverse_transform(pred)

    st.success(f"Next Predicted Sales: ${float(pred[0][0]):,.2f}")

# =====================================================
# RECOMMENDATION
# =====================================================
elif page == "Recommendation":

    st.header("🛒 Product Recommendation")

    basket = create_basket(df)
    rules = generate_rules(basket)

    if not rules.empty:
        st.dataframe(rules.head(10))
    else:
        st.warning("No recommendation rules found.")

# =====================================================
# CHURN
# =====================================================
elif page == "Churn":

    st.header("⚠️ Customer Churn Prediction")

    rfm = get_rfm(df)

    X, y, churn_df = prepare_churn_data(rfm)

    model_churn, report = get_churn_model(X, y)

    probs = predict_churn(model_churn, X)

    churn_df["Churn Probability"] = probs
    churn_df["Risk"] = churn_df["Churn Probability"].apply(
        lambda x: "High" if x > 0.70 else "Medium" if x > 0.40 else "Low"
    )

    st.dataframe(churn_df.head(20))

    st.subheader("Model Performance")
    st.json(report)

# =====================================================
# EXPLAINABILITY
# =====================================================
elif page == "Explainability":

    st.header("🧠 Explainable AI Dashboard")

    rfm = get_rfm(df)

    X, y, churn_df = prepare_churn_data(rfm)

    model_churn, _ = get_churn_model(X, y)

    explainer, shap_values, expected_value = explain(model_churn, X)

    # ------------------------
    # CUSTOMER SELECTOR
    # ------------------------
    index = st.slider(
        "Select Customer",
        min_value=0,
        max_value=len(X)-1,
        value=0
    )

    # ------------------------
    # WATERFALL
    # ------------------------
    st.subheader("🔍 Individual Explanation")

    fig1 = plot_waterfall(explainer, shap_values, X, index)
    st.pyplot(fig1)

    # ------------------------
    # GLOBAL IMPACT
    # ------------------------
    st.subheader("📊 Global Feature Impact")

    fig2 = plot_summary(shap_values, X)
    st.pyplot(fig2)

    # ------------------------
    # FEATURE IMPORTANCE
    # ------------------------
    st.subheader("📌 Feature Importance")

    fig3 = plot_feature_importance(shap_values, X)
    st.pyplot(fig3)

    # ------------------------
    # AI TEXT EXPLANATION
    # ------------------------
    st.subheader("🤖 AI Insight")

    top_features = get_top_features(shap_values, X, index)

    for feature, val in top_features:
        if val > 0:
            st.write(f"🔴 {feature} increases churn risk")
        else:
            st.write(f"🟢 {feature} reduces churn risk")