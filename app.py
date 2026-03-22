import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .prediction-card {
        padding: 20px;
        border-radius: 10px;
        font-size: 18px;
        font-weight: bold;
        text-align: center;
    }
    .churn-positive {
        background-color: #ffcccc;
        color: #990000;
    }
    .churn-negative {
        background-color: #ccffcc;
        color: #009900;
    }
    </style>
""", unsafe_allow_html=True)

# ============ LOAD MODEL ============
@st.cache_resource
def load_model():
    try:
        with open('models/churn_model.pkl', 'rb') as f:
            pipeline = pickle.load(f)
            return pipeline
    except FileNotFoundError:
        st.error(" Model files not found! Please run 'python train.py' first.")
        st.stop()

pipeline = load_model()

# ============ SIDEBAR NAVIGATION ============
st.sidebar.title("🔮 Churn Predictor")
page = st.sidebar.radio(
    "Choose a section:",
    ["🎯 Make Prediction", "📊 Model Performance"]
)

# ============ PAGE 1: PREDICTION ============
if page == "🎯 Make Prediction":
    st.title("🔮 Customer Churn Prediction")
    st.write("Enter customer details to predict churn probability")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Demographics")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        
    with col2:
        st.subheader("Phone & Internet")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet_service = st.selectbox(
            "Internet Service Type",
            ["DSL", "Fiber optic", "No"]
        )
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Internet Add-ons")
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        
    with col4:
        st.subheader("Contract & Billing")
        contract = st.selectbox(
            "Contract Type",
            ["Month-to-month", "One year", "Two year"]
        )
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )
        monthly_charges = st.slider("Monthly Charges ($)", 0.0, 150.0, 65.0)
        total_charges = st.slider("Total Charges ($)", 0.0, 10000.0, 1500.0)
    
    # Make prediction
    if st.button("🚀 Predict Churn", key="predict_btn"):
        try:
            # Create dataframe with EXACT columns from dataset (excluding customerID and Churn)
            input_data = pd.DataFrame({
                'gender': [gender],
                'SeniorCitizen': [1 if senior_citizen == "Yes" else 0],
                'Partner': [partner],
                'Dependents': [dependents],
                'tenure': [tenure],
                'PhoneService': [phone_service],
                'MultipleLines': [multiple_lines],
                'InternetService': [internet_service],
                'OnlineSecurity': [online_security],
                'OnlineBackup': [online_backup],
                'DeviceProtection': [device_protection],
                'TechSupport': [tech_support],
                'StreamingTV': [streaming_tv],
                'StreamingMovies': [streaming_movies],
                'Contract': [contract],
                'PaperlessBilling': [paperless_billing],
                'PaymentMethod': [payment_method],
                'MonthlyCharges': [monthly_charges],
                'TotalCharges': [total_charges],
            })
            
            # Predict
            prediction = pipeline.predict(input_data)[0]
            probability = pipeline.predict_proba(input_data)[0]
            
            st.divider()
            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                if prediction == 1:
                    st.markdown(
                        f"""
                        <div class="prediction-card churn-positive">
                        ⚠️ WILL CHURN<br>
                        Probability: {probability[1]:.1%}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="prediction-card churn-negative">
                        ✅ WILL NOT CHURN<br>
                        Probability: {probability[0]:.1%}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            
            with col_res2:
                st.metric(
                    label="Prediction Confidence",
                    value=f"{max(probability) * 100:.1f}%",
                    delta=f"Risk: {'🔴 High' if probability[1] > 0.7 else '🟡 Medium' if probability[1] > 0.4 else '🟢 Low'}"
                )
            
            st.subheader("Probability Breakdown")
            prob_df = pd.DataFrame({
                'Class': ['No Churn', 'Churn'],
                'Probability': probability
            })
            st.bar_chart(prob_df.set_index('Class'))
            
            st.subheader("💡 Recommendations")
            if prediction == 1:
                st.warning("""
                **Customer at Risk - Action Required!**
                
                - 📞 Proactive outreach and engagement
                - 💰 Retention discounts or offers
                - 🎁 Service upgrades at reduced rates
                - 📧 Personalized campaigns
                """)
            else:
                st.success("""
                **Customer Satisfied - Growth Opportunity!**
                
                - ⭐ Upsell premium services
                - 🎁 Loyalty rewards program
                - 📊 Regular satisfaction checks
                """)
                
        except Exception as e:
            st.error(f" Error: {str(e)}")
            st.info("Ensure all features match the training data format.")

# ============ PAGE 2: MODEL PERFORMANCE ============
elif page == "📊 Model Performance":
    st.title("📊 Model Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", "~82%")
    with col2:
        st.metric("Precision", "~0.78")
    with col3:
        st.metric("Recall", "~0.65")
    with col4:
        st.metric("F1-Score", "~0.71")
    
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Confusion Matrix")
        if os.path.exists('models/confusion_matrix.png'):
            st.image(Image.open('models/confusion_matrix.png'))
        else:
            st.info("Run 'python train.py' first")
    
    with col2:
        st.subheader("ROC Curve")
        if os.path.exists('models/roc_curve.png'):
            st.image(Image.open('models/roc_curve.png'))
        else:
            st.info("Run 'python train.py' first")
