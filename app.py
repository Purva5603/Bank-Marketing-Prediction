import streamlit as st
import pandas as pd
import pickle
import base64

# -----------------------------------------------------------
# Function: Add a Corporate Bank Background Image
# -----------------------------------------------------------
def add_bg(image_file):
    with open(image_file, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode()

    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded_string}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        font-family: 'Segoe UI', sans-serif;
    }}

    /* Form Container */
    .block-container {{
        backdrop-filter: blur(4px);
        background: rgba(255, 255, 255, 0.80);
        padding: 2rem;
        border-radius: 12px;
        margin-top: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }}

    /* Titles */
    h1, h2, h3, h4 {{
        color: #003366;
        font-weight: 600;
    }}

    /* Buttons */
    .stButton>button {{
        background-color: #003366;
        color: white;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-size: 1rem;
        border: none;
        font-weight: 500;
    }}

    .stButton>button:hover {{
        background-color: #00509E;
        color: white;
        transition: 0.3s;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# -----------------------------------------------------------
# Add Background
# -----------------------------------------------------------
add_bg("bank.jpg")  # <--- Use your image here

# -----------------------------------------------------------
# Load Trained Model
# -----------------------------------------------------------
model = pickle.load(open("xgb_smote_pipeline.pkl", "rb"))

# -----------------------------------------------------------
# Corporate Header
# -----------------------------------------------------------
st.markdown("<h1 style='text-align:center;'>🏦 Term Deposit Prediction System</h1>", unsafe_allow_html=True)
# st.markdown("<h4 style='text-align:center; color:#003366;'>Powered by Machine Learning — Bank Marketing Decision Support</h4>", unsafe_allow_html=True)
st.markdown("---")

# -----------------------------------------------------------
# Input Form (Corporate Style)
# -----------------------------------------------------------
with st.form("input_form"):
    st.markdown("### 📋 Customer Information Form")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 18, 100, 30)
        job = st.selectbox("Job", [
            'admin.', 'blue-collar', 'technician', 'services', 'management',
            'retired', 'self-employed', 'student', 'unemployed', 'housemaid',
            'entrepreneur'
        ])
        marital = st.selectbox("Marital Status", ['married', 'single', 'divorced'])
        education = st.selectbox("Education", [
            'unknown', 'illiterate', 'basic.4y', 'basic.6y', 'basic.9y',
            'high.school', 'professional.course', 'university.degree'
        ])
        housing = st.selectbox("Housing Loan", ['no', 'yes'])
        loan = st.selectbox("Personal Loan", ['no', 'yes'])

    with col2:
        contact = st.selectbox("Contact Type", ['cellular', 'telephone'])
        month = st.selectbox("Month", [
            'jan', 'feb', 'mar', 'apr', 'may', 'jun',
            'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
        ])
        day_of_week = st.selectbox("Day of Week", ['mon', 'tue', 'wed', 'thu', 'fri'])
        poutcome = st.selectbox("Previous Outcome", ['unknown', 'success', 'failure'])
        campaign = st.number_input("Campaign Calls", 1, 60, 1)
        previous = st.number_input("Previous Contacts", 0, 30, 0)

    st.markdown("### 📊 Economic Indicators")
    col3, col4, col5 = st.columns(3)

    with col3:
        cons_price_idx = st.number_input("Consumer Price Index", 90.0, 100.0, 93.5)

    with col4:
        cons_conf_idx = st.number_input("Confidence Index", -60.0, -20.0, -40.0)

    with col5:
        euribor3m = st.number_input("Euribor 3M", 0.5, 8.0, 4.0)

    pdays_not_contacted = st.selectbox("Previously Contacted?", [0, 1])

    submitted = st.form_submit_button("Predict")

# -----------------------------------------------------------
# Prediction Logic
# -----------------------------------------------------------
if submitted:
    input_df = pd.DataFrame([{
        'age': age,
        'job': job,
        'marital': marital,
        'education': education,
        'housing': housing,
        'loan': loan,
        'contact': contact,
        'month': month,
        'day_of_week': day_of_week,
        'poutcome': poutcome,
        'campaign': campaign,
        'previous': previous,
        'cons.price.idx': cons_price_idx,
        'cons.conf.idx': cons_conf_idx,
        'euribor3m': euribor3m,
        'pdays_not_contacted': pdays_not_contacted
    }])

    pred_prob = model.predict_proba(input_df)[0][1]
    prediction = model.predict(input_df)[0]

    st.markdown("---")
    st.markdown("###  Prediction Result")

    if prediction == 1:
        st.success(f"✔ The customer is *LIKELY* to subscribe.\n\n Probability: *{pred_prob:.2f}*")
    else:
        st.error(f"✘ The customer is *NOT LIKELY* to subscribe.\n\n Probability: *{pred_prob:.2f}*")
