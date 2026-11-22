
import streamlit as st
import pandas as pd
import joblib
import time

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="CARDIO AI | Heart Prediction",
    page_icon="💜",
    layout="wide"
)

# ==========================================
# 2. LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():
    return joblib.load("heart_model.pkl")


try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file not found! Run python train_model.py first.")
    st.stop()


# ==========================================
# 3. CUSTOM CSS
# ==========================================

st.markdown("""
<style>

/* Overall page */

.stApp {
    background: #F5F2FC;
    color: #38304B;
}

/* Main container */

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Header */

.header-box {
    text-align: center;
    padding: 20px 10px 25px;
}

.heart-icon {
    font-size: 55px;
    animation: heartbeat 1.6s infinite;
}

.main-title {
    color: #8569BD;
    font-size: 43px;
    font-weight: 800;
    letter-spacing: 3px;
    margin: 5px 0;
}

.subtitle {
    color: #817595;
    font-size: 16px;
}

/* Heart animation */

@keyframes heartbeat {
    0%, 100% { transform: scale(1); }
    25% { transform: scale(1.12); }
    40% { transform: scale(1); }
    60% { transform: scale(1.08); }
}

/* Section headings */

.section-title {
    color: #8268B5;
    font-size: 23px;
    font-weight: 750;
    margin: 20px 0 15px;
}

/* Input labels */

.stApp label,
.stApp [data-testid="stWidgetLabel"] p {
    color: #514665 !important;
    font-weight: 600 !important;
}

/* Text and number inputs */

.stApp input {
    background-color: #FBFAFF !important;
    color: #38304B !important;
    border: 1px solid #DDD3F0 !important;
    border-radius: 10px !important;
}

/* Select boxes */

.stApp [data-baseweb="select"] > div {
    background-color: #FBFAFF !important;
    border: 1px solid #DDD3F0 !important;
    border-radius: 10px !important;
}

.stApp [data-baseweb="select"] span {
    color: #38304B !important;
}

/* Dropdown menu */

div[role="listbox"] {
    background-color: #FFFFFF !important;
}

div[role="option"] {
    background-color: #FFFFFF !important;
    color: #38304B !important;
}

div[role="option"]:hover {
    background-color: #EEE8FA !important;
}

/* Number input buttons */

.stApp [data-testid="stNumberInput"] button {
    background-color: #EAE3F8 !important;
    color: #7559A9 !important;
    border: 1px solid #D9CDEE !important;
}

/* Focus effect */

.stApp input:focus,
.stApp [data-baseweb="select"] > div:focus-within {
    border-color: #9B82D0 !important;
    box-shadow: 0 0 0 2px rgba(155,130,208,0.15) !important;
}

/* Predict button */

.stButton > button,
[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(
        100deg,
        #9B83CE,
        #8066B8
    ) !important;

    color: #FFFFFF !important;
    border: none !important;
    border-radius: 12px !important;
    min-height: 48px;
    font-size: 16px !important;
    font-weight: 700 !important;
    width: 100% !important;
    transition: all 0.3s ease !important;
}

.stButton > button:hover,
[data-testid="stFormSubmitButton"] button:hover {
    background: #7056A8 !important;
    color: #FFFFFF !important;
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(128,102,184,0.25);
}

/* Result card */

.result-card {
    background: #FFFFFF;
    border: 1px solid #E5DCF4;
    border-radius: 22px;
    padding: 38px 25px;
    text-align: center;
    margin-top: 18px;
    box-shadow: 0 8px 30px rgba(116,91,161,0.09);
    animation: reveal 0.8s ease;
}

@keyframes reveal {
    from {
        opacity: 0;
        transform: translateY(18px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.result-icon {
    font-size: 58px;
    animation: heartbeat 1.6s infinite;
}

.result-heading {
    color: #8569BD;
    font-size: 27px;
    font-weight: 750;
}

.result-description {
    color: #777087;
    font-size: 15px;
    margin-top: 12px;
}

/* Footer */

.footer {
    text-align: center;
    color: #9186A5;
    font-size: 13px;
    padding: 25px 10px;
    margin-top: 35px;
    border-top: 1px solid #E5DCF4;
}

/* Dividers */

hr {
    border-color: #E4DCEF !important;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 4. HEADER
# ==========================================

st.html("""
<div class="header-box">

    <div class="heart-icon">💜</div>

    <div class="main-title">CARDIO AI</div>

    <div class="subtitle">
        Intelligent Heart Disease Prediction System
    </div>

</div>
""")

st.markdown("---")

st.markdown("""
<div style="
    text-align:center;
    color:#817595;
    font-size:16px;
    padding:10px;
">
    Welcome! Enter the required details below
    to explore the machine learning prediction.
</div>
""", unsafe_allow_html=True)


# ==========================================
# 5. INPUT FORM
# ==========================================

st.markdown(
    '<div class="section-title">✦ Enter Your Details</div>',
    unsafe_allow_html=True
)

with st.form("prediction_form"):

    col1, col2, col3 = st.columns(3)

    with col1:

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=45
        )

        sex = st.selectbox(
            "Sex",
            ["M", "F"]
        )

        chest_pain = st.selectbox(
            "Chest Pain Type",
            ["ATA", "NAP", "ASY", "TA"]
        )

        resting_bp = st.number_input(
            "Resting Blood Pressure",
            min_value=1,
            max_value=250,
            value=120
        )

    with col2:

        cholesterol = st.number_input(
            "Cholesterol",
            min_value=0,
            max_value=700,
            value=200
        )

        fasting_bs = st.selectbox(
            "Fasting Blood Sugar",
            ["No", "Yes"]
        )

        resting_ecg = st.selectbox(
            "Resting ECG",
            ["Normal", "ST", "LVH"]
        )

        max_hr = st.number_input(
            "Maximum Heart Rate",
            min_value=1,
            max_value=250,
            value=150
        )

    with col3:

        exercise_angina = st.selectbox(
            "Exercise-Induced Angina",
            ["N", "Y"]
        )

        oldpeak = st.number_input(
            "Oldpeak",
            min_value=-5.0,
            max_value=10.0,
            value=0.0,
            step=0.1
        )

        st_slope = st.selectbox(
            "ST Slope",
            ["Up", "Flat", "Down"]
        )

    submitted = st.form_submit_button(
        "✦  Predict Now"
    )


# ==========================================
# 6. PREDICTION
# ==========================================

if submitted:

    input_data = pd.DataFrame({

        "Age": [age],
        "Sex": [sex],
        "ChestPainType": [chest_pain],
        "RestingBP": [resting_bp],
        "Cholesterol": [cholesterol],

        "FastingBS": [
            1 if fasting_bs == "Yes" else 0
        ],

        "RestingECG": [resting_ecg],
        "MaxHR": [max_hr],
        "ExerciseAngina": [exercise_angina],
        "Oldpeak": [oldpeak],
        "ST_Slope": [st_slope]

    })

    # Loading animation

    with st.spinner("💜 Analyzing your input data..."):
        time.sleep(1.2)
        prediction = model.predict(input_data)[0]

    st.markdown("---")

    st.markdown(
        '<div class="section-title">✦ Prediction Result</div>',
        unsafe_allow_html=True
    )


    # ======================================
    # 7. RESULT DISPLAY
    # ======================================

    if prediction == 1:

        st.html("""
        <div class="result-card"
             style="border-top:5px solid #D9829A;">

            <div class="result-icon">💗</div>

            <h2 style="color:#C36E89;">
                Prediction Complete
            </h2>

            <h1 style="color:#493B59;">
                Heart Disease Predicted
            </h1>

            <p class="result-description">
                The model predicts class 1
                for the entered values.
            </p>

        </div>
        """)

    else:

        st.html("""
        <div class="result-card"
             style="border-top:5px solid #83B99A;">

            <div class="result-icon">💚</div>

            <h2 style="color:#679D7C;">
                Prediction Complete
            </h2>

            <h1 style="color:#493B59;">
                No Heart Disease Predicted
            </h1>

            <p class="result-description">
                The model predicts class 0
                for the entered values.
            </p>

        </div>
        """)

    st.html("""
    <div style="
        text-align:center;
        color:#9186A5;
        font-size:13px;
        margin-top:20px;
    ">
        Educational demonstration only.
        This is not a medical diagnosis.
    </div>
    """)


# ==========================================
# 8. FOOTER
# ==========================================

st.html("""
<div class="footer">

    <p>💜 CARDIO AI | Machine Learning Project</p>

    <p>
        Built using Python, Streamlit and Scikit-learn
    </p>

</div>
""")
