import streamlit as st
import pandas as pd
import joblib

# Load Model
model = joblib.load("house_model.pkl")
columns = joblib.load("columns.pkl")

# Page Configuration
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

# Custom Light UI
st.markdown("""
<style>

.main-title{
    text-align:center;
    font-size:48px;
    font-weight:bold;
    color:#1e3a8a;
    margin-bottom:5px;
}

.sub-title{
    text-align:center;
    color:#64748b;
    font-size:18px;
    margin-bottom:30px;
}

.input-card{
    background-color:#f8fafc;
    padding:25px;
    border-radius:20px;
    border:1px solid #e2e8f0;
    box-shadow:0px 2px 10px rgba(0,0,0,0.05);
}

.result-card{
    background:#dbeafe;
    padding:25px;
    border-radius:20px;
    text-align:center;
    margin-top:20px;
    border:1px solid #93c5fd;
}

.result-price{
    color:#1e40af;
    font-size:40px;
    font-weight:bold;
}

div.stButton > button{
    width:100%;
    height:55px;
    font-size:18px;
    font-weight:bold;
    border-radius:12px;
    background-color:#2563eb;
    color:white;
    border:none;
}

div.stButton > button:hover{
    background-color:#1d4ed8;
}

</style>
""", unsafe_allow_html=True)

# Heading
st.markdown(
    "<div class='main-title'>🏠 House Price Prediction</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='sub-title'>Enter House Details</div>",
    unsafe_allow_html=True
)

# Input Card
st.markdown("<div class='input-card'>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    lotarea = st.number_input(
        "Lot Area",
        min_value=1000,
        value=8000
    )

    yearbuilt = st.number_input(
        "Year Built",
        min_value=1900,
        max_value=2025,
        value=2000
    )

with col2:
    overallcond = st.selectbox(
        "Overall Condition",
        [1,2,3,4,5,6,7,8,9,10]
    )

    totalbsmt = st.number_input(
        "Basement Area",
        min_value=0,
        value=1000
    )

yearremod = st.number_input(
    "Year Remodeled",
    min_value=1900,
    max_value=2025,
    value=2000
)

st.markdown("</div>", unsafe_allow_html=True)

st.write("")

# Prediction
if st.button("🔮 Predict House Price"):

    sample = pd.DataFrame(
        [[0] * len(columns)],
        columns=columns
    )

    if "LotArea" in columns:
        sample["LotArea"] = lotarea

    if "OverallCond" in columns:
        sample["OverallCond"] = overallcond

    if "YearBuilt" in columns:
        sample["YearBuilt"] = yearbuilt

    if "YearRemodAdd" in columns:
        sample["YearRemodAdd"] = yearremod

    if "TotalBsmtSF" in columns:
        sample["TotalBsmtSF"] = totalbsmt

    prediction = model.predict(sample)

    st.markdown(
        f"""
        <div class="result-card">
            <div style="font-size:20px; color:#1e40af;">
                Estimated House Price
            </div>
            <div class="result-price">
                ₹ {prediction[0]:,.0f}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )