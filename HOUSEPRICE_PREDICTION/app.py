import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import time

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏡",
    layout="centered",
    initial_sidebar_state="expanded"
)

# =====================================
# CUSTOM CSS
# =====================================

st.markdown("""
<style>

.main {
    background-color: #f8fafc;
}

.title-box{
    background: linear-gradient(90deg,#2563eb,#06b6d4);
    padding:25px;
    border-radius:15px;
    text-align:center;
    color:white;
    margin-bottom:20px;
}

.prediction{
    background:#d1fae5;
    padding:20px;
    border-radius:12px;
    border-left:8px solid #10b981;
    text-align:center;
}

.metric-card{
    background:white;
    padding:18px;
    border-radius:12px;
    box-shadow:0px 2px 8px rgba(0,0,0,0.08);
}

.stButton>button{
    width:100%;
    background:#2563eb;
    color:white;
    border:none;
    border-radius:10px;
    height:50px;
    font-size:18px;
    font-weight:bold;
}

.stButton>button:hover{
    background:#1d4ed8;
}

div[data-testid="stNumberInput"]{
    background:white;
    padding:10px;
    border-radius:10px;
}

.footer{
    text-align:center;
    color:gray;
    margin-top:40px;
}

</style>
""", unsafe_allow_html=True)

# =====================================
# HEADER
# =====================================

st.markdown("""
<div class="title-box">
<h1>🏡 House Price Prediction</h1>
<p>Predict house prices using Machine Learning (Linear Regression)</p>
</div>
""", unsafe_allow_html=True)

# =====================================
# SIDEBAR
# =====================================

st.sidebar.title("📌 Project Info")

st.sidebar.info("""
### Model
- Linear Regression

### Feature
- House Area

### Target
- House Price

### ML Workflow
Dataset ➜ Training ➜ Prediction
""")

# =====================================
# LOAD DATA
# =====================================

df = pd.read_csv("houseprice.csv")

X = df.drop("price", axis=1)
y = df["price"]

model = LinearRegression()
model.fit(X, y)

# =====================================
# METRICS
# =====================================

col1, col2 = st.columns(2)

with col1:
    st.metric("📊 Total Houses", len(df))

with col2:
    st.metric("💰 Average Price", f"₹ {df['price'].mean():,.0f}")

# =====================================
# DATASET
# =====================================

with st.expander("📂 View Dataset"):
    st.dataframe(df, use_container_width=True)

# =====================================
# USER INPUT
# =====================================

st.subheader("🏠 Enter House Details")

area = st.slider(
    "House Area (Square Feet)",
    500,
    10000,
    3300,
    100
)

st.write(f"Selected Area: **{area:,} sq ft**")

# =====================================
# PREDICTION
# =====================================

if st.button("🔍 Predict House Price"):

    with st.spinner("Calculating Prediction..."):
        time.sleep(1.5)

    prediction = model.predict([[area]])

    st.balloons()

    st.markdown(f"""
    <div class="prediction">
        <h2>💰 Estimated House Price</h2>
        <h1>₹ {prediction[0]:,.2f}</h1>
    </div>
    """, unsafe_allow_html=True)

# =====================================
# MODEL DETAILS
# =====================================

st.subheader("📈 Model Information")

c1, c2 = st.columns(2)

with c1:
    st.metric("Coefficient", f"{model.coef_[0]:,.2f}")

with c2:
    st.metric("Intercept", f"{model.intercept_:,.2f}")

# =====================================
# FOOTER
# =====================================

st.markdown("""
<div class="footer">
Made by Anushka ⭐ using Streamlit & Scikit-Learn
</div>
""", unsafe_allow_html=True)
