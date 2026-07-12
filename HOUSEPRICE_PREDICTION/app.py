import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from streamlit_extras.let_it_rain import rain
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

.main{
    background-color:#f8fafc;
}

.title-box{
    background:linear-gradient(90deg,#2563eb,#06b6d4);
    padding:25px;
    border-radius:15px;
    text-align:center;
    color:white;
    margin-bottom:20px;
}

.prediction{
    background:#dcfce7;
    padding:25px;
    border-radius:15px;
    border-left:8px solid #16a34a;
    text-align:center;
    box-shadow:0px 4px 12px rgba(0,0,0,0.08);
}

.footer{
    text-align:center;
    color:gray;
    margin-top:40px;
}

.stButton>button{
    width:100%;
    background:#2563eb;
    color:white;
    height:50px;
    border-radius:10px;
    font-size:18px;
    font-weight:bold;
    border:none;
}

.stButton>button:hover{
    background:#1d4ed8;
}

</style>
""", unsafe_allow_html=True)

# =====================================
# HEADER
# =====================================

st.markdown("""
<div class="title-box">
<h1>🏡 House Price Prediction</h1>
<h4>Predict House Prices using Linear Regression</h4>
</div>
""", unsafe_allow_html=True)

# =====================================
# SIDEBAR
# =====================================

st.sidebar.title("📌 Project Details")

st.sidebar.success("""
### Machine Learning Model
✅ Linear Regression

### Input Feature
🏠 House Area

### Output
💰 Estimated House Price
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
    st.metric("🏠 Houses", len(df))

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

st.subheader("🏠 Enter House Area")

area = st.slider(
    "Area (Square Feet)",
    min_value=500,
    max_value=10000,
    value=3300,
    step=100
)

st.write(f"Selected Area: **{area:,} sq ft**")

# =====================================
# PREDICTION
# =====================================

if st.button("🔍 Predict House Price"):

    with st.spinner("Predicting..."):
        time.sleep(1.5)

    prediction = model.predict([[area]])

    # Floating Emoji Animation
    rain(
        emoji="⭐",
        font_size=42,
        falling_speed=5,
        animation_length="4"
    )

    st.markdown(f"""
    <div class="prediction">
        <h2>💰 Estimated Price</h2>
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
<hr>
<div class="footer">
Made by ANUSHKA ⭐ using Streamlit | Scikit-Learn | Python
</div>
""", unsafe_allow_html=True)
