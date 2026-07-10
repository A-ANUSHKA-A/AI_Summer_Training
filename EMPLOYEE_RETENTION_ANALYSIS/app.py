import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

# -------------------------------
# Page Configuration
# -------------------------------

st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="🏢",
    layout="wide"
)

# -------------------------------
# Custom CSS
# -------------------------------

st.markdown("""
<style>
.main{
    background-color:#f8fafc;
}
.title{
    font-size:42px;
    font-weight:bold;
    color:#0f172a;
}
.subtitle{
    color:gray;
    font-size:18px;
}
.metric{
    background:#ffffff;
    padding:15px;
    border-radius:10px;
    box-shadow:0px 0px 10px rgba(0,0,0,0.08);
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Header
# -------------------------------

st.markdown("<div class='title'>🏢 Employee Attrition Prediction</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Predict whether an employee is likely to leave the company using Machine Learning.</div>", unsafe_allow_html=True)

st.divider()

# -------------------------------
# Sidebar
# -------------------------------

st.sidebar.header("📂 Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload HR_comma_sep.csv",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Upload the HR dataset to begin.")
    st.stop()

# -------------------------------
# Read Dataset
# -------------------------------

df = pd.read_csv(uploaded_file)

# -------------------------------
# Dataset Preview
# -------------------------------

st.subheader("📋 Dataset Preview")

st.dataframe(df.head(), use_container_width=True)

# -------------------------------
# Dataset Information
# -------------------------------

col1, col2, col3 = st.columns(3)

col1.metric("Rows", df.shape[0])
col2.metric("Columns", df.shape[1])
col3.metric("Missing Values", df.isnull().sum().sum())

st.divider()

# -------------------------------
# Charts
# -------------------------------

st.subheader("📊 Data Visualization")

col1, col2 = st.columns(2)

with col1:

    fig, ax = plt.subplots(figsize=(6,4))
    sns.countplot(data=df, x="left", palette="viridis", ax=ax)
    ax.set_title("Employee Attrition")
    st.pyplot(fig)

with col2:

    fig, ax = plt.subplots(figsize=(6,4))
    pd.crosstab(df.salary, df.left).plot(kind="bar", ax=ax)
    ax.set_title("Salary vs Attrition")
    st.pyplot(fig)

st.divider()

# -------------------------------
# Model Training
# -------------------------------

features = [
    "satisfaction_level",
    "average_montly_hours",
    "promotion_last_5years",
    "salary"
]

X = df[features]
y = df["left"]

categorical = ["salary"]
numeric = [
    "satisfaction_level",
    "average_montly_hours",
    "promotion_last_5years"
]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(drop="first"), categorical),
    ("num", "passthrough", numeric)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

st.subheader("📈 Model Performance")

st.success(f"Model Accuracy : {accuracy*100:.2f}%")

col1, col2 = st.columns(2)

with col1:

    st.write("### Confusion Matrix")

    cm = confusion_matrix(y_test, predictions)

    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm,
                annot=True,
                cmap="Blues",
                fmt="d",
                ax=ax)

    st.pyplot(fig)

with col2:

    st.write("### Classification Report")

    report = classification_report(
        y_test,
        predictions,
        output_dict=True
    )

    st.dataframe(pd.DataFrame(report).transpose())

st.divider()

# -------------------------------
# Prediction Section
# -------------------------------

st.subheader("🤖 Predict Employee Attrition")

col1, col2 = st.columns(2)

with col1:

    satisfaction = st.slider(
        "Satisfaction Level",
        0.0,
        1.0,
        0.50
    )

    hours = st.slider(
        "Average Monthly Hours",
        90,
        320,
        180
    )

with col2:

    promotion = st.selectbox(
        "Promotion in Last 5 Years",
        [0,1]
    )

    salary = st.selectbox(
        "Salary",
        ["low","medium","high"]
    )

if st.button("🔍 Predict"):

    sample = pd.DataFrame({

        "satisfaction_level":[satisfaction],
        "average_montly_hours":[hours],
        "promotion_last_5years":[promotion],
        "salary":[salary]

    })

    prediction = model.predict(sample)[0]
    probability = model.predict_proba(sample)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error("⚠️ Employee is likely to Leave.")

    else:

        st.success("✅ Employee is likely to Stay.")

    st.progress(float(probability))

    st.write(f"Probability of Leaving : **{probability:.2%}**")

st.divider()

# -------------------------------
# Footer
# -------------------------------

st.markdown(
"""
### 💻 Technologies Used

- Streamlit
- Scikit-Learn
- Pandas
- NumPy
- Matplotlib
- Seaborn

---
Made ANUSHKA⭐ using Machine Learning
"""
)
