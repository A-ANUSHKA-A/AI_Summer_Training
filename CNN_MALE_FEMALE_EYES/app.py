import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time

# =====================================
# PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="Male vs Female Eye Classifier",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================
# CUSTOM CSS
# =====================================

st.markdown("""
<style>

html, body, [class*="css"]{
    font-family: 'Segoe UI', sans-serif;
}

.stApp{
    background:
    radial-gradient(circle at top left,#3b82f6 0%,transparent 25%),
    radial-gradient(circle at bottom right,#8b5cf6 0%,transparent 25%),
    linear-gradient(135deg,#07111f,#0b1b33,#131c4d);
    color:white;
}

/* Hide Streamlit Branding */

#MainMenu{
visibility:hidden;
}

footer{
visibility:hidden;
}

header{
visibility:hidden;
}

/* Hero Card */

.hero{

padding:40px;

border-radius:25px;

background:linear-gradient(
90deg,
rgba(37,99,235,.95),
rgba(124,58,237,.95),
rgba(6,182,212,.95)
);

box-shadow:
0px 10px 40px rgba(0,0,0,.35);

text-align:center;

margin-bottom:30px;

animation:fade 1.2s ease;
}

.hero h1{

font-size:54px;

margin-bottom:10px;

color:white;
}

.hero p{

font-size:20px;

color:#f5f5f5;

}

/* Glass Card */

.glass{

background:rgba(255,255,255,.08);

backdrop-filter:blur(16px);

padding:25px;

border-radius:20px;

border:1px solid rgba(255,255,255,.12);

box-shadow:0 10px 30px rgba(0,0,0,.25);

margin-bottom:20px;

transition:0.4s;
}

.glass:hover{

transform:translateY(-6px);

box-shadow:0 20px 45px rgba(59,130,246,.35);

}

/* Buttons */

.stButton>button{

width:100%;

height:55px;

border:none;

border-radius:15px;

font-size:18px;

font-weight:bold;

background:linear-gradient(
90deg,
#2563eb,
#7c3aed
);

color:white;

transition:.3s;
}

.stButton>button:hover{

transform:scale(1.02);

box-shadow:0 0 20px #3b82f6;
}

/* Upload */

[data-testid="stFileUploader"]{

background:rgba(255,255,255,.05);

padding:15px;

border-radius:15px;

border:2px dashed #60a5fa;

}

/* Metrics */

.metric-card{

background:linear-gradient(
135deg,
rgba(37,99,235,.8),
rgba(124,58,237,.8)
);

padding:20px;

border-radius:18px;

text-align:center;

color:white;

box-shadow:0 10px 30px rgba(0,0,0,.25);

}

/* Animation */

@keyframes fade{

from{

opacity:0;

transform:translateY(-25px);

}

to{

opacity:1;

transform:translateY(0);

}

}

</style>

""", unsafe_allow_html=True)

# =====================================
# SIDEBAR
# =====================================

with st.sidebar:

    st.image(
        "https://img.icons8.com/fluency/96/visible.png",
        width=90
    )

    st.title("👁️ Eye Classifier")

    st.markdown("---")

    st.success("### CNN Deep Learning Model")

    st.write("This application predicts whether an uploaded eye image belongs to a **Male** or **Female** using a trained CNN model.")

    st.markdown("### 📚 Technologies")

    st.markdown("""
- TensorFlow
- Keras
- Streamlit
- NumPy
- Pillow
""")

    st.markdown("---")

    st.info(
        """
📌 Upload a clear eye image.

Supported formats:

• JPG

• JPEG

• PNG
"""
    )

# =====================================
# HERO SECTION
# =====================================

st.markdown("""

<div class="hero">

<h1>👁️ Male vs Female Eye Classifier</h1>

<p>

Deep Learning • CNN • TensorFlow • Streamlit

</p>

</div>

""", unsafe_allow_html=True)

# =====================================
# FEATURE CARDS
# =====================================

col1,col2,col3=st.columns(3)

with col1:

    st.markdown("""

<div class="glass">

<h2 align="center">⚡ Fast</h2>

<p align="center">

Instant predictions powered by CNN.

</p>

</div>

""",unsafe_allow_html=True)

with col2:

    st.markdown("""

<div class="glass">

<h2 align="center">🎯 Accurate</h2>

<p align="center">

Classifies Male and Female eye images.

</p>

</div>

""",unsafe_allow_html=True)

with col3:

    st.markdown("""

<div class="glass">

<h2 align="center">🚀 AI Powered</h2>

<p align="center">

TensorFlow + Keras Deep Learning Model.

</p>

</div>

""",unsafe_allow_html=True)

st.write("")
st.write("")
# =====================================
# LOAD CNN MODEL
# =====================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.keras")

try:
    model = load_model()
except Exception:
    st.error("❌ model.keras not found.")
    st.stop()

# =====================================
# MAIN SECTION
# =====================================

st.markdown("## 📤 Upload Eye Image")

left, right = st.columns([1, 1], gap="large")

# -----------------------------
# LEFT COLUMN
# -----------------------------

with left:

    st.markdown("""
    <div class="glass">
    <h2 align="center">📁 Upload Image</h2>
    <p align="center">
    Drag & Drop or Click Browse
    </p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "",
        type=["jpg", "jpeg", "png"]
    )

# -----------------------------
# RIGHT COLUMN
# -----------------------------

with right:

    st.markdown("""
    <div class="glass">
    <h2 align="center">🖼 Image Preview</h2>
    </div>
    """, unsafe_allow_html=True)

    if uploaded_file is None:

        st.info("Upload an eye image to preview.")

    else:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            use_container_width=True
        )

# =====================================
# IMAGE DETAILS
# =====================================

if uploaded_file is not None:

    width, height = image.size

    size = uploaded_file.size / 1024

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
        <div class="metric-card">
        <h3>{width} × {height}</h3>
        Resolution
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card">
        <h3>{size:.1f} KB</h3>
        File Size
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card">
        <h3>{uploaded_file.type.split('/')[-1].upper()}</h3>
        Format
        </div>
        """, unsafe_allow_html=True)

# =====================================
# PREPROCESS IMAGE
# =====================================

if uploaded_file is not None:

    img = image.resize((299, 299))      # <-- Change if your model uses another size

    img = np.array(img).astype("float32") / 255.0

    img = np.expand_dims(img, axis=0)

# =====================================
# ANALYZE BUTTON
# =====================================

st.write("")

analyze = st.button("🚀 Analyze Image")

if analyze and uploaded_file is None:

    st.warning("Please upload an image first.")

if analyze and uploaded_file is not None:

    with st.spinner("🧠 CNN is analyzing the eye image..."):

        time.sleep(1.5)

    st.success("✅ Image processed successfully.")

    # Prediction comes in Part 3
# =====================================
# PREDICTION
# =====================================

if analyze and uploaded_file is not None:

    # -------------------------
    # Prediction Timer
    # -------------------------

    start_time = time.time()

    prediction = model.predict(img, verbose=0)

    end_time = time.time()

    inference_time = end_time - start_time

    probability = float(prediction[0][0])

    # ------------------------------------
    # CLASS LABEL
    # ------------------------------------

    if probability >= 0.5:

        label = "👨 Male Eye"

        confidence = probability

        male_prob = probability * 100

        female_prob = (1 - probability) * 100

        theme_color = "#2563eb"

    else:

        label = "👩 Female Eye"

        confidence = 1 - probability

        male_prob = probability * 100

        female_prob = (1 - probability) * 100

        theme_color = "#ec4899"

    # ------------------------------------
    # AI CONFIDENCE LEVEL
    # ------------------------------------

    confidence_percent = confidence * 100

    if confidence_percent >= 95:

        ai_level = "🟢 Excellent"

    elif confidence_percent >= 85:

        ai_level = "🟢 High"

    elif confidence_percent >= 70:

        ai_level = "🟡 Medium"

    else:

        ai_level = "🔴 Low"

    # ------------------------------------
    # RESULT CARD
    # ------------------------------------

    st.write("")

    st.markdown(f"""
    <div style="
        background:linear-gradient(135deg,{theme_color},#7c3aed);
        padding:35px;
        border-radius:25px;
        text-align:center;
        color:white;
        box-shadow:0px 15px 40px rgba(0,0,0,.35);
    ">

    <h3>🎯 Prediction Result</h3>

    <h1>{label}</h1>

    <h2>{confidence_percent:.2f}% Confidence</h2>

    </div>

    """, unsafe_allow_html=True)

    # ------------------------------------
    # SUCCESS EFFECT
    # ------------------------------------

    if confidence_percent >= 95:

        st.balloons()

    # ------------------------------------
    # CONFIDENCE BAR
    # ------------------------------------

    st.write("")

    st.subheader("🧠 AI Confidence")

    st.progress(float(confidence))

    st.write(f"### {confidence_percent:.2f}%")

    st.caption(f"Confidence Level : **{ai_level}**")

    # ------------------------------------
    # PROBABILITY SECTION
    # ------------------------------------

    st.write("")

    st.subheader("📊 Prediction Probability")

    st.write("👨 Male")

    st.progress(male_prob / 100)

    st.write(f"{male_prob:.2f}%")

    st.write("")

    st.write("👩 Female")

    st.progress(female_prob / 100)

    st.write(f"{female_prob:.2f}%")

    # ------------------------------------
    # METRICS
    # ------------------------------------

    st.write("")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Prediction",
        label
    )

    c2.metric(
        "Confidence",
        f"{confidence_percent:.2f}%"
    )

    c3.metric(
        "Inference Time",
        f"{inference_time:.3f} sec"
    )

    # ------------------------------------
    # SUMMARY CARD
    # ------------------------------------

    st.write("")

    st.markdown("""
    ## 📋 AI Prediction Summary
    """)

    st.info(f"""

Prediction : **{label}**

Confidence : **{confidence_percent:.2f}%**

Inference Time : **{inference_time:.3f} sec**

Model : **TensorFlow CNN**

Image Size : **299 × 299**

Classes : **Male • Female**

""")
if analyze and uploaded_file is not None:    
    # ============================================
    # DASHBOARD
    # ============================================
    
    st.write("")
    st.markdown("---")
    st.write("")
    
    st.markdown("""
    <h2 style='text-align:center;color:white;'>
    📊 AI Dashboard
    </h2>
    """, unsafe_allow_html=True)
    
    # ============================================
    # METRIC CARDS
    # ============================================
    
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.markdown(f"""
        <div style="
        background:linear-gradient(135deg,#2563eb,#1d4ed8);
        padding:25px;
        border-radius:18px;
        text-align:center;
        color:white;
        box-shadow:0px 10px 25px rgba(0,0,0,.3);
        ">
        <h1>🎯</h1>
        <h2>{label}</h2>
        <p>Prediction</p>
        </div>
        """, unsafe_allow_html=True)
    
    with c2:
        st.markdown(f"""
        <div style="
        background:linear-gradient(135deg,#7c3aed,#6d28d9);
        padding:25px;
        border-radius:18px;
        text-align:center;
        color:white;
        box-shadow:0px 10px 25px rgba(0,0,0,.3);
        ">
        <h1>🧠</h1>
        <h2>{confidence_percent:.2f}%</h2>
        <p>Confidence</p>
        </div>
        """, unsafe_allow_html=True)
    
    with c3:
        st.markdown(f"""
        <div style="
        background:linear-gradient(135deg,#0891b2,#06b6d4);
        padding:25px;
        border-radius:18px;
        text-align:center;
        color:white;
        box-shadow:0px 10px 25px rgba(0,0,0,.3);
        ">
        <h1>⚡</h1>
        <h2>{inference_time:.3f}s</h2>
        <p>Prediction Time</p>
        </div>
        """, unsafe_allow_html=True)
    
    with c4:
        st.markdown("""
        <div style="
        background:linear-gradient(135deg,#16a34a,#22c55e);
        padding:25px;
        border-radius:18px;
        text-align:center;
        color:white;
        box-shadow:0px 10px 25px rgba(0,0,0,.3);
        ">
        <h1>🤖</h1>
        <h2>CNN</h2>
        <p>Deep Learning</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.write("")
    st.write("")
    
    # ============================================
    # TWO COLUMN LAYOUT
    # ============================================
    
    left, right = st.columns([1.2, 1])
    
    # ============================================
    # MODEL INFORMATION
    # ============================================
    
    with left:
    
        st.markdown("""
        <div class="glass">
        <h2>🧬 Model Information</h2>
        </div>
        """, unsafe_allow_html=True)
    
        info1, info2 = st.columns(2)
    
        with info1:
    
            st.metric("Framework", "TensorFlow")
    
            st.metric("Architecture", "CNN")
    
            st.metric("Classes", "2")
    
            st.metric("Input Size", "299×299")
    
        with info2:
    
            st.metric("Image Type", "RGB")
    
            st.metric("Normalization", "255")
    
            st.metric("Output", "Sigmoid")
    
            st.metric("Prediction", "Binary")
    
    # ============================================
    # DATASET PANEL
    # ============================================
    
    with right:
    
        st.markdown("""
        <div class="glass">
        <h2>📚 Dataset</h2>
        </div>
        """, unsafe_allow_html=True)
    
        st.success("✔ Male Eye Images")
    
        st.success("✔ Female Eye Images")
    
        st.info("✔ Binary Classification")
    
        st.info("✔ CNN Training")
    
        st.metric("Classes", "2")
    
        st.metric("Model", "model.keras")
    
    st.write("")
    st.write("")
    
    # ============================================
    # CONFIDENCE VISUALIZATION
    # ============================================
    
    st.markdown("""
    <div class="glass">
    <h2 align="center">📈 Confidence Visualization</h2>
    </div>
    """, unsafe_allow_html=True)
    
    male_col, female_col = st.columns(2)
    
    with male_col:
    
        st.write("### 👨 Male")
    
        st.progress(male_prob/100)
    
        st.markdown(f"""
        <h2 style='color:#60a5fa'>
        {male_prob:.2f}%
        </h2>
        """, unsafe_allow_html=True)
    
    with female_col:
    
        st.write("### 👩 Female")
    
        st.progress(female_prob/100)
    
        st.markdown(f"""
        <h2 style='color:#ec4899'>
        {female_prob:.2f}%
        </h2>
        """, unsafe_allow_html=True)
    
    st.write("")
    st.write("")
    
    # ============================================
    # AI STATUS
    # ============================================
    
    if confidence_percent >= 95:
    
        st.success("🟢 AI is extremely confident about this prediction.")
    
    elif confidence_percent >= 85:
    
        st.info("🔵 AI has high confidence in this prediction.")
    
    elif confidence_percent >= 70:
    
        st.warning("🟡 Prediction is moderately confident.")
    
    else:
    
        st.error("🔴 Confidence is low. Try another eye image.")
    # ============================================
    # AI INSIGHTS
    # ============================================
    
    st.write("")
    st.markdown("---")
    st.write("")
    
    st.markdown("""
    <h2 style='text-align:center;color:white;'>
    🧠 AI Insights
    </h2>
    """, unsafe_allow_html=True)
    
    left, right = st.columns(2)
    
    # ---------------------------------------------------
    # WHY THIS PREDICTION
    # ---------------------------------------------------
    
    with left:
    
        st.markdown("""
        <div class="glass">
        <h2>🔍 Why this prediction?</h2>
        </div>
        """, unsafe_allow_html=True)
    
        if confidence_percent >= 95:
    
            st.success("""
    The CNN found highly discriminative eye features.
    
    ✔ Strong edge patterns
    
    ✔ Clear iris & eyelid structure
    
    ✔ High feature confidence
    """)
    
        elif confidence_percent >= 85:
    
            st.info("""
    The CNN confidently identified the eye.
    
    Minor uncertainty exists but prediction is reliable.
    """)
    
        else:
    
            st.warning("""
    Prediction confidence is lower.
    
    Try uploading:
    
    • Better lighting
    
    • Higher resolution
    
    • Clearly visible eye
    """)
    
    # ---------------------------------------------------
    # MODEL HIGHLIGHTS
    # ---------------------------------------------------
    
    with right:
    
        st.markdown("""
        <div class="glass">
        <h2>🏆 Model Highlights</h2>
        </div>
        """, unsafe_allow_html=True)
    
        st.markdown("""
    ✅ Convolutional Neural Network
    
    ✅ Binary Image Classification
    
    ✅ TensorFlow / Keras
    
    ✅ Fast Inference
    
    ✅ Real-time Prediction
    
    ✅ Streamlit Deployment
    """)
    
    # ============================================
    # PERFORMANCE
    # ============================================
    
    st.write("")
    st.write("")
    
    st.markdown("""
    <div class="glass">
    <h2 align="center">
    ⚡ Performance Summary
    </h2>
    </div>
    """, unsafe_allow_html=True)
    
    perf1, perf2, perf3 = st.columns(3)
    
    with perf1:
    
        st.metric(
            "Prediction Time",
            f"{inference_time:.3f}s"
        )
    
    with perf2:
    
        st.metric(
            "Confidence",
            f"{confidence_percent:.2f}%"
        )
    
    with perf3:
    
        st.metric(
            "Model",
            "CNN"
        )
    
    # ============================================
    # USER TIPS
    # ============================================
    
    st.write("")
    st.write("")
    
    st.markdown("""
    <div class="glass">
    <h2>💡 Tips for Better Prediction</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("""
    ✔ Upload only one eye.
    
    ✔ Avoid blurry images.
    
    ✔ Ensure good lighting.
    
    ✔ Crop unnecessary background.
    
    ✔ Use JPG or PNG images.
    """)
    
    # ============================================
    # TECHNOLOGY BADGES
    # ============================================
    
    st.write("")
    st.write("")
    
    st.markdown("""
    <h2 style='text-align:center;color:white;'>
    🚀 Built With
    </h2>
    """, unsafe_allow_html=True)
    
    b1,b2,b3,b4 = st.columns(4)
    
    with b1:
        st.success("🐍 Python")
    
    with b2:
        st.success("🧠 TensorFlow")
    
    with b3:
        st.success("⚡ Streamlit")
    
    with b4:
        st.success("📷 CNN")
    
    # ============================================
    # FOOTER
    # ============================================
    
    st.write("")
    st.markdown("---")
    
    st.markdown("""
    <div style='
    text-align:center;
    padding:30px;
    background:rgba(255,255,255,.06);
    border-radius:20px;
    '>
    
    <h2>👁️ Male vs Female Eye Classifier</h2>
    
    <p>
    Deep Learning • CNN • TensorFlow • Streamlit
    </p>
    
    <br>
    
    Developed by
    
    <h3 style='color:#60a5fa;'>
    Anushka Singh
    </h3>
    
    <p>
    Built by ANUSHKA ⭐ using Artificial Intelligence
    </p>
    
    </div>
    """, unsafe_allow_html=True)    
        
        
    
