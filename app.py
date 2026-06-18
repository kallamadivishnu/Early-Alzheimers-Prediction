import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
from predict import predict_image
from datetime import datetime

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Early Alzheimer's Disease Prediction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">

<style>

html, body, [class*="css"]{
    font-family:'Poppins',sans-serif;
}

.stApp{
    background:#F4F9FF;
}

section[data-testid="stSidebar"]{
    background:#EAF4FF;
}

.main-title{
    font-size:40px;
    font-weight:700;
    color:#0B5ED7;
    text-align:center;
    margin-top:10px;
}

.sub-title{
    font-size:18px;
    color:#666666;
    text-align:center;
    margin-bottom:30px;
}

.card{
    background:white;
    padding:22px;
    border-radius:18px;
    box-shadow:0px 8px 20px rgba(0,0,0,0.08);
}

.metric-card{
    background:white;
    padding:18px;
    border-radius:15px;
    text-align:center;
    box-shadow:0px 5px 15px rgba(0,0,0,0.08);
}

.metric-number{
    font-size:32px;
    color:#0B5ED7;
    font-weight:bold;
}

.metric-text{
    color:#666;
    font-size:15px;
}

.stButton>button{
    width:100%;
    height:50px;
    border-radius:10px;
    background:#0B5ED7;
    color:white;
    font-size:17px;
    font-weight:600;
    border:none;
}

.stButton>button:hover{
    background:#0849A5;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# SESSION STATE
# -----------------------------
if "results" not in st.session_state:
    st.session_state.results = []

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/2966/2966486.png",
    width=90
)

st.sidebar.title("Hospital Dashboard")

page = st.sidebar.radio(
    "Navigation",
    [
        "Prediction Dashboard",
        "About"
    ]
)

# -----------------------------
# ABOUT PAGE
# -----------------------------
if page == "About":

    st.title("About")

    st.write("""
### Early Alzheimer's Disease Prediction

This application predicts Alzheimer's disease stage from MRI brain images using Artificial Intelligence.

### Features

- Multiple MRI Image Upload
- AI Prediction
- Interactive Dashboard
- CSV Download
- Charts
- Modern Hospital Interface

**Note:** This application is developed for educational and research purposes.
""")

    st.stop()

# -----------------------------
# HEADER
# -----------------------------
st.markdown(
    "<div class='main-title'>🧠 Early Alzheimer's Disease Prediction</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='sub-title'>AI Powered MRI Analysis Dashboard</div>",
    unsafe_allow_html=True
)

# -----------------------------
# DASHBOARD CARDS
# -----------------------------
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class='metric-card'>
    <div class='metric-number'>AI</div>
    <div class='metric-text'>Prediction Model</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class='metric-card'>
    <div class='metric-number'>MRI</div>
    <div class='metric-text'>Brain Scan Analysis</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class='metric-card'>
    <div class='metric-number'>24×7</div>
    <div class='metric-text'>Hospital Support</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# -----------------------------
# IMAGE UPLOAD
# -----------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)

st.subheader("Upload MRI Images")

uploaded_files = st.file_uploader(
    "Choose one or more MRI images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

predict_button = st.button("Predict Images")

st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# PREDICTION
# -----------------------------
if predict_button:

    if not uploaded_files:
        st.warning("Please upload at least one MRI image.")
        st.stop()

    progress = st.progress(0)

    total = len(uploaded_files)

    for i, uploaded_file in enumerate(uploaded_files):

        try:

            # Open image directly
            image = Image.open(uploaded_file)

            col1, col2 = st.columns([1, 2])

            with col1:
                st.image(
                    image,
                    caption=uploaded_file.name,
                    use_container_width=True
                )

            # Predict directly (NO temporary file)
            prediction = predict_image(image)

            with col2:

                st.success("Prediction Completed")

                st.markdown(f"""
                ### Result

                **Image:** {uploaded_file.name}

                **Prediction:** {prediction}
                """)

            # Save result
            st.session_state.results.append({
                "Image": uploaded_file.name,
                "Prediction": prediction,
                "Date": datetime.now().strftime("%d-%m-%Y %H:%M:%S")
            })

        except Exception as e:

            st.error(f"Error processing {uploaded_file.name}")

            st.exception(e)

        progress.progress((i + 1) / total)

    progress.empty()

    st.success("Prediction completed successfully.")

st.write("")

# -----------------------------
# RESULTS SECTION
# -----------------------------
if len(st.session_state.results) > 0:

    st.markdown("---")

    st.header("Prediction Results")

    results_df = pd.DataFrame(st.session_state.results)

    # -----------------------------
    # SUMMARY CARDS
    # -----------------------------
    total_images = len(results_df)

    non_demented = len(results_df[results_df["Prediction"] == "Non Demented"])
    very_mild = len(results_df[results_df["Prediction"] == "Very Mild Demented"])
    mild = len(results_df[results_df["Prediction"] == "Mild Demented"])
    moderate = len(results_df[results_df["Prediction"] == "Moderate Demented"])

    a, b, c, d = st.columns(4)

    with a:
        st.metric("Total MRI", total_images)

    with b:
        st.metric("Non Demented", non_demented)

    with c:
        st.metric("Very Mild", very_mild)

    with d:
        st.metric("Demented", mild + moderate)

    st.write("")

    # -----------------------------
    # RESULTS TABLE
    # -----------------------------
    st.subheader("Prediction Table")

    st.dataframe(
        results_df,
        use_container_width=True,
        hide_index=True
    )

    # -----------------------------
    # DOWNLOAD CSV
    # -----------------------------
    csv = results_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Results CSV",
        data=csv,
        file_name="alzheimers_predictions.csv",
        mime="text/csv"
    )

    st.write("")

    # -----------------------------
    # SEARCH RESULT
    # -----------------------------
    search = st.text_input("Search Image Name")

    if search:

        filtered = results_df[
            results_df["Image"].str.contains(
                search,
                case=False,
                na=False
            )
        ]

        st.dataframe(
            filtered,
            use_container_width=True,
            hide_index=True
        )

    st.write("")

    # -----------------------------
    # RECENT PREDICTIONS
    # -----------------------------
    st.subheader("Recent Predictions")

    st.dataframe(
        results_df.tail(10),
        use_container_width=True,
        hide_index=True
    )

    st.write("")

# -----------------------------
# ANALYTICS DASHBOARD
# -----------------------------
if len(st.session_state.results) > 0:

    st.markdown("---")
    st.header("Analytics Dashboard")

    chart_df = pd.DataFrame(st.session_state.results)

    prediction_counts = (
        chart_df["Prediction"]
        .value_counts()
        .reset_index()
    )

    prediction_counts.columns = [
        "Prediction",
        "Count"
    ]

    col1, col2 = st.columns(2)

    # -----------------------------
    # BAR CHART
    # -----------------------------
    with col1:

        st.subheader("Prediction Distribution")

        fig_bar = px.bar(
            prediction_counts,
            x="Prediction",
            y="Count",
            color="Prediction",
            text="Count",
            template="plotly_white"
        )

        fig_bar.update_layout(
            height=450,
            title="MRI Prediction Count",
            xaxis_title="Prediction",
            yaxis_title="Number of MRI Images",
            showlegend=False
        )

        fig_bar.update_traces(
            textposition="outside"
        )

        st.plotly_chart(
            fig_bar,
            use_container_width=True
        )

    # -----------------------------
    # PIE CHART
    # -----------------------------
    with col2:

        st.subheader("Prediction Percentage")

        fig_pie = px.pie(
            prediction_counts,
            names="Prediction",
            values="Count",
            hole=0.45,
            template="plotly_white"
        )

        fig_pie.update_layout(
            height=450
        )

        st.plotly_chart(
            fig_pie,
            use_container_width=True
        )

    st.write("")

    # -----------------------------
    # HOSPITAL ANALYTICS
    # -----------------------------
    st.subheader("Hospital Analytics")

    left, right = st.columns(2)

    with left:

        st.info(f"""
### Total MRI Scans

**{len(chart_df)}**

MRI scans have been processed successfully.
""")

    with right:

        most_common = chart_df["Prediction"].mode()[0]

        st.success(f"""
### Most Common Prediction

**{most_common}**
""")

    st.write("")

    # -----------------------------
    # PREDICTION FREQUENCY
    # -----------------------------
    st.subheader("Prediction Frequency")

    frequency_df = prediction_counts.sort_values(
        by="Count",
        ascending=False
    )

    st.dataframe(
        frequency_df,
        use_container_width=True,
        hide_index=True
    )

    st.write("")

    # -----------------------------
    # CLEAR HISTORY
    # -----------------------------
    if st.button("🗑️ Clear Prediction History"):

        st.session_state.results = []

        st.success("Prediction history cleared.")

        st.rerun()

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")

st.markdown("""
<div style="
background:#0B5ED7;
padding:25px;
border-radius:15px;
color:white;
text-align:center;
">

<h2>🏥 Early Alzheimer's Disease Prediction System</h2>

<p style="font-size:17px;">
AI-powered MRI analysis dashboard for early Alzheimer's disease prediction.
</p>

</div>
""", unsafe_allow_html=True)

st.write("")

# =====================================================
# DISCLAIMER
# =====================================================
with st.expander("⚠ Medical Disclaimer"):

    st.write("""
This application is intended **only for educational and research purposes**.

The AI model provides prediction results based on MRI images and should **not**
be considered a final medical diagnosis.

Always consult a qualified neurologist or healthcare professional for
clinical diagnosis and treatment decisions.
""")

# =====================================================
# PROJECT INFORMATION
# =====================================================
with st.expander("ℹ Project Information"):

    st.markdown("""
### Project Name

Early Alzheimer's Disease Prediction using Deep Learning

### Technologies Used

- Python
- TensorFlow / Keras
- Streamlit
- Plotly
- NumPy
- Pandas
- Pillow

### Features

✅ Multiple MRI Upload

✅ AI Prediction

✅ Interactive Dashboard

✅ Result Table

✅ CSV Download

✅ Plotly Charts

✅ Hospital Style Interface

✅ About Page

### Model Classes

- Non Demented
- Very Mild Demented
- Mild Demented
- Moderate Demented
""")

# =====================================================
# VERSION
# =====================================================
col1, col2, col3 = st.columns(3)

with col1:
    st.info("Version 1.0")

with col2:
    st.success("AI Model Loaded")

with col3:
    st.info(datetime.now().strftime("%d-%m-%Y"))

# =====================================================
# COPYRIGHT
# =====================================================
st.markdown(
    """
    <div style="text-align:center;
                color:gray;
                margin-top:25px;
                margin-bottom:10px;">

    © 2026 Early Alzheimer's Disease Prediction System

    <br>

    Developed using Streamlit & TensorFlow

    </div>
    """,
    unsafe_allow_html=True
)