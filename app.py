import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
from predict import predict_image
import datetime

st.set_page_config(
    page_title="Early Alzheimer's Disease Prediction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>

.main{
background:#f5f7fb;
}

.block-container{
padding-top:1rem;
padding-bottom:2rem;
}

.title{
font-size:40px;
font-weight:800;
color:#0F172A;
}

.subtitle{
font-size:18px;
color:#64748B;
}

.card{
background:white;
padding:20px;
border-radius:15px;
box-shadow:0px 3px 15px rgba(0,0,0,0.08);
}

.small{
font-size:15px;
color:gray;
}

</style>
""",unsafe_allow_html=True)

st.sidebar.image(
"https://img.icons8.com/color/96/brain.png",
width=90
)

st.sidebar.title("Navigation")

page=st.sidebar.radio(
"",
[
"Prediction",
"About Project"
]
)

if page=="About Project":

    st.title("About")

    st.write("""
### Early Alzheimer's Disease Prediction

This system predicts the stage of Alzheimer's disease from MRI scans.

### Deep Learning Model

MobileNetV2

### MRI Classes

• Non Demented

• Very Mild Demented

• Mild Demented

• Moderate Demented

### Technology

Python

TensorFlow

Streamlit

Plotly

OpenCV
""")

    st.stop()

st.markdown(
"<div class='title'>🧠 Early Alzheimer's Disease Prediction System</div>",
unsafe_allow_html=True
)

st.markdown(
"<div class='subtitle'>Artificial Intelligence Based MRI Classification</div>",
unsafe_allow_html=True
)

st.write("")

left,right=st.columns([1,1])

with left:

    st.markdown("## 👤 Patient Information")

    patient=st.text_input("Patient Name")

    age=st.number_input(
        "Age",
        1,
        120,
        40
    )

    gender=st.selectbox(
        "Gender",
        [
            "Male",
            "Female",
            "Other"
        ]
    )

    uploaded=st.file_uploader(
        "Upload MRI Image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

with right:

    st.markdown("## 📅 Scan Information")

    st.write("Date :",datetime.date.today())

    st.write("Hospital : AI Medical Center")

    st.write("Model : MobileNetV2")

    st.write("Classes : 4")

if uploaded is None:

    st.info("Upload an MRI image to begin prediction.")

    st.stop()

image=Image.open(uploaded)

st.divider()

col1,col2=st.columns([1,1])

with col1:

    st.subheader("🖼 Uploaded MRI")

    st.image(
        image,
        use_container_width=True
    )

result=predict_image(image)

label=result["label"]

confidence=result["confidence"]

prob=result["probabilities"]

with col2:

    st.subheader("🧠 Prediction Result")

    if label=="Non Demented":

        st.success(label)

    elif label=="Very Mild Demented":

        st.warning(label)

    elif label=="Mild Demented":

        st.error(label)

    else:

        st.error(label)

    st.metric(
        "Prediction Confidence",
        f"{confidence:.2f}%"
    )

    st.progress(int(confidence))

st.divider()

st.subheader("📊 Probability Distribution")

df=pd.DataFrame({

    "Stage":list(prob.keys()),

    "Probability":list(prob.values())

})

fig=px.bar(

    df,

    x="Stage",

    y="Probability",

    text="Probability",

    color="Probability",

    color_continuous_scale="Blues"

)

fig.update_layout(

    height=500,

    title="Prediction Probability"

)

st.plotly_chart(

    fig,

    use_container_width=True

)

st.divider()

st.subheader("📋 Patient Summary")

summary=pd.DataFrame({

    "Field":[

        "Patient",

        "Age",

        "Gender",

        "Prediction",

        "Confidence"

    ],

    "Value":[

        patient,

        age,

        gender,

        label,

        f"{confidence:.2f}%"

    ]

})

st.dataframe(

    summary,

    use_container_width=True,

    hide_index=True

)

st.divider()

st.subheader("🩺 Clinical Recommendation")

if label=="Non Demented":

    st.success("""

No significant signs of dementia were detected.

Continue regular health check-ups and maintain a healthy lifestyle.

""")

elif label=="Very Mild Demented":

    st.warning("""

Very early Alzheimer's-related changes may be present.

A neurological consultation and follow-up MRI are recommended.

""")

elif label=="Mild Demented":

    st.error("""

Signs are consistent with Mild Dementia.

Please consult a neurologist for cognitive assessment and treatment planning.

""")

else:

    st.error("""

Moderate Dementia detected.

Immediate specialist consultation is strongly recommended.

""")

st.divider()

st.subheader("ℹ️ Model Information")

c1,c2,c3,c4=st.columns(4)

c1.metric("Model","MobileNetV2")

c2.metric("Classes","4")

c3.metric("Image Size","224×224")

c4.metric("Framework","TensorFlow")