# app.py
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Alzheimer's Detection",
    page_icon="🧠",
    layout="centered"
)

# ── Load model (cached so it loads only once) ──────────────────
@st.cache_resource
def load_alzheimer_model():
    model = load_model('saved_model/best_model.h5')
    return model

model = load_alzheimer_model()

# ── Class info ─────────────────────────────────────────────────
CLASS_NAMES = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

CLASS_INFO = {
    'NonDemented': {
        'color': '#2ecc71',
        'emoji': '✅',
        'description': 'No signs of dementia detected. Brain structure appears normal.',
        'advice': 'Continue regular health checkups and maintain a healthy lifestyle.'
    },
    'VeryMildDemented': {
        'color': '#f39c12',
        'emoji': '⚠️',
        'description': 'Very mild cognitive decline detected. Early stage indicators present.',
        'advice': 'Consult a neurologist for further evaluation. Early intervention is key.'
    },
    'MildDemented': {
        'color': '#e67e22',
        'emoji': '🔶',
        'description': 'Mild dementia detected. Noticeable memory and cognitive changes.',
        'advice': 'Immediate medical consultation recommended. Treatment can slow progression.'
    },
    'ModerateDemented': {
        'color': '#e74c3c',
        'emoji': '🚨',
        'description': 'Moderate dementia detected. Significant cognitive impairment present.',
        'advice': 'Urgent medical attention required. Specialist care and support needed.'
    }
}

IMG_SIZE = (128, 128)

# ── UI ─────────────────────────────────────────────────────────
st.title("🧠 Early Detection of Alzheimer's Disease")
st.markdown("#### AI-powered MRI Brain Scan Analysis")
st.markdown("---")

st.markdown("""
This tool uses a deep learning model (MobileNetV2) trained on the OASIS dataset
to analyze MRI brain scans and detect early signs of Alzheimer's disease.
""")

# ── Sidebar info ───────────────────────────────────────────────
with st.sidebar:
    st.header("📋 About")
    st.markdown("""
    **Model:** MobileNetV2 (Transfer Learning)
    
    **Dataset:** OASIS Brain MRI
    
    **Classes:**
    - ✅ Non Demented
    - ⚠️ Very Mild Demented
    - 🔶 Mild Demented
    - 🚨 Moderate Demented
    
    **Accuracy:** ~75%
    
    ---
    ⚕️ *This tool is for educational 
    purposes only. Always consult 
    a medical professional.*
    """)

# ── File uploader ──────────────────────────────────────────────
st.markdown("### 📤 Upload MRI Brain Scan")
uploaded_file = st.file_uploader(
    "Upload a brain MRI image (JPG, PNG, JPEG)",
    type=['jpg', 'jpeg', 'png']
)

if uploaded_file is not None:
    # Show uploaded image
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Uploaded MRI Scan:**")
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, use_container_width=True)

    # Preprocess image
    img_resized = image.resize(IMG_SIZE)
    img_array  = np.array(img_resized) / 255.0
    img_array  = np.expand_dims(img_array, axis=0)

    # Predict
    with st.spinner('🔄 Analyzing MRI scan...'):
        predictions = model.predict(img_array)
        predicted_idx   = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence      = predictions[0][predicted_idx] * 100

    info = CLASS_INFO[predicted_class]

    with col2:
        st.markdown("**Analysis Result:**")
        st.markdown(f"""
        <div style='background-color: {info["color"]}22; 
                    border-left: 5px solid {info["color"]};
                    padding: 20px; border-radius: 8px; margin-top: 10px;'>
            <h3 style='color: {info["color"]}; margin:0;'>
                {info["emoji"]} {predicted_class}
            </h3>
            <h4 style='margin: 10px 0 5px;'>Confidence: {confidence:.1f}%</h4>
        </div>
        """, unsafe_allow_html=True)

    # Description and advice
    st.markdown("---")
    st.markdown("### 📊 Detailed Analysis")

    col3, col4 = st.columns(2)
    with col3:
        st.info(f"**Diagnosis:** {info['description']}")
    with col4:
        st.warning(f"**Recommendation:** {info['advice']}")

    # Probability bar chart for all classes
    st.markdown("### 📈 Prediction Probabilities")
    prob_dict = {CLASS_NAMES[i]: float(predictions[0][i]) * 100 for i in range(4)}

    for cls, prob in sorted(prob_dict.items(), key=lambda x: x[1], reverse=True):
        color = CLASS_INFO[cls]['color']
        emoji = CLASS_INFO[cls]['emoji']
        st.markdown(f"**{emoji} {cls}**")
        st.progress(prob / 100)
        st.markdown(f"`{prob:.1f}%`")

    # Disclaimer
    st.markdown("---")
    st.error("⚕️ **Medical Disclaimer:** This prediction is for educational purposes only and should NOT be used as a medical diagnosis. Always consult a qualified neurologist or medical professional.")

else:
    # Show sample instructions when no image uploaded
    st.info("👆 Upload an MRI brain scan image above to get started.")

    st.markdown("### 🔍 What to expect:")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("✅ **Non Demented**\nNo signs of dementia")
    with col2:
        st.markdown("⚠️ **Very Mild**\nEarly indicators")
    with col3:
        st.markdown("🔶 **Mild**\nNoticeable changes")
    with col4:
        st.markdown("🚨 **Moderate**\nSignificant impairment")