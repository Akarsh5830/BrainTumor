import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import json

# 🎨 Page config
st.set_page_config(
    page_title="🧠 Brain Tumor MRI Classifier",
    page_icon="🧪",
    layout="centered",
)

# ✏️ Title & description
st.title("🧠 Brain Tumor MRI Classifier")
st.markdown("""
Upload a brain MRI image and this app will predict the tumor type using an InceptionV3 deep learning model.
""")

# 📥 Sidebar upload
st.sidebar.header("📷 Upload MRI Image")
uploaded_file = st.sidebar.file_uploader(
    "Choose a brain MRI image", type=['jpg', 'png', 'jpeg'])

# 🧬 Load trained model (cached)
@st.cache_resource
def load_trained_model():
    return load_model('brain_tumor_inceptionv3.keras')

model = load_trained_model()

# 🏷 Load class names dynamically
with open('class_names.json') as f:
    class_names = json.load(f)

# 📊 Predict function
def predict_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array, verbose=0)[0]
    return preds

# 📷 Main area
if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='🩺 Uploaded MRI Image', use_container_width=True)

    with st.spinner('🔍 Analyzing image...'):
        preds = predict_image(img)

    top_idx = int(np.argmax(preds))
    top_class = class_names[top_idx]
    confidence = preds[top_idx]

    st.success(f"✅ **Predicted:** `{top_class.capitalize()}` (Confidence: {confidence*100:.2f}%)")

    # 📊 Detailed probabilities
    st.subheader("📊 Class probabilities:")
    prob_df = {cls.capitalize(): f"{prob*100:.2f}%" for cls, prob in zip(class_names, preds)}
    for cls, prob in prob_df.items():
        st.write(f"• **{cls}**: {prob}")

else:
    st.info("⬆️ Please upload an MRI image to get prediction.")

# 📌 Footer
st.markdown("---")
st.caption("Made with ❤️ by [Your Name or Team] • Powered by Streamlit & TensorFlow")
