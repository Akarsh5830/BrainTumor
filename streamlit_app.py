import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# 🎨 Page config
st.set_page_config(
    page_title="🧠 Brain Tumor MRI Classifier",
    page_icon="🧪",
    layout="centered",
)

# ✏️ Title and description
st.title("🧠 Brain Tumor MRI Classifier")
st.markdown("""
This app uses a deep learning model (InceptionV3) to predict brain tumor type from MRI images.
Upload an image to get prediction!
""")

# 📥 Sidebar
st.sidebar.header("Upload MRI Image")
uploaded_file = st.sidebar.file_uploader("Choose a brain MRI image", type=['jpg', 'png', 'jpeg'])

# 🧬 Load model (cached for speed)
@st.cache_resource
def load_trained_model():
    return load_model('brain_tumor_inceptionv3.keras')

model = load_trained_model()

# 🏷 Class names
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# 📊 Predict function
def predict_image(img):
    img = img.resize((224, 224))  # match model input size
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)[0]
    return preds

# 📷 Main area
if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded MRI Image', use_column_width=True)

    preds = predict_image(img)
    top_idx = np.argmax(preds)
    top_class = class_names[top_idx]
    confidence = preds[top_idx]

    st.success(f"**Predicted:** {top_class.capitalize()} (Confidence: {confidence*100:.2f}%)")

    # 🔍 Show all class probabilities
    st.subheader("Class probabilities:")
    for cls, prob in zip(class_names, preds):
        st.write(f"• **{cls.capitalize()}**: {prob*100:.2f}%")

else:
    st.info("⬆️ Please upload an MRI image to get prediction.")

# 📌 Footer
st.markdown("---")
st.caption("Made with ❤️ by [Your Name or Team] • Powered by Streamlit & TensorFlow")

