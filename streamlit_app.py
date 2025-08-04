import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time

# 🎨 Page config with wide layout
st.set_page_config(
    page_title="🧠 Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🎨 Custom CSS for beautiful styling
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
    }
    
    /* Header styling */
    .main-header {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .main-header h1 {
        color: #2c3e50;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .main-header p {
        color: #7f8c8d;
        font-size: 1.2rem;
        line-height: 1.6;
    }
    
    /* Card styling */
    .prediction-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border-left: 5px solid #3498db;
    }
    
    .upload-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        text-align: center;
        border: 2px dashed #3498db;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #3498db, #2980b9);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(52, 152, 219, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(52, 152, 219, 0.4);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #3498db, #2ecc71);
        border-radius: 10px;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
    }
    
    /* Success message styling */
    .success-message {
        background: linear-gradient(45deg, #2ecc71, #27ae60);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(46, 204, 113, 0.3);
        margin: 1rem 0;
    }
    
    /* Info message styling */
    .info-message {
        background: linear-gradient(45deg, #3498db, #2980b9);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.2rem;
        box-shadow: 0 4px 15px rgba(52, 152, 219, 0.3);
        margin: 1rem 0;
    }
    
    /* Footer styling */
    .footer {
        background: rgba(255, 255, 255, 0.9);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin-top: 2rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* Animation for loading */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
</style>
""", unsafe_allow_html=True)

# 🧬 Load model (cached for speed)
@st.cache_resource
def load_trained_model():
    try:
        return load_model('brain_tumor_inceptionv3.keras')
    except:
        return load_model('brain_tumor_inceptionv3.keras')

# 🏷 Class names with descriptions
class_info = {
    'glioma': {
        'name': 'Glioma',
        'description': 'A type of tumor that occurs in the brain and spinal cord',
        'color': '#e74c3c',
        'icon': '🔴'
    },
    'meningioma': {
        'name': 'Meningioma',
        'description': 'A tumor that forms on membranes covering the brain and spinal cord',
        'color': '#f39c12',
        'icon': '🟡'
    },
    'notumor': {
        'name': 'No Tumor',
        'description': 'Normal brain tissue with no tumor detected',
        'color': '#27ae60',
        'icon': '🟢'
    },
    'pituitary': {
        'name': 'Pituitary',
        'description': 'A tumor in the pituitary gland at the base of the brain',
        'color': '#9b59b6',
        'icon': '🟣'
    }
}

class_names = list(class_info.keys())

# 📊 Enhanced predict function with progress
def predict_image(img):
    img = img.resize((224, 224))  # match model input size
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Add loading animation
    with st.spinner('🔬 Analyzing MRI image...'):
        time.sleep(1)  # Simulate processing time
        preds = model.predict(img_array, verbose=0)[0]
    
    return preds

# 📈 Create beautiful charts
def create_prediction_chart(preds):
    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=[class_info[cls]['name'] for cls in class_names],
            y=preds * 100,
            marker_color=[class_info[cls]['color'] for cls in class_names],
            text=[f'{p*100:.1f}%' for p in preds],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Probability: %{y:.1f}%<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title={
            'text': '📊 Prediction Confidence by Class',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#2c3e50'}
        },
        xaxis_title="Tumor Type",
        yaxis_title="Confidence (%)",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#2c3e50'),
        height=400,
        showlegend=False
    )
    
    return fig

def create_radar_chart(preds):
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=preds * 100,
        theta=[class_info[cls]['name'] for cls in class_names],
        fill='toself',
        name='Prediction',
        line_color='#3498db',
        fillcolor='rgba(52, 152, 219, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=False,
        title={
            'text': '🎯 Radar Chart of Predictions',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#2c3e50'}
        },
        height=400
    )
    
    return fig

# 🎯 Main App
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🧠 Brain Tumor MRI Classifier</h1>
        <p>Advanced AI-powered analysis of brain MRI images using deep learning technology</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.95); padding: 1.5rem; border-radius: 15px; margin-bottom: 1rem;">
            <h3 style="color: #2c3e50; text-align: center;">📤 Upload MRI Image</h3>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a brain MRI image",
            type=['jpg', 'png', 'jpeg'],
            help="Upload a clear MRI image for analysis"
        )
        
        st.markdown("---")
        
        # Model info
        st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.95); padding: 1rem; border-radius: 10px;">
            <h4 style="color: #2c3e50;">🤖 Model Information</h4>
            <p style="color: #7f8c8d; font-size: 0.9rem;">
                <strong>Architecture:</strong> InceptionV3<br>
                <strong>Input Size:</strong> 224x224 pixels<br>
                <strong>Classes:</strong> 4 tumor types
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if uploaded_file:
            # Display uploaded image
            st.markdown("""
            <div class="prediction-card">
                <h3 style="color: #2c3e50; text-align: center;">📷 Uploaded MRI Image</h3>
            </div>
            """, unsafe_allow_html=True)
            
            img = Image.open(uploaded_file).convert('RGB')
            st.image(img, use_column_width=True, caption="MRI Scan for Analysis")
            
            # Prediction
            try:
                model = load_trained_model()
                preds = predict_image(img)
                top_idx = np.argmax(preds)
                top_class = class_names[top_idx]
                confidence = preds[top_idx]
                
                # Success message
                st.markdown(f"""
                <div class="success-message">
                    {class_info[top_class]['icon']} <strong>Prediction: {class_info[top_class]['name']}</strong><br>
                    Confidence: {confidence*100:.1f}%
                </div>
                """, unsafe_allow_html=True)
                
                # Class description
                st.markdown(f"""
                <div class="prediction-card">
                    <h4 style="color: #2c3e50;">ℹ️ About {class_info[top_class]['name']}</h4>
                    <p style="color: #7f8c8d;">{class_info[top_class]['description']}</p>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
        else:
            # Upload prompt
            st.markdown("""
            <div class="upload-card">
                <h3 style="color: #2c3e50;">📤 Upload Your MRI Image</h3>
                <p style="color: #7f8c8d;">Use the sidebar to upload a brain MRI image for analysis</p>
                <div style="font-size: 4rem; margin: 2rem 0;">🧠</div>
                <p style="color: #7f8c8d; font-size: 0.9rem;">
                    Supported formats: JPG, PNG, JPEG<br>
                    Recommended: Clear, high-resolution images
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if uploaded_file and 'preds' in locals():
            # Charts
            st.markdown("""
            <div class="prediction-card">
                <h3 style="color: #2c3e50; text-align: center;">📊 Analysis Results</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Bar chart
            chart1 = create_prediction_chart(preds)
            st.plotly_chart(chart1, use_container_width=True)
            
            # Radar chart
            chart2 = create_radar_chart(preds)
            st.plotly_chart(chart2, use_container_width=True)
            
            # Detailed probabilities
            st.markdown("""
            <div class="prediction-card">
                <h4 style="color: #2c3e50;">📋 Detailed Probabilities</h4>
            </div>
            """, unsafe_allow_html=True)
            
            for i, (cls, prob) in enumerate(zip(class_names, preds)):
                color = class_info[cls]['color']
                icon = class_info[cls]['icon']
                name = class_info[cls]['name']
                
                st.markdown(f"""
                <div style="background: rgba(255, 255, 255, 0.8); padding: 1rem; border-radius: 10px; margin: 0.5rem 0; border-left: 4px solid {color};">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 1.2rem;">{icon} {name}</span>
                        <span style="font-weight: bold; color: {color};">{prob*100:.1f}%</span>
                    </div>
                    <div style="background: #ecf0f1; height: 8px; border-radius: 4px; margin-top: 0.5rem;">
                        <div style="background: {color}; height: 8px; border-radius: 4px; width: {prob*100}%; transition: width 0.5s ease;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            # Info about the app
            st.markdown("""
            <div class="info-message">
                <h3>🔬 How It Works</h3>
                <p>Our AI model analyzes MRI images to detect and classify brain tumors with high accuracy.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Features
            features = [
                ("🤖", "Deep Learning", "Powered by InceptionV3 architecture"),
                ("🎯", "High Accuracy", "Trained on extensive medical datasets"),
                ("⚡", "Fast Analysis", "Real-time prediction results"),
                ("🛡️", "Secure", "Your data stays private and local")
            ]
            
            for icon, title, desc in features:
                st.markdown(f"""
                <div style="background: rgba(255, 255, 255, 0.9); padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
                    <h4 style="color: #2c3e50; margin: 0;">{title}</h4>
                    <p style="color: #7f8c8d; margin: 0; font-size: 0.9rem;">{desc}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <p style="color: #7f8c8d; margin: 0;">
            Made with ❤️ by Akarsh Yadav • Powered by Streamlit & TensorFlow<br>
            <small>For educational and research purposes only. Always consult healthcare professionals for medical decisions.</small>
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

