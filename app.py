import streamlit as st
import pickle
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI styling
st.markdown("""
    <style>
    /* Global styles */
    .main {
        padding: 2rem 3rem;
    }
    
    /* Card Container Style */
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        border: 1px solid #e9ecef;
        margin-bottom: 20px;
    }
    
    /* Prediction Box Style */
    .prediction-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 10px 20px rgba(118, 75, 162, 0.3);
    }
    
    .prediction-title {
        font-size: 1.2rem;
        font-weight: 500;
        margin-bottom: 8px;
        opacity: 0.9;
    }
    
    .prediction-value {
        font-size: 3.2rem;
        font-weight: 700;
        margin: 0;
    }
    
    /* Header styling */
    .main-header {
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #64748B;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Model Loading
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    """Loads the pickled machine learning model."""
    try:
        with open("model.pkl", "rb") as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Error: `model.pkl` file not found. Please ensure it is in the root directory.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()


# -----------------------------------------------------------------------------
# Sidebar Content
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/illustrations/100/learning.png", width=100)
    st.title("Settings & Info")
    st.write("Adjust parameters to predict the final output score.")
    
    st.divider()
    
    st.markdown("### Model Details")
    if model is not None:
        st.info(
            f"**Algorithm:** {type(model).__name__}\n\n"
            f"**Algorithm Type:** {model.algorithm.title()}\n\n"
            f"**Neighbors (k):** {model.n_neighbors}"
        )
    else:
        st.warning("Model not loaded.")

    st.divider()
    st.caption("🚀 Built with Streamlit & Scikit-Learn")


# -----------------------------------------------------------------------------
# Main Application Layout
# -----------------------------------------------------------------------------
st.markdown("<h1 class='main-header'>🎓 Student Score Estimator</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Predict student performance based on enrolled courses and study hours using KNN Regression.</p>", unsafe_allow_html=True)

if model is not None:
    # Creating a two-column layout: Inputs on left, Prediction display on right
    col1, col2 = st.columns([1.2, 1], gap="large")

    with col1:
        st.markdown("### 🎛️ Input Features")
        
        with st.container():
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            
            # Feature 1: Number of Courses
            number_courses = st.number_input(
                label="Number of Courses",
                min_value=1,
                max_value=20,
                value=5,
                step=1,
                help="Enter the total number of courses the student is currently enrolled in."
            )
            
            st.write("") # Spacing
            
            # Feature 2: Time spent studying
            time_study = st.slider(
                label="Study Time (Hours / Day)",
                min_value=0.0,
                max_value=15.0,
                value=4.5,
                step=0.25,
                help="Select average daily study duration in hours."
            )
            
            st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("### 🎯 Predicted Result")
        
        # Prepare input payload matching training feature names
        input_data = pd.DataFrame([[number_courses, time_study]], columns=['number_courses', 'time_study'])
        
        # Real-time Prediction
        try:
            prediction = model.predict(input_data)[0]
            
            # Render Styled Score Box
            st.markdown(f"""
                <div class="prediction-container">
                    <div class="prediction-title">Estimated Target Score</div>
                    <div class="prediction-value">{prediction:.2f}</div>
                </div>
            """, unsafe_allow_html=True)
            
            st.write("")
            
            # Quick summary / insight metrics
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.metric(label="Selected Courses", value=f"{number_courses}")
            with m_col2:
                st.metric(label="Daily Hours", value=f"{time_study} hrs")

        except Exception as e:
            st.error(f"Error during inference: {e}")

    # Section for Batch Inputs / Explanations
    st.divider()
    with st.expander("📊 View Input Feature Summary"):
        st.dataframe(input_data, use_container_width=True)
