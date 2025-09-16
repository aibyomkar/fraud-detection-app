import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import plotly.express as px
from io import StringIO

# Disable file watching to prevent inotify limit error
os.environ['STREAMLIT_WATCHER_TYPE'] = 'none'
os.environ['STREAMLIT_SERVER_ENABLE_STATIC_SERVING'] = 'false'

# Create necessary directories
for directory in ['data', 'models']:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Check if dataset exists function
def dataset_exists():
    return os.path.exists('data/creditcard.csv')

# Check if model exists function
def model_exists():
    return os.path.exists('models/fraud_model.pkl')

# Load model function
@st.cache_resource
def load_model():
    try:
        return joblib.load('models/fraud_model.pkl')
    except FileNotFoundError:
        return None

# Validate uploaded dataset
def validate_dataset(df):
    required_columns = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
                       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount', 'Class']
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    return len(missing_columns) == 0, missing_columns

# Futuristic styling
st.markdown("""
<style>
    :root {
        --primary: #00f5ff;
        --secondary: #8a2be2;
        --dark: #0a0e17;
        --light: #0d1b2a;
        --accent: #ff2a6d;
        --text: #e0e1dd;
    }
    
    body {
        color: var(--text);
        background-color: var(--dark);
    }
    
    .stApp {
        background: linear-gradient(135deg, var(--dark) 0%, #1b263b 100%);
    }
    
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, var(--primary), var(--accent));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        padding: 1rem 0;
    }
    
    .card {
        background: rgba(13, 27, 42, 0.7);
        border: 1px solid rgba(0, 245, 255, 0.3);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 245, 255, 0.1);
    }
    
    .status-box {
        background: rgba(255, 42, 109, 0.1);
        border: 1px solid var(--accent);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .success-box {
        background: rgba(0, 245, 255, 0.1);
        border: 1px solid var(--primary);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .stButton>button {
        background: linear-gradient(90deg, var(--primary), var(--secondary));
        color: white;
        border: none;
        border-radius: 30px;
        padding: 0.75rem 1.5rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 245, 255, 0.4);
    }
    
    .download-btn {
        background: linear-gradient(90deg, #00c9ff, #92fe9d) !important;
        color: #0a0e17 !important;
        font-weight: bold !important;
    }
    
    .upload-btn {
        background: linear-gradient(90deg, var(--accent), #ff7bac) !important;
        color: white !important;
        font-weight: bold !important;
    }
    
    .or-container {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 100%;
        margin: 2rem 0;
    }
    
    .or-text {
        background: linear-gradient(90deg, var(--primary), var(--accent));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2rem;
        font-weight: bold;
        padding: 0 1rem;
    }
    
    .line {
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--primary), transparent);
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown('<h2 style="color: #00f5ff;">💳 Creddy</h2>', unsafe_allow_html=True)
page = st.sidebar.radio("Navigate", [
    "🏠 Home", 
    "📤 Upload", 
    "🤖 Train", 
    "🔍 Detect", 
    "📊 Analysis"
])

# Home Page
if page == "🏠 Home":
    st.markdown('<h1 class="main-header">💳 Creddy</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #a9d6e5;">Fraud Detection In Credit Card Transactions</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="card"><h3>🛡️ Secure</h3><p>ML-powered detection</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card"><h3>⚡ Fast</h3><p>Real-time analysis</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="card"><h3>📈 Accurate</h3><p>99%+ precision</p></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card"><h3>🎯 System Overview</h3><p>This system detects fraudulent credit card transactions using machine learning. Requires the Credit Card Fraud dataset.</p></div>', unsafe_allow_html=True)
    
    st.info("💡 Next: Upload dataset in '📤 Upload' section")

# Upload Page
elif page == "📤 Upload":
    st.markdown('<h1 class="main-header">📤 Upload Dataset</h1>', unsafe_allow_html=True)
    
    st.markdown('<div class="card"><h2>📋 Dataset Requirements</h2><p>Required columns: Time, V1-V28, Amount, Class<br>Format: CSV file</p></div>', unsafe_allow_html=True)
    
    # Create three columns: left content, OR, right content
    col1, col2, col3 = st.columns([4, 1, 4])

    with col1:
        st.markdown('<div class="card" style="height: 100%;"><h3>🌐 Download from Kaggle</h3><p>Get the official dataset used for training</p></div>', unsafe_allow_html=True)
        st.link_button("📥 Download Dataset", "https://www.kaggle.com/mlg-ulb/creditcardfraud", type="primary")
        
    with col2:
        st.markdown('<div class="or-container"><div class="line"></div><div class="or-text">OR</div><div class="line"></div></div>', unsafe_allow_html=True)
        
    with col3:
        st.markdown('<div class="card" style="height: 100%;"><h3>📁 Upload Your Dataset</h3><p>Use your own credit card fraud dataset</p></div>', unsafe_allow_html=True)
        
    uploaded_file = st.file_uploader("Choose CSV file", type="csv", label_visibility="collapsed")
        
    # Handle file upload
    if uploaded_file is not None:
        try:
            with st.spinner("Processing uploaded file..."):
                stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                df = pd.read_csv(stringio)
                
                is_valid, missing_cols = validate_dataset(df)
                
                if is_valid:
                    df.to_csv('data/creditcard.csv', index=False)
                    st.markdown('<div class="success-box"><h3>✅ Success!</h3><p>Dataset uploaded successfully</p></div>', unsafe_allow_html=True)
                    st.metric("Total Rows", f"{len(df):,}")
                    st.metric("Fraud Cases", f"{df['Class'].sum():,}")
                    st.success("🚀 You can now proceed to '🤖 Train' model")
                else:
                    st.error(f"❌ Invalid dataset format. Missing columns: {missing_cols}")
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

# Train Page
elif page == "🤖 Train":
    st.markdown('<h1 class="main-header">🤖 Train Model</h1>', unsafe_allow_html=True)
    
    if not dataset_exists():
        st.markdown('<div class="status-box"><h3>⚠️ No Dataset</h3><p>Upload dataset first in "📤 Upload" section</p></div>', unsafe_allow_html=True)
        st.stop()
    
    try:
        df = pd.read_csv('data/creditcard.csv')
        st.markdown('<div class="success-box"><h3>✅ Dataset Loaded</h3></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Rows", f"{len(df):,}")
        with col2:
            st.metric("Fraud Rate", f"{df['Class'].sum()/len(df)*100:.4f}%")
        
        if st.button("🚀 Train Model"):
            with st.spinner("Training... (2-3 mins)"):
                X = df.drop(['Class'], axis=1)
                y = df['Class']
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                
                model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
                model.fit(X_train, y_train)
                
                joblib.dump(model, 'models/fraud_model.pkl')
                
                accuracy = model.score(X_test, y_test)
                st.markdown('<div class="success-box"><h3>✅ Training Complete!</h3></div>', unsafe_allow_html=True)
                st.metric("Accuracy", f"{accuracy:.4f}")
                st.success("🚀 You can now proceed to '🔍 Detect' fraud")
                
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Detect Page
elif page == "🔍 Detect":
    st.markdown('<h1 class="main-header">🔍 Fraud Detection</h1>', unsafe_allow_html=True)
    
    if not model_exists():
        st.markdown('<div class="status-box"><h3>⚠️ No Model</h3><p>Train model first in "🤖 Train" section</p></div>', unsafe_allow_html=True)
        st.stop()
    
    model = load_model()
    if model is None:
        st.error("Model error")
        st.stop()
    
    st.markdown('<div class="card"><h3>💳 Transaction Details</h3></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        time = st.number_input("Time", value=0.0, format="%.2f")
        v1 = st.number_input("V1", value=0.0, format="%.6f")
        v2 = st.number_input("V2", value=0.0, format="%.6f")
        v3 = st.number_input("V3", value=0.0, format="%.6f")
        v4 = st.number_input("V4", value=0.0, format="%.6f")
    
    with col2:
        v5 = st.number_input("V5", value=0.0, format="%.6f")
        v6 = st.number_input("V6", value=0.0, format="%.6f")
        v7 = st.number_input("V7", value=0.0, format="%.6f")
        v8 = st.number_input("V8", value=0.0, format="%.6f")
        v9 = st.number_input("V9", value=0.0, format="%.6f")
    
    with col3:
        v10 = st.number_input("V10", value=0.0, format="%.6f")
        amount = st.number_input("Amount", value=0.0, format="%.2f")
        v11 = st.number_input("V11", value=0.0, format="%.6f")
        v12 = st.number_input("V12", value=0.0, format="%.6f")
        v13 = st.number_input("V13", value=0.0, format="%.6f")
        v14 = st.number_input("V14", value=0.0, format="%.6f")
    
    # Add remaining V columns
    v_columns = []
    for i in range(15, 29):
        v_columns.append(st.number_input(f"V{i}", value=0.0, format="%.6f"))
    
    if st.button("🔍 Analyze Transaction"):
        # Create complete input array with all 30 features (Time + V1-V28 + Amount)
        input_data = [time, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
                     v11, v12, v13, v14] + v_columns + [amount]
        input_data = np.array([input_data])
        
        try:
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0]
            
            if prediction == 1:
                st.markdown('<div style="background: rgba(255, 42, 109, 0.2); border: 1px solid #ff2a6d; border-radius: 10px; padding: 1rem; text-align: center;"><h2>🚨 FRAUD DETECTED</h2></div>', unsafe_allow_html=True)
                st.metric("Fraud Probability", f"{probability[1]:.2%}")
            else:
                st.markdown('<div style="background: rgba(0, 245, 255, 0.2); border: 1px solid #00f5ff; border-radius: 10px; padding: 1rem; text-align: center;"><h2>✅ LEGITIMATE</h2></div>', unsafe_allow_html=True)
                st.metric("Legitimate Probability", f"{probability[0]:.2%}")
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

# Analysis Page
elif page == "📊 Analysis":
    st.markdown('<h1 class="main-header">📊 Data Analysis</h1>', unsafe_allow_html=True)
    
    if not dataset_exists():
        st.markdown('<div class="status-box"><h3>⚠️ No Dataset</h3><p>Upload dataset first in "📤 Upload" section</p></div>', unsafe_allow_html=True)
        st.stop()
    
    try:
        df = pd.read_csv('data/creditcard.csv')
        st.markdown('<div class="success-box"><h3>✅ Dataset Loaded</h3></div>', unsafe_allow_html=True)
        
        class_counts = df['Class'].value_counts()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Legitimate", f"{class_counts[0]:,}")
        with col2:
            st.metric("Fraud", f"{class_counts[1]:,}")
        with col3:
            st.metric("Fraud Rate", f"{class_counts[1]/len(df)*100:.4f}%")
        
        fig = px.pie(values=class_counts.values, names=['Legitimate', 'Fraud'])
        st.plotly_chart(fig)
        
    except Exception as e:
        st.error(f"Error: {str(e)}")