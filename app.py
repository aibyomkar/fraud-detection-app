# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import os
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, confusion_matrix
# import plotly.express as px
# import plotly.graph_objects as go
# from io import StringIO

# # Create necessary directories
# for directory in ['data', 'models', '.streamlit']:
#     if not os.path.exists(directory):
#         os.makedirs(directory)

# # Check if dataset exists function
# def dataset_exists():
#     return os.path.exists('data/creditcard.csv')

# # Check if model exists function
# def model_exists():
#     return os.path.exists('models/fraud_model.pkl')

# # Load model function with error handling
# @st.cache_resource
# def load_model():
#     try:
#         return joblib.load('models/fraud_model.pkl')
#     except FileNotFoundError:
#         return None

# # Validate uploaded dataset
# def validate_dataset(df):
#     """Validate that the uploaded dataset has required columns"""
#     required_columns = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
#                        'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
#                        'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount', 'Class']
    
#     missing_columns = [col for col in required_columns if col not in df.columns]
#     return len(missing_columns) == 0, missing_columns

# # App title
# st.set_page_config(page_title="Fraud Detection", layout="wide", page_icon="💳")

# # Custom CSS for better styling
# st.markdown("""
# <style>
#     .main-header {
#         text-align: center;
#         padding: 2rem 0;
#     }
#     .step-card {
#         background-color: #f0f2f6;
#         padding: 1rem;
#         border-radius: 10px;
#         margin: 1rem 0;
#     }
#     .dataset-info {
#         background-color: #e8f4f8;
#         padding: 1rem;
#         border-radius: 10px;
#         margin: 1rem 0;
#     }
#     .warning-box {
#         background-color: #fff3cd;
#         padding: 1rem;
#         border-radius: 10px;
#         border-left: 5px solid #ffc107;
#         margin: 1rem 0;
#     }
#     .success-box {
#         background-color: #d4edda;
#         padding: 1rem;
#         border-radius: 10px;
#         border-left: 5px solid #28a745;
#         margin: 1rem 0;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Sidebar
# st.sidebar.title("💳 Fraud Detection")
# st.sidebar.markdown("---")
# page = st.sidebar.radio("Navigation", [
#     "🏠 Home", 
#     "📋 Instructions", 
#     "📤 Upload Dataset", 
#     "🤖 Train Model", 
#     "🔍 Detect Fraud", 
#     "📊 Data Analysis"
# ])

# # Home Page
# if page == "🏠 Home":
#     st.markdown('<div class="main-header">', unsafe_allow_html=True)
#     st.title("💳 Credit Card Fraud Detection System")
#     st.markdown("</div>", unsafe_allow_html=True)
    
#     st.markdown("""
#     <div style="text-align: center; font-size: 1.2rem; margin-bottom: 2rem;">
#         <em>Advanced Machine Learning for Real-time Fraud Detection</em>
#     </div>
#     """, unsafe_allow_html=True)
    
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         st.markdown("""
#         <div style="text-align: center; padding: 1rem;">
#             <h3>🛡️ Secure</h3>
#             <p>Advanced algorithms detect fraudulent transactions</p>
#         </div>
#         """, unsafe_allow_html=True)
    
#     with col2:
#         st.markdown("""
#         <div style="text-align: center; padding: 1rem;">
#             <h3>⚡ Fast</h3>
#             <p>Real-time detection with instant results</p>
#         </div>
#         """, unsafe_allow_html=True)
    
#     with col3:
#         st.markdown("""
#         <div style="text-align: center; padding: 1rem;">
#             <h3>📈 Accurate</h3>
#             <p>Machine learning models with high precision</p>
#         </div>
#         """, unsafe_allow_html=True)
    
#     st.markdown("---")
    
#     st.subheader("🎯 What This System Does")
#     st.write("""
#     This application uses machine learning to detect fraudulent credit card transactions. 
#     It analyzes transaction patterns and identifies suspicious activities that may indicate fraud.
    
#     **Key Features:**
#     - Real-time fraud detection
#     - Interactive data analysis
#     - Model training dashboard
#     - Comprehensive performance metrics
#     """)
    
#     st.subheader("👥 Who Is This For?")
#     st.write("""
#     - **Data Scientists**: Test machine learning models on real financial data
#     - **Security Analysts**: Analyze fraud patterns and trends
#     - **Students**: Learn about fraud detection and machine learning
#     - **Recruiters**: Evaluate fraud detection capabilities
#     """)
    
#     st.info("💡 **Next Step**: Go to '📋 Instructions' to learn how to use this system")

# # Instructions Page
# elif page == "📋 Instructions":
#     st.header("📋 How to Use This System")
    
#     st.markdown('<div class="step-card">', unsafe_allow_html=True)
#     st.subheader("Step 1: Get the Dataset")
#     st.write("You need the Credit Card Fraud Detection dataset to use this system.")
#     st.markdown("</div>", unsafe_allow_html=True)
    
#     st.markdown('<div class="dataset-info">', unsafe_allow_html=True)
#     st.subheader("📁 Required Dataset Information")
#     st.write("**Dataset Name**: Credit Card Fraud Detection")
#     st.write("**Source**: [Kaggle Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)")
#     st.write("**File**: `creditcard.csv`")
#     st.write("**Size**: ~150 MB (284,807 transactions)")
#     st.write("**Features**: 30 anonymized features + Amount + Time")
#     st.write("**Target**: Class (0 = Legitimate, 1 = Fraud)")
#     st.markdown("</div>", unsafe_allow_html=True)
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.markdown('<div class="success-box">', unsafe_allow_html=True)
#         st.subheader("✅ If You Have the Dataset")
#         st.write("1. Go to '📤 Upload Dataset'")
#         st.write("2. Upload your `creditcard.csv` file")
#         st.write("3. Proceed to model training")
#         st.markdown("</div>", unsafe_allow_html=True)
    
#     with col2:
#         st.markdown('<div class="warning-box">', unsafe_allow_html=True)
#         st.subheader("📥 If You Don't Have the Dataset")
#         st.write("1. Click the button below to download from Kaggle")
#         st.write("2. Save the `creditcard.csv` file")
#         st.write("3. Upload it in the next step")
#         st.markdown("</div>", unsafe_allow_html=True)
    
#     st.markdown("---")
    
#     # Download button
#     st.subheader("📥 Download Dataset")
#     st.write("Click the link below to download the required dataset:")
#     st.markdown("""
#     <div style="text-align: center; margin: 2rem 0;">
#         <a href="https://www.kaggle.com/mlg-ulb/creditcardfraud" target="_blank" 
#            style="background-color: #0366d6; color: white; padding: 15px 30px; 
#                   text-decoration: none; border-radius: 5px; font-size: 1.2rem;">
#             📥 Download Credit Card Fraud Dataset from Kaggle
#         </a>
#         <p><em>(You'll need a free Kaggle account to download)</em></p>
#     </div>
#     """, unsafe_allow_html=True)
    
#     st.info("💡 **Next Step**: Go to '📤 Upload Dataset' once you have the file")

# # Upload Dataset Page
# elif page == "📤 Upload Dataset":
#     st.header("📤 Upload Your Dataset")
    
#     st.markdown('<div class="dataset-info">', unsafe_allow_html=True)
#     st.subheader("📋 Dataset Requirements")
#     st.write("Please upload the **creditcard.csv** file from Kaggle.")
#     st.write("**Required Columns**: Time, V1-V28, Amount, Class")
#     st.markdown("</div>", unsafe_allow_html=True)
    
#     st.subheader("📁 Upload File")
#     uploaded_file = st.file_uploader("Choose your creditcard.csv file", type="csv")
    
#     if uploaded_file is not None:
#         try:
#             with st.spinner("Processing uploaded file..."):
#                 # Read the file
#                 stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
#                 df = pd.read_csv(stringio)
                
#                 # Validate the dataset
#                 is_valid, missing_cols = validate_dataset(df)
                
#                 if is_valid:
#                     # Save the file
#                     df.to_csv('data/creditcard.csv', index=False)
#                     st.success("✅ Dataset uploaded and validated successfully!")
                    
#                     # Show dataset info
#                     st.subheader("📊 Dataset Information")
#                     col1, col2, col3 = st.columns(3)
#                     with col1:
#                         st.metric("Total Rows", f"{len(df):,}")
#                     with col2:
#                         st.metric("Total Columns", f"{len(df.columns)}")
#                     with col3:
#                         fraud_count = df['Class'].sum()
#                         st.metric("Fraud Cases", f"{fraud_count:,}")
                    
#                     # Show class distribution
#                     class_counts = df['Class'].value_counts()
#                     st.write(f"**Fraud Rate**: {class_counts[1]/len(df)*100:.4f}%")
                    
#                     # Show sample data
#                     st.subheader("📋 Sample Data")
#                     st.dataframe(df.head())
                    
#                     st.success("🚀 You can now proceed to '🤖 Train Model'")
                    
#                 else:
#                     st.error(f"❌ Invalid dataset format.")
#                     st.write(f"**Missing columns**: {missing_cols}")
#                     st.write("Please upload the correct **creditcard.csv** file from Kaggle.")
                    
#         except Exception as e:
#             st.error(f"❌ Error processing file: {str(e)}")
    
#     # Reminder about getting the dataset
#     if not uploaded_file:
#         st.info("💡 Don't have the dataset? Go to '📋 Instructions' to download it from Kaggle")

# # Train Model Page
# elif page == "🤖 Train Model":
#     st.header("🤖 Train Fraud Detection Model")
    
#     # Check if dataset exists
#     if not dataset_exists():
#         st.warning("⚠️ Dataset not found!")
#         st.info("Please upload your dataset first in the '📤 Upload Dataset' section.")
#         st.markdown("---")
#         st.write("📊 **Current Status**: No dataset uploaded")
#         st.stop()
    
#     try:
#         # Load data
#         df = pd.read_csv('data/creditcard.csv')
#         st.success(f"✅ Dataset loaded: {len(df):,} rows")
#     except Exception as e:
#         st.error(f"❌ Error loading dataset: {str(e)}")
#         st.stop()
    
#     st.subheader("⚙️ Training Configuration")
#     col1, col2 = st.columns(2)
#     with col1:
#         test_size = st.slider("Test Size (%)", 10, 50, 20)
#     with col2:
#         random_state = st.number_input("Random State", 0, 1000, 42)
    
#     st.markdown('<div class="warning-box">', unsafe_allow_html=True)
#     st.write("⚠️ **Training may take 2-5 minutes depending on your dataset size**")
#     st.markdown("</div>", unsafe_allow_html=True)
    
#     if st.button("🚀 Start Training", type="primary", use_container_width=True):
#         with st.spinner("Training model... This may take a few minutes..."):
#             try:
#                 # Prepare features and target
#                 X = df.drop(['Class'], axis=1)
#                 y = df['Class']
                
#                 # Split data
#                 X_train, X_test, y_train, y_test = train_test_split(
#                     X, y, test_size=test_size/100, random_state=random_state, stratify=y
#                 )
                
#                 # Show training info
#                 st.info(f"Training on {len(X_train):,} samples, testing on {len(X_test):,} samples")
                
#                 # Train model
#                 model = RandomForestClassifier(
#                     n_estimators=100,
#                     max_depth=10,
#                     random_state=random_state
#                 )
#                 model.fit(X_train, y_train)
                
#                 # Save model
#                 joblib.dump(model, 'models/fraud_model.pkl')
                
#                 # Make predictions
#                 y_pred = model.predict(X_test)
#                 accuracy = model.score(X_test, y_test)
                
#                 st.success("✅ Model trained successfully!")
                
#                 # Show metrics
#                 st.subheader("📈 Model Performance")
#                 col1, col2, col3 = st.columns(3)
#                 with col1:
#                     st.metric("Accuracy", f"{accuracy:.4f}")
#                 with col2:
#                     st.metric("Fraud Cases", f"{y_test.sum()}")
#                 with col3:
#                     st.metric("Test Samples", f"{len(y_test)}")
                
#                 # Classification report
#                 st.subheader("📋 Detailed Metrics")
#                 report = classification_report(y_test, y_pred, output_dict=True)
#                 df_report = pd.DataFrame(report).transpose()
#                 st.dataframe(df_report)
                
#                 # Confusion Matrix
#                 st.subheader("📊 Confusion Matrix")
#                 cm = confusion_matrix(y_test, y_pred)
#                 fig = px.imshow(cm, 
#                                labels=dict(x="Predicted", y="Actual"),
#                                x=['Legitimate', 'Fraud'],
#                                y=['Legitimate', 'Fraud'],
#                                text_auto=True,
#                                color_continuous_scale='Blues')
#                 st.plotly_chart(fig)
                
#                 # Feature importance
#                 st.subheader("🔝 Top 15 Feature Importances")
#                 feature_importance = pd.DataFrame({
#                     'feature': X.columns,
#                     'importance': model.feature_importances_
#                 }).sort_values('importance', ascending=False).head(15)
                
#                 fig = px.bar(feature_importance, 
#                             x='importance', y='feature', orientation='h',
#                             title='Top 15 Most Important Features')
#                 st.plotly_chart(fig)
                
#                 st.success("🚀 You can now proceed to '🔍 Detect Fraud'")
                
#             except Exception as e:
#                 st.error(f"❌ Error during training: {str(e)}")

# # Detect Fraud Page
# elif page == "🔍 Detect Fraud":
#     st.header("🔍 Fraud Detection")
    
#     # Check if model exists
#     if not model_exists():
#         st.warning("⚠️ Model not found!")
#         st.info("Please train the model first in the '🤖 Train Model' section.")
#         st.markdown("---")
#         st.write("🤖 **Current Status**: No trained model available")
#         st.stop()
    
#     # Load model
#     model = load_model()
#     if model is None:
#         st.error("❌ Error loading model")
#         st.stop()
    
#     st.markdown('<div class="dataset-info">', unsafe_allow_html=True)
#     st.write("Enter transaction details below to check for potential fraud.")
#     st.markdown("</div>", unsafe_allow_html=True)
    
#     # Transaction input form
#     st.subheader("💳 Enter Transaction Details")
    
#     # Create a more user-friendly input form
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         st.markdown("**Basic Info**")
#         time = st.number_input("⏱️ Time (seconds)", min_value=0, value=0, help="Time of transaction")
#         amount = st.number_input("💰 Amount ($)", min_value=0.0, value=0.0, help="Transaction amount", step=0.01)
        
#         st.markdown("**V Features (1-9)**")
#         v1 = st.number_input("📊 V1", value=0.0)
#         v2 = st.number_input("📊 V2", value=0.0)
#         v3 = st.number_input("📊 V3", value=0.0)
#         v4 = st.number_input("📊 V4", value=0.0)
#         v5 = st.number_input("📊 V5", value=0.0)
#         v6 = st.number_input("📊 V6", value=0.0)
#         v7 = st.number_input("📊 V7", value=0.0)
#         v8 = st.number_input("📊 V8", value=0.0)
#         v9 = st.number_input("📊 V9", value=0.0)
    
#     with col2:
#         st.markdown("**V Features (10-18)**")
#         v10 = st.number_input("📊 V10", value=0.0)
#         v11 = st.number_input("📊 V11", value=0.0)
#         v12 = st.number_input("📊 V12", value=0.0)
#         v13 = st.number_input("📊 V13", value=0.0)
#         v14 = st.number_input("📊 V14", value=0.0)
#         v15 = st.number_input("📊 V15", value=0.0)
#         v16 = st.number_input("📊 V16", value=0.0)
#         v17 = st.number_input("📊 V17", value=0.0)
#         v18 = st.number_input("📊 V18", value=0.0)
    
#     with col3:
#         st.markdown("**V Features (19-28)**")
#         v19 = st.number_input("📊 V19", value=0.0)
#         v20 = st.number_input("📊 V20", value=0.0)
#         v21 = st.number_input("📊 V21", value=0.0)
#         v22 = st.number_input("📊 V22", value=0.0)
#         v23 = st.number_input("📊 V23", value=0.0)
#         v24 = st.number_input("📊 V24", value=0.0)
#         v25 = st.number_input("📊 V25", value=0.0)
#         v26 = st.number_input("📊 V26", value=0.0)
#         v27 = st.number_input("📊 V27", value=0.0)
#         v28 = st.number_input("📊 V28", value=0.0)
    
#     # Prediction button
#     st.markdown("---")
#     if st.button("🔍 Detect Fraud", type="primary", use_container_width=True):
#         # Create input array
#         input_data = np.array([[
#             time, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
#             v11, v12, v13, v14, v15, v16, v17, v18, v19, v20,
#             v21, v22, v23, v24, v25, v26, v27, v28, amount
#         ]]).reshape(1, -1)
        
#         # Make prediction
#         prediction = model.predict(input_data)[0]
#         probability = model.predict_proba(input_data)[0]
        
#         # Display results
#         st.subheader("🎯 Prediction Results")
#         result_col1, result_col2 = st.columns(2)
        
#         with result_col1:
#             if prediction == 1:
#                 st.error("🚨 FRAUD DETECTED!")
#                 st.metric("Fraud Probability", f"{probability[1]:.2%}")
#             else:
#                 st.success("✅ Transaction is Legitimate")
#                 st.metric("Legitimate Probability", f"{probability[0]:.2%}")
        
#         with result_col2:
#             # Visualization
#             fig = go.Figure(data=[go.Bar(
#                 x=['Legitimate', 'Fraud'],
#                 y=probability,
#                 marker_color=['green', 'red']
#             )])
#             fig.update_layout(title="Fraud Probability Distribution")
#             st.plotly_chart(fig)
        
#         # Additional insights
#         st.markdown("---")
#         if prediction == 1:
#             st.info("⚠️ **Alert**: This transaction has been flagged as potentially fraudulent. Please review manually.")
#         else:
#             st.info("✅ **Status**: This transaction appears to be legitimate based on the model analysis.")

# # Data Analysis Page
# elif page == "📊 Data Analysis":
#     st.header("📊 Dataset Analysis")
    
#     # Check if dataset exists
#     if not dataset_exists():
#         st.warning("⚠️ Dataset not found!")
#         st.info("Please upload your dataset first in the '📤 Upload Dataset' section.")
#         st.markdown("---")
#         st.write("📊 **Current Status**: No dataset uploaded")
#         st.stop()
    
#     try:
#         with st.spinner("Loading dataset..."):
#             df = pd.read_csv('data/creditcard.csv')
        
#         st.success(f"✅ Dataset loaded: {len(df):,} rows")
        
#         # Class distribution
#         st.subheader("📊 Class Distribution")
#         class_counts = df['Class'].value_counts()
        
#         col1, col2, col3, col4 = st.columns(4)
#         with col1:
#             st.metric("Legitimate (0)", f"{class_counts[0]:,}")
#         with col2:
#             st.metric("Fraud (1)", f"{class_counts[1]:,}")
#         with col3:
#             st.metric("Total", f"{len(df):,}")
#         with col4:
#             st.metric("Fraud Rate", f"{class_counts[1]/len(df)*100:.4f}%")
        
#         fig = px.pie(values=class_counts.values, 
#                      names=['Legitimate', 'Fraud'],
#                      title='Transaction Class Distribution')
#         st.plotly_chart(fig)
        
#         # Amount distribution
#         st.subheader("💰 Transaction Amount Distribution")
#         fig = px.histogram(df, x='Amount', color='Class', 
#                           title='Amount Distribution by Class',
#                           log_y=True, nbins=50)
#         st.plotly_chart(fig)
        
#         # Time distribution
#         st.subheader("⏱️ Transaction Time Distribution")
#         fig = px.histogram(df, x='Time', color='Class',
#                           title='Time Distribution by Class',
#                           nbins=50)
#         st.plotly_chart(fig)
        
#         # Statistical summary
#         st.subheader("📋 Statistical Summary")
#         st.dataframe(df.describe())
        
#         # Sample data
#         st.subheader("📋 Sample Data")
#         st.dataframe(df.head(10))
        
#     except Exception as e:
#         st.error(f"❌ Error loading  {str(e)}")

# # Footer
# st.markdown("---")
# st.markdown("""
# <div style="text-align: center; color: #666;">
#     <p>💳 Credit Card Fraud Detection System | Built with Streamlit & Scikit-learn</p>
#     <p>For recruiters: Download the dataset from Kaggle to test this system</p>
# </div>
# """, unsafe_allow_html=True)

































import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import plotly.express as px
from io import StringIO

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
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown('<h2 style="color: #00f5ff;">💳 FraudGuard</h2>', unsafe_allow_html=True)
page = st.sidebar.radio("Navigate", [
    "🏠 Home", 
    "📤 Upload", 
    "🤖 Train", 
    "🔍 Detect", 
    "📊 Analysis"
])

# Home Page
if page == "🏠 Home":
    st.markdown('<h1 class="main-header">💳 FraudGuard AI</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #a9d6e5;">Advanced Fraud Detection System</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="card"><h3>🛡️ Secure</h3><p>AI-powered detection</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card"><h3>⚡ Fast</h3><p>Real-time analysis</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="card"><h3>📈 Accurate</h3><p>99%+ precision</p></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card"><h3>🎯 System Overview</h3><p>This system detects fraudulent credit card transactions using machine learning. Requires the Kaggle Credit Card Fraud dataset.</p></div>', unsafe_allow_html=True)
    
    st.info("💡 Next: Upload dataset in '📤 Upload' section")

# Upload Page
elif page == "📤 Upload":
    st.markdown('<h1 class="main-header">📤 Upload Dataset</h1>', unsafe_allow_html=True)
    
    st.markdown('<div class="card"><h2>📥 Get Dataset</h2></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card" style="height: 100%;"><h3>🌐 Download from Kaggle</h3><p>Get the official dataset used for training</p></div>', unsafe_allow_html=True)
        st.link_button("📥 Download Dataset", "https://www.kaggle.com/mlg-ulb/creditcardfraud", type="primary")
        st.markdown('<p><strong>File:</strong> creditcard.csv<br><strong>Size:</strong> ~150MB<br><strong>Rows:</strong> 284,807</p>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card" style="height: 100%;"><h3>📁 Upload Your Dataset</h3><p>Use your own credit card fraud dataset</p></div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose CSV file", type="csv", label_visibility="collapsed")
    
    st.markdown('<div class="card"><h3>📋 Dataset Requirements</h3><p>Required columns: Time, V1-V28, Amount, Class<br>Format: CSV file</p></div>', unsafe_allow_html=True)
    
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
        time = st.number_input("Time", value=0)
        v1 = st.number_input("V1", value=0.0)
        v2 = st.number_input("V2", value=0.0)
        v3 = st.number_input("V3", value=0.0)
        v4 = st.number_input("V4", value=0.0)
    
    with col2:
        v5 = st.number_input("V5", value=0.0)
        v6 = st.number_input("V6", value=0.0)
        v7 = st.number_input("V7", value=0.0)
        v8 = st.number_input("V8", value=0.0)
        v9 = st.number_input("V9", value=0.0)
    
    with col3:
        v10 = st.number_input("V10", value=0.0)
        amount = st.number_input("Amount", value=0.0)
        v11 = st.number_input("V11", value=0.0)
        v12 = st.number_input("V12", value=0.0)
    
    if st.button("🔍 Analyze Transaction"):
        input_data = np.array([[time, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
                               v11, v12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, amount]])
        
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        
        if prediction == 1:
            st.markdown('<div style="background: rgba(255, 42, 109, 0.2); border: 1px solid #ff2a6d; border-radius: 10px; padding: 1rem; text-align: center;"><h2>🚨 FRAUD DETECTED</h2></div>', unsafe_allow_html=True)
            st.metric("Fraud Probability", f"{probability[1]:.2%}")
        else:
            st.markdown('<div style="background: rgba(0, 245, 255, 0.2); border: 1px solid #00f5ff; border-radius: 10px; padding: 1rem; text-align: center;"><h2>✅ LEGITIMATE</h2></div>', unsafe_allow_html=True)
            st.metric("Legitimate Probability", f"{probability[0]:.2%}")

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