import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

# Create necessary directories
for directory in ['data', 'models', '.streamlit']:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Load model function with error handling
@st.cache_resource
def load_model():
    try:
        return joblib.load('models/fraud_model.pkl')
    except FileNotFoundError:
        return None

# Validate uploaded dataset
def validate_dataset(df):
    """Validate that the uploaded dataset has required columns"""
    required_columns = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
                       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount', 'Class']
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    return len(missing_columns) == 0, missing_columns

# App title
st.set_page_config(page_title="Fraud Detection", layout="wide")
st.title("💳 Credit Card Fraud Detection System")
st.markdown("---")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Upload Dataset", "Train Model", "Detect Fraud", "Data Analysis"])

# Home Page
if page == "Home":
    st.header("About This Project")
    st.write("""
    This application detects fraudulent credit card transactions using machine learning.
    
    **How it works:**
    1. **Upload Dataset**: Provide the credit card transaction dataset
    2. **Train Model**: Train the fraud detection model
    3. **Detect Fraud**: Test individual transactions
    4. **Analyze Data**: Explore dataset patterns
    
    **Dataset Requirements:**
    - Must be the Credit Card Fraud Detection dataset from Kaggle
    - File: `creditcard.csv`
    - Columns: Time, V1-V28, Amount, Class
    """)
    
    st.info("ℹ️ **Next Step**: Go to 'Upload Dataset' to get started")
    
    st.subheader("Dataset Information")
    st.write("- **Source**: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)")
    st.write("- **Size**: ~150 MB (284,807 transactions)")
    st.write("- **Features**: 30 anonymized features + Amount + Time")
    st.write("- **Target**: Class (0 = Legitimate, 1 = Fraud)")

# Upload Dataset Page
elif page == "Upload Dataset":
    st.header("📤 Upload Dataset")
    
    st.write("### Instructions:")
    st.write("1. Download the dataset from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)")
    st.write("2. Download the `creditcard.csv` file")
    st.write("3. Upload it below:")
    
    uploaded_file = st.file_uploader("Choose CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            with st.spinner("Processing uploaded file..."):
                # Read the file
                stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                df = pd.read_csv(stringio)
                
                # Validate the dataset
                is_valid, missing_cols = validate_dataset(df)
                
                if is_valid:
                    # Save the file
                    df.to_csv('data/creditcard.csv', index=False)
                    st.success("✅ Dataset uploaded and validated successfully!")
                    st.success("✅ You can now proceed to 'Train Model'")
                    
                    # Show dataset info
                    st.subheader("Dataset Information")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Rows", f"{len(df):,}")
                    with col2:
                        st.metric("Columns", f"{len(df.columns)}")
                    with col3:
                        fraud_count = df['Class'].sum()
                        st.metric("Fraud Cases", f"{fraud_count:,}")
                    
                    # Show sample data
                    st.subheader("Sample Data")
                    st.dataframe(df.head())
                else:
                    st.error(f"❌ Invalid dataset format. Missing columns: {missing_cols}")
                    st.write("Please upload the correct creditcard.csv file from Kaggle.")
                    
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

# Train Model Page
elif page == "Train Model":
    st.header("🤖 Model Training")
    
    # Check if dataset exists
    if not os.path.exists('data/creditcard.csv'):
        st.warning("⚠️ Dataset not found! Please upload the dataset first.")
        st.info("Go to 'Upload Dataset' section to upload your creditcard.csv file")
        st.stop()
    
    try:
        # Load data
        df = pd.read_csv('data/creditcard.csv')
        st.success(f"✅ Dataset loaded: {len(df):,} rows")
    except Exception as e:
        st.error(f"❌ Error loading dataset: {str(e)}")
        st.stop()
    
    # Show data preview
    with st.expander("Dataset Preview"):
        st.dataframe(df.head())
    
    # Training options
    st.subheader("Training Options")
    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("Test Size (%)", 10, 50, 20)
    with col2:
        random_state = st.number_input("Random State", 0, 1000, 42)
    
    if st.button("🚀 Train Model"):
        with st.spinner("Training model... This may take a few minutes..."):
            try:
                # Prepare features and target
                X = df.drop(['Class'], axis=1)
                y = df['Class']
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size/100, random_state=random_state, stratify=y
                )
                
                # Show training info
                st.info(f"Training on {len(X_train):,} samples, testing on {len(X_test):,} samples")
                
                # Train model
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=random_state
                )
                model.fit(X_train, y_train)
                
                # Save model
                joblib.dump(model, 'models/fraud_model.pkl')
                
                # Make predictions
                y_pred = model.predict(X_test)
                accuracy = model.score(X_test, y_test)
                
                st.success("✅ Model trained successfully!")
                
                # Show metrics
                st.subheader("Model Performance")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Accuracy", f"{accuracy:.4f}")
                with col2:
                    st.metric("Fraud Cases", f"{y_test.sum()}")
                with col3:
                    st.metric("Test Samples", f"{len(y_test)}")
                
                # Classification report
                st.subheader("Detailed Metrics")
                report = classification_report(y_test, y_pred, output_dict=True)
                df_report = pd.DataFrame(report).transpose()
                st.dataframe(df_report)
                
                # Confusion Matrix
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig = px.imshow(cm, 
                               labels=dict(x="Predicted", y="Actual"),
                               x=['Legitimate', 'Fraud'],
                               y=['Legitimate', 'Fraud'],
                               text_auto=True)
                st.plotly_chart(fig)
                
                # Feature importance
                st.subheader("Top 15 Feature Importances")
                feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False).head(15)
                
                fig = px.bar(feature_importance, 
                            x='importance', y='feature', orientation='h',
                            title='Top 15 Most Important Features')
                st.plotly_chart(fig)
                
            except Exception as e:
                st.error(f"❌ Error during training: {str(e)}")

# Detect Fraud Page
elif page == "Detect Fraud":
    st.header("🔍 Fraud Detection")
    
    # Check if model exists
    model = load_model()
    if model is None:
        st.warning("⚠️ Model not found! Please train the model first.")
        st.info("Go to 'Train Model' section to train your fraud detection model")
        st.stop()
    
    # Transaction input form
    st.subheader("Enter Transaction Details")
    
    # Create a more user-friendly input form
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Basic Info**")
        time = st.number_input("Time (seconds)", min_value=0, value=0, help="Time of transaction")
        amount = st.number_input("Amount ($)", min_value=0.0, value=0.0, help="Transaction amount", step=0.01)
        
        st.markdown("**V Features (1-9)**")
        v1 = st.number_input("V1", value=0.0)
        v2 = st.number_input("V2", value=0.0)
        v3 = st.number_input("V3", value=0.0)
        v4 = st.number_input("V4", value=0.0)
        v5 = st.number_input("V5", value=0.0)
        v6 = st.number_input("V6", value=0.0)
        v7 = st.number_input("V7", value=0.0)
        v8 = st.number_input("V8", value=0.0)
        v9 = st.number_input("V9", value=0.0)
    
    with col2:
        st.markdown("**V Features (10-18)**")
        v10 = st.number_input("V10", value=0.0)
        v11 = st.number_input("V11", value=0.0)
        v12 = st.number_input("V12", value=0.0)
        v13 = st.number_input("V13", value=0.0)
        v14 = st.number_input("V14", value=0.0)
        v15 = st.number_input("V15", value=0.0)
        v16 = st.number_input("V16", value=0.0)
        v17 = st.number_input("V17", value=0.0)
        v18 = st.number_input("V18", value=0.0)
    
    with col3:
        st.markdown("**V Features (19-28)**")
        v19 = st.number_input("V19", value=0.0)
        v20 = st.number_input("V20", value=0.0)
        v21 = st.number_input("V21", value=0.0)
        v22 = st.number_input("V22", value=0.0)
        v23 = st.number_input("V23", value=0.0)
        v24 = st.number_input("V24", value=0.0)
        v25 = st.number_input("V25", value=0.0)
        v26 = st.number_input("V26", value=0.0)
        v27 = st.number_input("V27", value=0.0)
        v28 = st.number_input("V28", value=0.0)
    
    # Prediction button
    st.markdown("---")
    if st.button("🔍 Detect Fraud", type="primary", use_container_width=True):
        # Create input array
        input_data = np.array([[
            time, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
            v11, v12, v13, v14, v15, v16, v17, v18, v19, v20,
            v21, v22, v23, v24, v25, v26, v27, v28, amount
        ]]).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        
        # Display results
        st.subheader("Prediction Results")
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
            if prediction == 1:
                st.error("🚨 FRAUD DETECTED!")
                st.metric("Fraud Probability", f"{probability[1]:.2%}")
            else:
                st.success("✅ Transaction is Legitimate")
                st.metric("Legitimate Probability", f"{probability[0]:.2%}")
        
        with result_col2:
            # Visualization
            fig = go.Figure(data=[go.Bar(
                x=['Legitimate', 'Fraud'],
                y=probability,
                marker_color=['green', 'red']
            )])
            fig.update_layout(title="Fraud Probability")
            st.plotly_chart(fig)
        
        # Additional insights
        if prediction == 1:
            st.info("⚠️ This transaction has been flagged as potentially fraudulent. Please review manually.")
        else:
            st.info("✅ This transaction appears to be legitimate based on the model.")

# Data Analysis Page
elif page == "Data Analysis":
    st.header("📊 Dataset Analysis")
    
    # Check if dataset exists
    if not os.path.exists('data/creditcard.csv'):
        st.warning("⚠️ Dataset not found! Please upload the dataset first.")
        st.info("Go to 'Upload Dataset' section to upload your creditcard.csv file")
        st.stop()
    
    try:
        with st.spinner("Loading dataset..."):
            df = pd.read_csv('data/creditcard.csv')
        
        st.success(f"✅ Dataset loaded: {len(df):,} rows")
        
        # Class distribution
        st.subheader("Class Distribution")
        class_counts = df['Class'].value_counts()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Legitimate (0)", f"{class_counts[0]:,}")
        with col2:
            st.metric("Fraud (1)", f"{class_counts[1]:,}")
        with col3:
            st.metric("Total", f"{len(df):,}")
        with col4:
            st.metric("Fraud Rate", f"{class_counts[1]/len(df)*100:.4f}%")
        
        fig = px.pie(values=class_counts.values, 
                     names=['Legitimate', 'Fraud'],
                     title='Transaction Class Distribution')
        st.plotly_chart(fig)
        
        # Amount distribution
        st.subheader("Transaction Amount Distribution")
        fig = px.histogram(df, x='Amount', color='Class', 
                          title='Amount Distribution by Class',
                          log_y=True, nbins=50)
        st.plotly_chart(fig)
        
        # Time distribution
        st.subheader("Transaction Time Distribution")
        fig = px.histogram(df, x='Time', color='Class',
                          title='Time Distribution by Class',
                          nbins=50)
        st.plotly_chart(fig)
        
        # Statistical summary
        st.subheader("Statistical Summary")
        st.dataframe(df.describe())
        
        # Sample data
        st.subheader("Sample Data")
        st.dataframe(df.head(10))
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")