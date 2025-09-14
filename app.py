import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load model
@st.cache_resource
def load_model():
    return joblib.load('models/fraud_model.pkl')

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# App title
st.set_page_config(page_title="Fraud Detection", layout="wide")
st.title("💳 Credit Card Fraud Detection System")
st.markdown("---")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Train Model", "Detect Fraud", "Data Analysis"])

# Home Page
if page == "Home":
    st.header("About This Project")
    st.write("""
    This application detects fraudulent credit card transactions using machine learning.
    
    **Features:**
    - Real-time fraud detection
    - Model training dashboard
    - Data visualization
    - Performance metrics
    
    **How it works:**
    1. Train a Random Forest model on historical data
    2. Input transaction details for prediction
    3. Get instant fraud probability
    """)
    
    st.image("https://miro.medium.com/max/1400/1*7ZCrkXoKn9NqQwDm6rLXsg.png", 
             caption="Fraud Detection Process")

# Train Model Page
elif page == "Train Model":
    st.header("Model Training")
    
    if st.button("Train Model"):
        with st.spinner("Training model..."):
            try:
                # Load data
                df = pd.read_csv('data/creditcard.csv')
                
                # Prepare features and target
                X = df.drop(['Class'], axis=1)
                y = df['Class']
                
                # Train model
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )
                model.fit(X, y)
                
                # Save model
                joblib.dump(model, 'models/fraud_model.pkl')
                st.session_state.model_trained = True
                
                st.success("Model trained successfully!")
                
                # Show metrics
                st.subheader("Model Performance")
                y_pred = model.predict(X)
                accuracy = model.score(X, y)
                st.metric("Accuracy", f"{accuracy:.4f}")
                
                # Feature importance
                st.subheader("Feature Importance")
                feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                fig = px.bar(feature_importance.head(10), 
                             x='importance', y='feature', orientation='h')
                st.plotly_chart(fig)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Detect Fraud Page
elif page == "Detect Fraud":
    st.header("Fraud Detection")
    
    # Check if model exists
    try:
        model = load_model()
        st.session_state.model_trained = True
    except:
        st.warning("Model not found. Please train the model first.")
        st.stop()
    
    # Transaction input form
    st.subheader("Enter Transaction Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        time = st.number_input("Time (seconds)", min_value=0, value=0)
        v1 = st.number_input("V1", value=0.0)
        v2 = st.number_input("V2", value=0.0)
        v3 = st.number_input("V3", value=0.0)
        v4 = st.number_input("V4", value=0.0)
        v5 = st.number_input("V5", value=0.0)
        v6 = st.number_input("V6", value=0.0)
        v7 = st.number_input("V7", value=0.0)
        v8 = st.number_input("V8", value=0.0)
        v9 = st.number_input("V9", value=0.0)
        v10 = st.number_input("V10", value=0.0)
    
    with col2:
        v11 = st.number_input("V11", value=0.0)
        v12 = st.number_input("V12", value=0.0)
        v13 = st.number_input("V13", value=0.0)
        v14 = st.number_input("V14", value=0.0)
        v15 = st.number_input("V15", value=0.0)
        v16 = st.number_input("V16", value=0.0)
        v17 = st.number_input("V17", value=0.0)
        v18 = st.number_input("V18", value=0.0)
        v19 = st.number_input("V19", value=0.0)
        v20 = st.number_input("V20", value=0.0)
        amount = st.number_input("Amount ($)", min_value=0.0, value=0.0)
    
    # Additional features (V21-V28)
    hidden_features = st.expander("Additional Features (V21-V28)")
    with hidden_features:
        v21 = st.number_input("V21", value=0.0)
        v22 = st.number_input("V22", value=0.0)
        v23 = st.number_input("V23", value=0.0)
        v24 = st.number_input("V24", value=0.0)
        v25 = st.number_input("V25", value=0.0)
        v26 = st.number_input("V26", value=0.0)
        v27 = st.number_input("V27", value=0.0)
        v28 = st.number_input("V28", value=0.0)
    
    # Prediction button
    if st.button("Detect Fraud"):
        # Create input array
        input_data = np.array([[
            time, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
            v11, v12, v13, v14, v15, v16, v17, v18, v19, v20,
            v21, v22, v23, v24, v25, v26, v27, v28, amount
        ]])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        
        # Display results
        st.subheader("Prediction Results")
        if prediction == 1:
            st.error("🚨 FRAUD DETECTED!")
            st.metric("Fraud Probability", f"{probability[1]:.2%}")
        else:
            st.success("✅ Transaction is Legitimate")
            st.metric("Legitimate Probability", f"{probability[0]:.2%}")
        
        # Visualization
        fig, ax = plt.subplots()
        ax.bar(['Legitimate', 'Fraud'], probability)
        ax.set_ylabel('Probability')
        ax.set_title('Fraud Probability')
        st.pyplot(fig)

# Data Analysis Page
elif page == "Data Analysis":
    st.header("Dataset Analysis")
    
    try:
        df = pd.read_csv('data/creditcard.csv')
        
        # Class distribution
        st.subheader("Class Distribution")
        class_counts = df['Class'].value_counts()
        st.write(f"Legitimate: {class_counts[0]} ({class_counts[0]/len(df)*100:.2f}%)")
        st.write(f"Fraud: {class_counts[1]} ({class_counts[1]/len(df)*100:.2f}%)")
        
        fig = px.pie(values=class_counts.values, 
                     names=['Legitimate', 'Fraud'],
                     title='Transaction Class Distribution')
        st.plotly_chart(fig)
        
        # Amount distribution
        st.subheader("Transaction Amount Distribution")
        fig = px.histogram(df, x='Amount', color='Class', 
                          title='Amount Distribution by Class',
                          log_y=True)
        st.plotly_chart(fig)
        
        # Time distribution
        st.subheader("Transaction Time Distribution")
        fig = px.histogram(df, x='Time', color='Class',
                          title='Time Distribution by Class')
        st.plotly_chart(fig)
        
        # Correlation matrix (sample)
        st.subheader("Feature Correlation (Sample)")
        sample_df = df.sample(1000)
        corr = sample_df.corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto",
                       title="Feature Correlation Matrix")
        st.plotly_chart(fig)
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")