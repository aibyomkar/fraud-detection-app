import os
import subprocess
import sys
import pandas as pd
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- FIX FOR STREAMLIT CLOUD: Auto-download large file via Git LFS ---
def ensure_lfs_file(filepath):
    if not os.path.exists(filepath):
        st.warning(f"⚠️ {filepath} not found. Attempting to download via Git LFS...")
        try:
            # Initialize Git LFS
            subprocess.run(["git", "lfs", "install"], check=True, capture_output=True)
            # Pull the large file
            subprocess.run(["git", "lfs", "pull"], check=True, capture_output=True)
            st.success(f"✅ Successfully downloaded {filepath} via Git LFS.")
        except subprocess.CalledProcessError as e:
            st.error(f"❌ Failed to download {filepath} via Git LFS:")
            st.code(e.stderr.decode())
            st.stop()
        except FileNotFoundError:
            st.error("❌ Git LFS is not installed on this system. Please contact support.")
            st.stop()

# Ensure creditcard.csv is available
ensure_lfs_file("data/creditcard.csv")

# --- Load Data ---
@st.cache_data(show_spinner="Loading dataset...")
def load_data():
    df = pd.read_csv("data/creditcard.csv")
    return df

df = load_data()

# --- Preprocessing ---
@st.cache_data(show_spinner="Preprocessing data...")
def preprocess_data(df):
    X = df.drop(columns=['Class'])
    y = df['Class']
    
    scaler = StandardScaler()
    X['Amount'] = scaler.fit_transform(X[['Amount']])
    X['Time'] = scaler.fit_transform(X[['Time']])
    
    return X, y, scaler

X, y, scaler = preprocess_data(df)

# --- Train Model ---
@st.cache_resource(show_spinner="Training model...")
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # Save model and scaler
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    
    return model, X_test, y_test

model, X_test, y_test = train_model(X, y)

# --- Evaluation ---
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_proba)

# --- Streamlit UI ---
st.title("🔍 Fraud Detection App")
st.write("Predicts fraudulent credit card transactions using a Random Forest model.")

st.sidebar.header("📊 Model Performance")
st.sidebar.metric("AUC-ROC Score", f"{auc_score:.4f}")
st.sidebar.metric("True Positives", np.sum((y_test == 1) & (y_pred == 1)))
st.sidebar.metric("False Negatives", np.sum((y_test == 1) & (y_pred == 0)))

# Confusion Matrix
st.subheader("📈 Confusion Matrix")
fig, ax = plt.subplots(figsize=(6, 4))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Legit', 'Fraud'], yticklabels=['Legit', 'Fraud'], ax=ax)
ax.set_title('Confusion Matrix')
st.pyplot(fig)

# Classification Report
st.subheader("📝 Classification Report")
report = classification_report(y_test, y_pred, target_names=['Legit', 'Fraud'], output_dict=True)
st.json(report)

# Prediction Interface
st.subheader("🔮 Make a Prediction")
st.write("Enter transaction details below:")

col1, col2, col3 = st.columns(3)
with col1:
    time = st.number_input("Time (seconds)", min_value=0.0, max_value=float(X['Time'].max()), value=0.0)
with col2:
    amount = st.number_input("Amount ($)", min_value=0.0, max_value=float(X['Amount'].max()), value=0.0)
with col3:
    v_features = []
    for i in range(1, 29):  # V1 to V28
        v_features.append(st.number_input(f"V{i}", value=0.0))

# Prepare input
input_data = np.array([[time, amount] + v_features])

# Predict
if st.button("🔍 Detect Fraud"):
    try:
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        if prediction == 1:
            st.error(f"🚨 **FRAUD DETECTED!** Confidence: {probability:.2%}")
        else:
            st.success(f"✅ **LEGITIMATE TRANSACTION**. Confidence: {(1 - probability):.2%}")
    except Exception as e:
        st.error(f"Prediction error: {e}")

# Footer
st.markdown("---")
st.caption("Built with Streamlit | Data: Credit Card Fraud Detection Dataset | Model: Random Forest | LFS Managed by GitHub")