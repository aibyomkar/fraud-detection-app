import os
import subprocess
import sys
import urllib.request
import shutil
import stat
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

# --- 1. AUTO-INSTALL GIT LFS ON STREAMLIT CLOUD ---
def install_git_lfs():
    """Download and install Git LFS binary for Linux (Streamlit Cloud)"""
    lfs_bin_path = "/tmp/git-lfs"
    if os.path.exists(lfs_bin_path):
        return

    print("📥 Downloading Git LFS binary...")
    try:
        # Download Git LFS for Linux (64-bit)
        url = "https://github.com/git-lfs/git-lfs/releases/download/v3.7.0/git-lfs-linux-amd64-v3.7.0.tar.gz"
        with urllib.request.urlopen(url) as response:
            with open("/tmp/git-lfs.tar.gz", "wb") as f:
                shutil.copyfileobj(response, f)

        # Extract tar.gz
        subprocess.run(["tar", "-xzf", "/tmp/git-lfs.tar.gz", "-C", "/tmp"], check=True)

        # Move binary to /tmp/git-lfs
        shutil.move("/tmp/git-lfs-3.7.0/git-lfs", lfs_bin_path)
        os.chmod(lfs_bin_path, stat.S_IEXEC | stat.S_IREAD | stat.S_IWRITE)  # Make executable

        # Clean up
        shutil.rmtree("/tmp/git-lfs-3.7.0")
        os.remove("/tmp/git-lfs.tar.gz")

        print("✅ Git LFS installed successfully at", lfs_bin_path)

    except Exception as e:
        st.error(f"❌ Failed to download/install Git LFS: {e}")
        st.stop()

# --- 2. ENSURE LARGE FILE IS AVAILABLE VIA LFS ---
def ensure_lfs_file(filepath):
    """Check if file exists; if not, install LFS and pull it."""
    if os.path.exists(filepath):
        return

    st.warning(f"⚠️ {filepath} not found. Attempting to download via Git LFS...")

    # Install Git LFS if not present
    install_git_lfs()

    # Configure Git to use our downloaded LFS binary
    git_lfs_bin = "/tmp/git-lfs"

    try:
        # Initialize LFS with custom binary
        subprocess.run([git_lfs_bin, "install"], check=True, capture_output=True)
        # Pull the large file
        subprocess.run([git_lfs_bin, "pull"], check=True, capture_output=True)
        st.success(f"✅ Successfully downloaded {filepath} via Git LFS.")
    except subprocess.CalledProcessError as e:
        st.error(f"❌ Failed to download {filepath} via Git LFS:")
        st.code(e.stderr.decode())
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        st.stop()

# --- 3. LOAD DATA ---
ensure_lfs_file("data/creditcard.csv")

@st.cache_data(show_spinner="Loading dataset...")
def load_data():
    df = pd.read_csv("data/creditcard.csv")
    return df

df = load_data()

# --- 4. PREPROCESSING ---
@st.cache_data(show_spinner="Preprocessing data...")
def preprocess_data(df):
    X = df.drop(columns=['Class'])
    y = df['Class']
    
    scaler = StandardScaler()
    X['Amount'] = scaler.fit_transform(X[['Amount']])
    X['Time'] = scaler.fit_transform(X[['Time']])
    
    return X, y, scaler

X, y, scaler = preprocess_data(df)

# --- 5. TRAIN MODEL ---
@st.cache_resource(show_spinner="Training model...")
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    
    return model, X_test, y_test

model, X_test, y_test = train_model(X, y)

# --- 6. EVALUATION ---
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_proba)

# --- 7. STREAMLIT UI ---
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

input_data = np.array([[time, amount] + v_features])

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
st.caption("Built with Streamlit | Data: Credit Card Fraud Detection Dataset | Model: Random Forest | Auto LFS Installed on Deploy")