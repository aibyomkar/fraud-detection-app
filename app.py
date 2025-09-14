import os
import subprocess
import sys
import urllib.request
import shutil
import stat
import time
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
        print("Git LFS already installed.")
        return lfs_bin_path

    print("📥 Downloading Git LFS binary...")
    try:
        # Download Git LFS for Linux (64-bit)
        url = "https://github.com/git-lfs/git-lfs/releases/download/v3.7.0/git-lfs-linux-amd64-v3.7.0.tar.gz"
        tar_path = "/tmp/git-lfs.tar.gz"
        urllib.request.urlretrieve(url, tar_path)

        # Extract tar.gz
        subprocess.run(["tar", "-xzf", tar_path, "-C", "/tmp"], check=True, capture_output=True)

        # Move binary to /tmp/git-lfs
        extracted_bin = "/tmp/git-lfs-3.7.0/git-lfs"
        shutil.move(extracted_bin, lfs_bin_path)
        os.chmod(lfs_bin_path, stat.S_IEXEC | stat.S_IREAD | stat.S_IWRITE)  # Make executable

        # Clean up
        shutil.rmtree("/tmp/git-lfs-3.7.0", ignore_errors=True)
        os.remove(tar_path)

        print("✅ Git LFS installed successfully at", lfs_bin_path)
        return lfs_bin_path

    except Exception as e:
        st.error(f"❌ Failed to download/install Git LFS: {e}")
        raise
        # st.stop() # Let the exception propagate for better debugging if needed initially

# --- 2. ENSURE LARGE FILE IS AVAILABLE VIA LFS ---
def ensure_lfs_file(filepath):
    """Check if file exists; if not, install LFS and pull it."""
    if os.path.exists(filepath):
        print(f"✅ File '{filepath}' already exists.")
        return

    st.warning(f"⚠️ {filepath} not found. Attempting to download via Git LFS...")

    # Install Git LFS if not present
    git_lfs_bin = install_git_lfs()

    try:
        # Initialize LFS with custom binary
        result_init = subprocess.run([git_lfs_bin, "install"], capture_output=True, text=True)
        if result_init.returncode != 0:
             st.error(f"`git-lfs install` failed: {result_init.stderr}")
             st.stop()

        # Pull the large file
        result_pull = subprocess.run([git_lfs_bin, "pull"], capture_output=True, text=True)
        if result_pull.returncode != 0:
             st.error(f"`git-lfs pull` failed: {result_pull.stderr}")
             st.stop()

        # Add a small delay to ensure filesystem sync (sometimes needed in containers)
        time.sleep(2)

        # Verify file exists after pull
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath)
            st.success(f"✅ Successfully downloaded {filepath} via Git LFS. Size: {file_size / (1024*1024):.2f} MB")
        else:
             st.error(f"❌ Download appeared successful, but file '{filepath}' was not found afterwards.")
             # List files in data directory for debugging
             if os.path.exists("data"):
                 st.write("Files in 'data' directory:", os.listdir("data"))
             st.stop()

    except Exception as e:
        st.error(f"❌ Failed to download {filepath} via Git LFS: {e}")
        raise # Re-raise for Streamlit's error handler
        # st.stop()

# --- 3. LOAD DATA ---
DATA_FILE_PATH = "data/creditcard.csv"
ensure_lfs_file(DATA_FILE_PATH)

@st.cache_data(show_spinner="Loading dataset...", ttl=3600) # Cache for 1 hour
def load_data():
    if not os.path.exists(DATA_FILE_PATH):
        st.error(f"Critical Error: File '{DATA_FILE_PATH}' still not found when trying to load data!")
        st.stop()
    try:
        df = pd.read_csv(DATA_FILE_PATH)
        return df
    except Exception as e:
         st.error(f"Error loading CSV file: {e}")
         raise

try:
    df = load_data()
except Exception as e:
    st.error("Failed to load the dataset even after attempting LFS download.")
    st.exception(e) # Show the full traceback in the app for easier debugging
    st.stop()

# --- 4. PREPROCESSING ---
@st.cache_data(show_spinner="Preprocessing data...", ttl=3600)
def preprocess_data(df):
    X = df.drop(columns=['Class'])
    y = df['Class']

    scaler = StandardScaler()
    # It's generally better to fit the scaler on training data only, but for simplicity...
    X_scaled = X.copy()
    X_scaled['Amount'] = scaler.fit_transform(X[['Amount']])
    X_scaled['Time'] = scaler.fit_transform(X[['Time']])

    return X_scaled, y, scaler

X, y, scaler = preprocess_data(df)

# --- 5. TRAIN MODEL ---
@st.cache_resource(show_spinner="Training model...", ttl=3600)
def train_model(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    # Save model and scaler (optional, but good practice if retraining isn't desired on every run)
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    return model, X_test, y_test

model, X_test, y_test = train_model(X, y)

# --- 6. EVALUATION ---
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
try:
    auc_score = roc_auc_score(y_test, y_proba)
except ValueError as e:
    st.error(f"Error calculating AUC-ROC: {e}")
    auc_score = "N/A"

# --- 7. STREAMLIT UI ---
st.title("🔍 Fraud Detection App")
st.write("Predicts fraudulent credit card transactions using a Random Forest model.")

st.sidebar.header("📊 Model Performance")
if auc_score != "N/A":
    st.sidebar.metric("AUC-ROC Score", f"{auc_score:.4f}")
else:
    st.sidebar.write("AUC-ROC Score: Calculation Error")

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
try:
    report = classification_report(y_test, y_pred, target_names=['Legit', 'Fraud'], output_dict=True)
    st.json(report)
except ValueError as e:
    st.error(f"Error generating classification report: {e}")

# Prediction Interface
st.subheader("🔮 Make a Prediction")
st.write("Enter transaction details below:")

# Simplify input for demo purposes - just a few key features or a generic input
# Using V1, V2, V3, Amount, Time as examples
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    v1 = st.number_input("V1", value=0.0)
with col2:
    v2 = st.number_input("V2", value=0.0)
with col3:
    v3 = st.number_input("V3", value=0.0)
with col4:
    amount = st.number_input("Amount ($)", min_value=0.0, value=0.0)
with col5:
    time_val = st.number_input("Time", value=0.0)

# Create a placeholder input array (28 V features + Time + Amount)
# Fill mostly with zeros, override the ones we care about
input_data_list = [0.0] * 30 # V1 to V28 (index 0-27), Time (28), Amount (29)
input_data_list[0] = v1
input_data_list[1] = v2
input_data_list[2] = v3
input_data_list[28] = time_val
input_data_list[29] = amount
input_data = np.array([input_data_list])

if st.button("🔍 Detect Fraud"):
    try:
        # Scale Time and Amount features
        input_scaled = input_data.copy()
        input_scaled[0, 28] = scaler.transform([[time_val]])[0][0] # Scale Time
        input_scaled[0, 29] = scaler.transform([[amount]])[0][0]  # Scale Amount

        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        if prediction == 1:
            st.error(f"🚨 **FRAUD DETECTED!** Confidence: {probability:.2%}")
        else:
            st.success(f"✅ **LEGITIMATE TRANSACTION**. Confidence: {(1 - probability):.2%}")
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.exception(e)

# Footer
st.markdown("---")
st.caption("Built with Streamlit | Data: Credit Card Fraud Detection Dataset | Auto LFS Installed on Deploy")