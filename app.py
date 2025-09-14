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
        # Ensure it's executable
        os.chmod(lfs_bin_path, stat.S_IEXEC | stat.S_IREAD | stat.S_IWRITE)
        return lfs_bin_path

    print("📥 Downloading Git LFS binary...")
    try:
        # --- CRITICAL DEBUGGING ---
        st.write("Debug: Installing Git LFS...")
        # --- END CRITICAL DEBUGGING ---

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
        # --- CRITICAL DEBUGGING ---
        st.write(f"Debug: Git LFS installed at {lfs_bin_path}")
        # --- END CRITICAL DEBUGGING ---
        return lfs_bin_path

    except Exception as e:
        st.error(f"❌ Failed to download/install Git LFS: {e}")
        st.exception(e)
        st.stop() # Stop execution if LFS can't be installed

# --- 2. ENSURE LARGE FILE IS AVAILABLE VIA LFS ---
def ensure_lfs_file(filepath):
    """Check if file exists; if not, install LFS and pull it."""
    # --- CRITICAL DEBUGGING ---
    current_dir = os.getcwd()
    st.write("Debug: Current Working Directory:", current_dir)
    st.write("Debug: Contents of current directory:", os.listdir(current_dir))
    data_dir = os.path.dirname(filepath)
    if os.path.exists(data_dir):
        st.write(f"Debug: Contents of '{data_dir}' directory:", os.listdir(data_dir))
    else:
        st.write(f"Debug: Data directory '{data_dir}' does not exist.")
    # --- END CRITICAL DEBUGGING ---

    if os.path.exists(filepath):
        file_size = os.path.getsize(filepath)
        st.success(f"✅ File '{filepath}' already exists. Size: {file_size / (1024*1024):.2f} MB")
        return

    st.warning(f"⚠️ {filepath} not found. Attempting to download via Git LFS...")

    # Install Git LFS if not present
    git_lfs_bin = install_git_lfs()

    try:
        # --- CRITICAL DEBUGGING ---
        st.write("Debug: Running `git-lfs install`...")
        # --- END CRITICAL DEBUGGING ---
        # Initialize LFS with custom binary - Explicitly set cwd
        result_init = subprocess.run(
            [git_lfs_bin, "install"],
            capture_output=True,
            text=True,
            cwd=current_dir
        )
        if result_init.returncode != 0:
            st.error(f"`git-lfs install` failed (Exit Code {result_init.returncode}): {result_init.stderr}")
            st.code(result_init.stdout) # Sometimes output is in stdout
            st.stop()

        # --- CRITICAL DEBUGGING ---
        st.write("Debug: Running `git-lfs fetch`...")
        # --- END CRITICAL DEBUGGING ---
        # Fetch first - Explicitly set cwd
        result_fetch = subprocess.run(
            [git_lfs_bin, "fetch"],
            capture_output=True,
            text=True,
            cwd=current_dir
        )
        # Fetch can have non-zero exit on partial success, so check stderr
        if result_fetch.stderr:
             st.warning(f"`git-lfs fetch` reported messages/warnings: {result_fetch.stderr}")
        st.write("Debug: `git-lfs fetch` completed (check warnings above).")
        st.code(result_fetch.stdout)


        # --- CRITICAL DEBUGGING ---
        st.write("Debug: Running `git-lfs pull`...")
        # --- END CRITICAL DEBUGGING ---
        # Pull the large file - Use check=True and explicit cwd
        result_pull = subprocess.run(
            [git_lfs_bin, "pull"],
            check=True, # This will raise CalledProcessError if exit code != 0
            capture_output=True,
            text=True,
            cwd=current_dir
        )
        # If we reach here, pull command succeeded (exit code 0)
        st.write("Debug: `git-lfs pull` command succeeded (Exit Code 0).")
        st.code(result_pull.stdout)
        if result_pull.stderr:
            st.write("Debug: `git-lfs pull` stderr (might be warnings):", result_pull.stderr)

        # Add a small delay to ensure filesystem sync (sometimes needed in containers)
        time.sleep(3) # Increased slightly

        # --- CRITICAL VERIFICATION ---
        st.write("Debug: Checking file status after `git-lfs pull`...")
        # Re-check directory contents
        if os.path.exists(data_dir):
            st.write(f"Debug: Contents of '{data_dir}' directory after pull:", os.listdir(data_dir))
        else:
            st.write(f"❌ Data directory '{data_dir}' still does not exist after pull.")

        # Check if file now exists
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath)
            st.success(f"✅ Successfully downloaded {filepath} via Git LFS. Size: {file_size / (1024*1024):.2f} MB")
        else:
            st.error(f"❌ `git-lfs pull` ran successfully, but file '{filepath}' was still not found.")
            # Run `git-lfs status` for more info
            try:
                st.write("Debug: Running `git-lfs status` for diagnostics...")
                lfs_status = subprocess.run(
                    [git_lfs_bin, "status"],
                    capture_output=True,
                    text=True,
                    cwd=current_dir
                )
                st.write("`git-lfs status` output:", lfs_status.stdout)
                if lfs_status.stderr:
                    st.write("`git-lfs status` errors/warnings:", lfs_status.stderr)
            except Exception as status_e:
                st.write("Failed to run `git-lfs status`:", status_e)
            st.stop()

    except subprocess.CalledProcessError as e: # Catch specific error from check=True
        st.error(f"❌ `git-lfs pull` command failed (Exit Code {e.returncode}):")
        st.code(e.stderr) # Show stderr from the failed command
        st.code(e.stdout) # Sometimes useful info is in stdout
        # Add status check on failure
        try:
            st.write("Debug: Running `git-lfs status` after pull failure...")
            lfs_status = subprocess.run(
                [git_lfs_bin, "status"],
                capture_output=True,
                text=True,
                cwd=current_dir
            )
            st.write("`git-lfs status` output (on failure):", lfs_status.stdout)
            if lfs_status.stderr:
                st.write("`git-lfs status` errors/warnings (on failure):", lfs_status.stderr)
        except Exception as se:
            st.write("Failed to run `git-lfs status` after failure:", se)
        st.stop()
    except Exception as e:
        st.error(f"❌ Unexpected error during LFS download: {e}")
        st.exception(e) # Show full traceback
        st.stop()

# --- 3. LOAD DATA ---
DATA_FILE_PATH = "data/creditcard.csv"
ensure_lfs_file(DATA_FILE_PATH) # This will stop execution if file is not available

@st.cache_data(show_spinner="Loading dataset...", ttl=3600) # Cache for 1 hour
def load_data():
    # Final check before loading
    if not os.path.exists(DATA_FILE_PATH):
        st.error(f"Critical Error: File '{DATA_FILE_PATH}' still not found when trying to load data!")
        st.stop()
    try:
        df = pd.read_csv(DATA_FILE_PATH)
        return df
    except Exception as e:
         st.error(f"Error loading CSV file: {e}")
         st.exception(e)
         st.stop() # Stop if data can't be loaded

# Attempt to load data
try:
    df = load_data()
except Exception as e:
    # load_data should already handle errors, but just in case
    st.error("Failed to load the dataset.")
    st.exception(e)
    st.stop()

# --- 4. PREPROCESSING ---
@st.cache_data(show_spinner="Preprocessing data...", ttl=3600)
def preprocess_data(df):
    X = df.drop(columns=['Class'])
    y = df['Class']

    scaler = StandardScaler()
    # It's generally better to fit the scaler on training data only, but for simplicity in a demo app...
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

    # Save model and scaler (optional)
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
    # st.json(report) # JSON can be hard to read
    # Display key metrics in a table
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)
except ValueError as e:
    st.error(f"Error generating classification report: {e}")

# Prediction Interface (Simplified)
st.subheader("🔮 Make a Prediction")
st.write("Enter transaction details below (simplified for demo):")

# Simple input for key features or use default values from the dataset for testing
col1, col2, col3 = st.columns(3)
with col1:
    # Use median values for a "normal" transaction as defaults
    v1_default = float(df['V1'].median()) if not df['V1'].empty else 0.0
    v1 = st.number_input("V1 (Median ~{:.2f})".format(v1_default), value=v1_default, format="%.6f")
with col2:
    v2_default = float(df['V2'].median()) if not df['V2'].empty else 0.0
    v2 = st.number_input("V2 (Median ~{:.2f})".format(v2_default), value=v2_default, format="%.6f")
with col3:
    amount_default = float(df['Amount'].median()) if not df['Amount'].empty else 0.0
    amount = st.number_input("Amount ($) (Median ~{:.2f})".format(amount_default), min_value=0.0, value=amount_default, format="%.2f")

# Create a placeholder input array (28 V features + Time + Amount)
# Initialize with medians or zeros
input_data_list = [float(df[f'V{i}'].median()) if not df[f'V{i}'].empty else 0.0 for i in range(1, 29)]
input_data_list.append(float(df['Time'].median()) if not df['Time'].empty else 0.0) # Time
input_data_list.append(amount) # Override Amount

# Override V1 and V2 with user input
input_data_list[0] = v1 # V1
input_data_list[1] = v2 # V2

input_data = np.array([input_data_list])

if st.button("🔍 Detect Fraud"):
    try:
        # Scale Time and Amount features (assuming they are the last two)
        input_scaled = input_data.copy()
        # Time is index -2, Amount is index -1
        input_scaled[0, -2] = scaler.transform([[input_scaled[0, -2]]])[0][0] # Scale Time
        input_scaled[0, -1] = scaler.transform([[input_scaled[0, -1]]])[0][0]  # Scale Amount

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
st.caption("Built with Streamlit | Data: Credit Card Fraud Detection Dataset | Auto LFS Installed & Downloaded on Deploy")