```markdown
# 🚨 Fraud Detection App — Credit Card Fraud Prediction

A machine learning-powered application to detect fraudulent credit card transactions using the [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). Built with Python, scikit-learn, and Streamlit for real-time inference.

> 💡 **Dataset Size**: 144 MB — managed via **Git LFS** (Large File Storage) to comply with GitHub limits.

---

## 📌 Features

- ✅ Trained logistic regression & Random Forest models for fraud detection  
- ✅ Interactive web dashboard built with **Streamlit**  
- ✅ Real-time prediction interface with confidence scores  
- ✅ Model evaluation metrics: Precision, Recall, F1-Score, AUC-ROC  
- ✅ Data preprocessing pipeline (scaling, anomaly handling)  
- ✅ **Git LFS** used to safely track large dataset (`data/creditcard.csv`)

---

## 🚀 Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/aibyomkar/fraud-detection-app.git
cd fraud-detection-app
```

### 2. Install Git LFS (Critical!)

> ⚠️ **First-time users must run this before pulling files!**

```bash
git lfs install
```

This ensures the 144 MB `creditcard.csv` file downloads correctly.

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the App

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501` to interact with the app!

---

## 📂 Project Structure

```
fraud-detection-app/
├── data/
│   └── creditcard.csv           # 144MB dataset (managed via Git LFS)
├── models/
│   ├── model.pkl                # Trained Random Forest model
│   └── scaler.pkl               # StandardScaler for feature normalization
├── app.py                       # Streamlit frontend UI
├── train.py                     # Model training script
├── requirements.txt             # Python dependencies
├── README.md                    # You're here!
└── .gitignore                   # Ignores venv, __pycache__, etc.
```

---

## 📊 Model Performance (Sample)

| Metric       | Score     |
|--------------|-----------|
| Accuracy     | 99.94%    |
| Precision    | 0.87      |
| Recall       | 0.78      |
| F1-Score     | 0.82      |
| AUC-ROC      | 0.99      |

*Trained on 284,807 transactions (492 fraudulent cases)*

> 🔍 The model prioritizes **recall** to minimize false negatives — critical in fraud detection.

---

## 💡 Why Git LFS?

GitHub limits individual files to **100 MB**. Since the dataset exceeds this limit, we use **[Git Large File Storage (LFS)](https://git-lfs.github.com)** to store `creditcard.csv` externally while keeping its pointer in the repo.

👉 **Users must run `git lfs install` before cloning** to download the full dataset.

---

## 🤝 Contributing

Feel free to fork this repo and improve it! Suggestions welcome:
- Try other models (XGBoost, Neural Networks)
- Add explainability with SHAP values
- Deploy as a Docker container
- Add user authentication or API endpoints

Create a pull request with your enhancements!

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👋 Author

Built by **Omkar** — AI Engineer & ML Enthusiast  
🔗 Connect on 🐙 GitHub: [@aibyomkar](https://github.com/aibyomkar)

> *“Detecting fraud isn’t just about accuracy — it’s about saving people from financial harm.”*
```

---