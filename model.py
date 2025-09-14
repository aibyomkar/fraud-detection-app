import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def train_model():
    # Load data
    df = pd.read_csv('data/creditcard.csv')
    
    # Prepare features and target
    X = df.drop(['Class'], axis=1)
    y = df['Class']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Save model
    joblib.dump(model, 'models/fraud_model.pkl')
    
    # Print metrics
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    return model

if __name__ == "__main__":
    train_model()