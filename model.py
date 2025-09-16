import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

class FraudDetectionModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def create_sample_data(self):
        """Create sample data for demonstration"""
        np.random.seed(42)
        n_samples = 10000
        
        # Generate synthetic features
        data = {
            'transaction_amount': np.random.lognormal(3, 1.5, n_samples),
            'account_age_days': np.random.randint(1, 365*10, n_samples),
            'transaction_frequency': np.random.poisson(5, n_samples),
            'hour_of_day': np.random.randint(0, 24, n_samples),
            'is_weekend': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'merchant_risk_score': np.random.uniform(0, 1, n_samples),
            'user_risk_score': np.random.uniform(0, 1, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Create fraud labels based on some logic
        fraud_probability = (
            (df['transaction_amount'] > df['transaction_amount'].quantile(0.95)).astype(int) * 0.3 +
            (df['merchant_risk_score'] > 0.8).astype(int) * 0.4 +
            (df['user_risk_score'] > 0.7).astype(int) * 0.3 +
            (df['hour_of_day'].isin([2, 3, 4])).astype(int) * 0.2
        )
        
        df['is_fraud'] = np.random.binomial(1, np.clip(fraud_probability, 0, 0.8))
        
        return df
    
    def preprocess_data(self, df):
        """Preprocess the data"""
        # Feature engineering
        df['amount_log'] = np.log1p(df['transaction_amount'])
        df['is_high_amount'] = (df['transaction_amount'] > df['transaction_amount'].quantile(0.90)).astype(int)
        df['is_late_night'] = df['hour_of_day'].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
        
        # Select features for training
        feature_columns = [
            'transaction_amount', 'account_age_days', 'transaction_frequency',
            'hour_of_day', 'is_weekend', 'merchant_risk_score', 'user_risk_score',
            'amount_log', 'is_high_amount', 'is_late_night'
        ]
        
        X = df[feature_columns]
        y = df['is_fraud'] if 'is_fraud' in df.columns else None
        
        return X, y, feature_columns
    
    def train_model(self):
        """Train the fraud detection model"""
        # Create and prepare data
        df = self.create_sample_data()
        X, y, feature_columns = self.preprocess_data(df)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Save model and scaler
        joblib.dump(self.model, 'fraud_model.pkl')
        joblib.dump(self.scaler, 'scaler.pkl')
        
        return self.model
    
    def load_model(self):
        """Load pre-trained model"""
        try:
            self.model = joblib.load('fraud_model.pkl')
            self.scaler = joblib.load('scaler.pkl')
            self.is_trained = True
        except FileNotFoundError:
            print("Model files not found. Training new model...")
            self.train_model()
    
    def predict_fraud(self, transaction_data):
        """Predict fraud probability for a transaction"""
        if not self.is_trained:
            self.load_model()
        
        # Convert to DataFrame if it's a dict
        if isinstance(transaction_data, dict):
            df = pd.DataFrame([transaction_data])
        else:
            df = transaction_data
        
        # Preprocess the data
        X, _, _ = self.preprocess_data(df)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        fraud_probability = self.model.predict_proba(X_scaled)[:, 1]
        
        return fraud_probability[0] if len(fraud_probability) == 1 else fraud_probability

# Initialize and train the model
def initialize_model():
    model = FraudDetectionModel()
    try:
        model.load_model()
        print("Model loaded successfully!")
    except:
        print("Training new model...")
        model.train_model()
        print("Model trained successfully!")
    return model