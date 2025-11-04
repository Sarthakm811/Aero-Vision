"""
Flight Delay Prediction Model
This script trains a machine learning model to predict flight delays using engineered features.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, 
                           classification_report, confusion_matrix, roc_auc_score)
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FlightDelayPredictor:
    """
    A comprehensive flight delay prediction model with multiple algorithms and evaluation metrics.
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.best_model = None
        self.best_model_name = None
        
    def load_data(self, data_path='engineered_flight_data.parquet'):
        """
        Load the engineered flight data.
        
        Args:
            data_path (str): Path to the engineered data file
            
        Returns:
            pd.DataFrame: Loaded dataframe
        """
        try:
            print(f"Loading data from {data_path}...")
            df = pd.read_parquet(data_path)
            print(f"Data loaded successfully: {df.shape}")
            return df
        except FileNotFoundError:
            print(f"Data file not found. Creating sample data...")
            return self.create_sample_data()
    
    def create_sample_data(self):
        """
        Create sample data that mirrors the structure of engineered features.
        This is used when the actual engineered data is not available.
        
        Returns:
            pd.DataFrame: Sample dataframe with engineered features
        """
        print("Creating sample engineered dataset...")
        
        np.random.seed(42)
        n_samples = 10000
        
        # Create sample data with engineered features
        data = {
            # Time-based features
            'YEAR': np.random.choice([2018, 2019, 2020, 2021, 2022, 2023, 2024], n_samples),
            'MONTH': np.random.randint(1, 13, n_samples),
            'DAY_OF_WEEK': np.random.randint(0, 7, n_samples),
            'DAY_OF_MONTH': np.random.randint(1, 32, n_samples),
            'QUARTER': np.random.randint(1, 5, n_samples),
            'DEP_HOUR': np.random.randint(5, 24, n_samples),
            'DEP_MINUTE': np.random.randint(0, 60, n_samples),
            'IS_PEAK_HOUR': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'IS_WEEKEND': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'IS_HOLIDAY_SEASON': np.random.choice([0, 1], n_samples, p=[0.75, 0.25]),
            'IS_SUMMER': np.random.choice([0, 1], n_samples, p=[0.75, 0.25]),
            
            # Airport-based features
            'DAILY_DEPARTURES': np.random.randint(50, 800, n_samples),
            'HOURLY_DEPARTURES': np.random.randint(1, 50, n_samples),
            'AIRPORT_AVG_DELAY': np.random.normal(12, 8, n_samples),
            'AIRPORT_DELAY_STD': np.random.normal(15, 5, n_samples),
            'ROUTE_AVG_DELAY': np.random.normal(10, 6, n_samples),
            'ROUTE_FREQUENCY': np.random.randint(1, 100, n_samples),
            
            # Airline-based features
            'AIRLINE_AVG_DELAY': np.random.normal(11, 7, n_samples),
            'AIRLINE_DELAY_STD': np.random.normal(14, 4, n_samples),
            'AIRLINE_FLIGHT_COUNT': np.random.randint(100, 10000, n_samples),
            'AIRLINE_PUNCTUALITY': np.random.uniform(0.6, 0.9, n_samples),
            
            # Weather proxy features
            'WINTER_WEATHER_RISK': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
            'SUMMER_STORM_RISK': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'AFTERNOON_STORM_RISK': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
            
            # Encoded categorical features
            'Origin_ENCODED': np.random.randint(0, 300, n_samples),
            'Dest_ENCODED': np.random.randint(0, 300, n_samples),
            'Marketing_Airline_Network_ENCODED': np.random.randint(0, 20, n_samples),
            
            # One-hot encoded features
            'TIME_PERIOD_Morning': np.random.choice([0, 1], n_samples, p=[0.75, 0.25]),
            'TIME_PERIOD_Afternoon': np.random.choice([0, 1], n_samples, p=[0.75, 0.25]),
            'TIME_PERIOD_Evening': np.random.choice([0, 1], n_samples, p=[0.75, 0.25]),
            'AIRPORT_SIZE_Medium': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'AIRPORT_SIZE_Large': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'AIRPORT_SIZE_Hub': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
            
            # Interaction features
            'AIRPORT_HOUR_INTERACTION': np.random.randint(0, 7200, n_samples),
            'AIRLINE_AIRPORT_INTERACTION': np.random.randint(0, 6000, n_samples),
            'CONGESTION_PEAK_INTERACTION': np.random.randint(0, 50, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Create realistic target variable based on features
        delay_probability = (
            0.1 +  # Base probability
            0.15 * df['IS_PEAK_HOUR'] +
            0.1 * df['IS_WEEKEND'] +
            0.05 * (df['HOURLY_DEPARTURES'] > 30) +
            0.1 * (df['AIRPORT_AVG_DELAY'] > 15) +
            0.08 * df['WINTER_WEATHER_RISK'] +
            0.06 * df['SUMMER_STORM_RISK'] +
            0.04 * df['AFTERNOON_STORM_RISK'] +
            0.05 * (df['AIRLINE_PUNCTUALITY'] < 0.75)
        )
        
        # Add some noise and ensure probabilities are valid
        delay_probability = np.clip(delay_probability + np.random.normal(0, 0.05, n_samples), 0, 1)
        
        # Generate binary target
        df['IS_DELAYED'] = np.random.binomial(1, delay_probability, n_samples)
        
        print(f"Sample data created: {df.shape}")
        print(f"Delay rate: {df['IS_DELAYED'].mean():.3f}")
        
        return df
    
    def prepare_features(self, df):
        """
        Prepare features for model training.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            tuple: (X, y) features and target
        """
        print("Preparing features for model training...")
        
        # Define target variable
        target_col = 'IS_DELAYED'
        if target_col not in df.columns:
            raise ValueError(f"Target variable '{target_col}' not found in dataframe")
        
        # Define feature columns (exclude target and non-predictive columns)
        exclude_cols = [
            'IS_DELAYED', 'DepDelay', 'DELAY_CATEGORY', 'FlightDate',
            'FLIGHT_NUM', 'Origin', 'Dest', 'Marketing_Airline_Network',
            'TIME_PERIOD', 'AIRPORT_SIZE'  # Original categorical columns (we use encoded versions)
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Handle missing values
        X = df[feature_cols].fillna(df[feature_cols].median())
        y = df[target_col]
        
        self.feature_names = feature_cols
        
        print(f"Features prepared: {X.shape}")
        print(f"Target distribution: {y.value_counts(normalize=True).to_dict()}")
        
        return X, y
    
    def train_models(self, X, y, test_size=0.2, random_state=42):
        """
        Train multiple machine learning models and compare their performance.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target variable
            test_size (float): Proportion of data for testing
            random_state (int): Random seed for reproducibility
        """
        print("Splitting data into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        # Store test data for evaluation
        self.X_test = X_test
        self.y_test = y_test
        
        # Define models to train
        models_config = {
            'GradientBoosting': {
                'model': GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=random_state
                ),
                'scale': False
            },
            'RandomForest': {
                'model': RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=random_state
                ),
                'scale': False
            },
            'LogisticRegression': {
                'model': LogisticRegression(
                    random_state=random_state,
                    max_iter=1000
                ),
                'scale': True
            }
        }
        
        print("\nTraining models...")
        print("=" * 50)
        
        model_scores = {}
        
        for name, config in models_config.items():
            print(f"\nTraining {name}...")
            
            # Prepare data (scale if needed)
            if config['scale']:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                self.scalers[name] = scaler
            else:
                X_train_scaled = X_train
                X_test_scaled = X_test
            
            # Train model
            model = config['model']
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            # Store results
            model_scores[name] = {
                'accuracy': accuracy,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'auc': auc
            }
            
            self.models[name] = {
                'model': model,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"AUC: {auc:.4f}")
        
        # Determine best model based on F1 score (good for imbalanced data)
        best_model_name = max(model_scores.keys(), key=lambda x: model_scores[x]['f1'])
        self.best_model_name = best_model_name
        self.best_model = self.models[best_model_name]['model']
        
        print(f"\n🏆 Best Model: {best_model_name} (F1 Score: {model_scores[best_model_name]['f1']:.4f})")
        
        return model_scores
    
    def evaluate_best_model(self):
        """
        Provide detailed evaluation of the best performing model.
        """
        if self.best_model is None:
            print("No model has been trained yet!")
            return
        
        print(f"\n📊 Detailed Evaluation of {self.best_model_name}")
        print("=" * 60)
        
        y_pred = self.models[self.best_model_name]['predictions']
        y_pred_proba = self.models[self.best_model_name]['probabilities']
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        
        # Confusion matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(self.y_test, y_pred)
        print(cm)
        
        # Feature importance (if available)
        if hasattr(self.best_model, 'feature_importances_'):
            print("\nTop 10 Most Important Features:")
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(feature_importance.head(10).to_string(index=False))
    
    def hyperparameter_tuning(self, X, y):
        """
        Perform hyperparameter tuning for the Gradient Boosting model.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target variable
        """
        print("\n🔧 Performing hyperparameter tuning for Gradient Boosting...")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [4, 6, 8],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        # Perform grid search
        gb_model = GradientBoostingClassifier(random_state=42)
        grid_search = GridSearchCV(
            gb_model, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1
        )
        
        # Use a subset of data for faster tuning
        sample_size = min(5000, len(X))
        X_sample = X.sample(n=sample_size, random_state=42)
        y_sample = y.loc[X_sample.index]
        
        grid_search.fit(X_sample, y_sample)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")
        
        # Update the best model with tuned parameters
        self.tuned_model = grid_search.best_estimator_
        
        return grid_search.best_params_
    
    def save_model(self, model_path='flight_delay_model.joblib'):
        """
        Save the trained model to disk.
        
        Args:
            model_path (str): Path to save the model
        """
        if self.best_model is None:
            print("No model to save!")
            return
        
        model_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'feature_names': self.feature_names,
            'scalers': self.scalers
        }
        
        joblib.dump(model_data, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path='flight_delay_model.joblib'):
        """
        Load a trained model from disk.
        
        Args:
            model_path (str): Path to the saved model
        """
        try:
            model_data = joblib.load(model_path)
            self.best_model = model_data['model']
            self.best_model_name = model_data['model_name']
            self.feature_names = model_data['feature_names']
            self.scalers = model_data.get('scalers', {})
            print(f"Model loaded from {model_path}")
        except FileNotFoundError:
            print(f"Model file {model_path} not found!")
    
    def predict_delay(self, flight_features):
        """
        Predict delay probability for new flight data.
        
        Args:
            flight_features (dict or pd.DataFrame): Flight features
            
        Returns:
            dict: Prediction results
        """
        if self.best_model is None:
            print("No model available for prediction!")
            return None
        
        # Convert to DataFrame if needed
        if isinstance(flight_features, dict):
            flight_features = pd.DataFrame([flight_features])
        
        # Ensure all required features are present
        missing_features = set(self.feature_names) - set(flight_features.columns)
        if missing_features:
            print(f"Missing features: {missing_features}")
            return None
        
        # Select and order features
        X = flight_features[self.feature_names]
        
        # Scale if needed
        if self.best_model_name in self.scalers:
            X = self.scalers[self.best_model_name].transform(X)
        
        # Make prediction
        prediction = self.best_model.predict(X)[0]
        probability = self.best_model.predict_proba(X)[0, 1]
        
        return {
            'is_delayed': bool(prediction),
            'delay_probability': float(probability),
            'risk_level': 'High' if probability > 0.7 else 'Medium' if probability > 0.3 else 'Low'
        }

def main():
    """
    Main function to run the complete machine learning pipeline.
    """
    print("🚀 Flight Delay Prediction Model Training")
    print("=" * 60)
    
    # Initialize predictor
    predictor = FlightDelayPredictor()
    
    # Load data
    df = predictor.load_data()
    
    # Prepare features
    X, y = predictor.prepare_features(df)
    
    # Train models
    model_scores = predictor.train_models(X, y)
    
    # Evaluate best model
    predictor.evaluate_best_model()
    
    # Optional: Hyperparameter tuning (uncomment to run)
    # predictor.hyperparameter_tuning(X, y)
    
    # Save the model
    predictor.save_model()
    
    # Example prediction
    print("\n🔮 Example Prediction:")
    print("-" * 30)
    
    sample_flight = {
        'DEP_HOUR': 17,  # 5 PM departure
        'IS_PEAK_HOUR': 1,
        'HOURLY_DEPARTURES': 45,
        'AIRPORT_AVG_DELAY': 18.5,
        'AIRLINE_PUNCTUALITY': 0.72,
        'IS_WEEKEND': 0,
        'WINTER_WEATHER_RISK': 0,
        'SUMMER_STORM_RISK': 1,
        # Add other required features with default values
        **{col: 0 for col in predictor.feature_names if col not in [
            'DEP_HOUR', 'IS_PEAK_HOUR', 'HOURLY_DEPARTURES', 
            'AIRPORT_AVG_DELAY', 'AIRLINE_PUNCTUALITY', 'IS_WEEKEND',
            'WINTER_WEATHER_RISK', 'SUMMER_STORM_RISK'
        ]}
    }
    
    result = predictor.predict_delay(sample_flight)
    if result:
        print(f"Delay Prediction: {'DELAYED' if result['is_delayed'] else 'ON TIME'}")
        print(f"Delay Probability: {result['delay_probability']:.3f}")
        print(f"Risk Level: {result['risk_level']}")
    
    print("\n✅ Model training completed successfully!")
    
    return predictor

if __name__ == "__main__":
    predictor = main()