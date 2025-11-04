"""
Flight Delay Prediction Model
Implements Random Forest and XGBoost models for predicting flight delays.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

class DelayPredictor:
    """Flight delay prediction using ensemble methods."""
    
    def __init__(self):
        self.models = {}
        self.encoders = {}
        self.feature_names = []
        self.model_dir = Path('models')
        self.model_dir.mkdir(exist_ok=True)
    
    def load_data(self, data_path='data/processed/flights_cleaned.parquet'):
        """Load proces