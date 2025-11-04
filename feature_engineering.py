"""
Feature Engineering for Flight Delay Prediction
This module processes raw flight data and creates engineered features for ML model training.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(parquet_path='2018-2024/flight_data.parquet'):
    """
    Load flight data and perform initial preprocessing.
    
    Args:
        parquet_path (str): Path to the parquet file containing flight data
        
    Returns:
        pd.DataFrame: Preprocessed flight data
    """
    print("Loading flight data...")
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df):,} flight records")
    
    # Basic data cleaning
    df = df.dropna(subset=['Origin', 'Dest', 'CRSDepTime', 'DepDelay'])
    df = df[df['CRSDepTime'].between(0, 2359)]  # Valid time format
    
    print(f"After cleaning: {len(df):,} records")
    return df

def create_time_features(df):
    """
    Create time-based features from flight date and scheduled departure time.
    
    Args:
        df (pd.DataFrame): Input dataframe with flight data
        
    Returns:
        pd.DataFrame: Dataframe with added time features
    """
    print("Creating time-based features...")
    
    # Convert FlightDate to datetime
    df['FlightDate'] = pd.to_datetime(df['FlightDate'], format='%Y%m%d')
    
    # Extract time components
    df['YEAR'] = df['FlightDate'].dt.year
    df['MONTH'] = df['FlightDate'].dt.month
    df['DAY_OF_WEEK'] = df['FlightDate'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['DAY_OF_MONTH'] = df['FlightDate'].dt.day
    df['QUARTER'] = df['FlightDate'].dt.quarter
    
    # Extract hour from CRSDepTime (format: HHMM)
    df['DEP_HOUR'] = (df['CRSDepTime'] // 100).clip(0, 23)
    df['DEP_MINUTE'] = (df['CRSDepTime'] % 100).clip(0, 59)
    
    # Create time period categories
    df['TIME_PERIOD'] = pd.cut(df['DEP_HOUR'], 
                              bins=[0, 6, 12, 18, 24], 
                              labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                              include_lowest=True)
    
    # Peak hour indicators
    df['IS_PEAK_HOUR'] = df['DEP_HOUR'].isin([7, 8, 9, 17, 18, 19]).astype(int)
    df['IS_WEEKEND'] = df['DAY_OF_WEEK'].isin([5, 6]).astype(int)  # Saturday, Sunday
    
    # Holiday indicators (simplified)
    df['IS_HOLIDAY_SEASON'] = df['MONTH'].isin([11, 12, 1]).astype(int)  # Nov, Dec, Jan
    df['IS_SUMMER'] = df['MONTH'].isin([6, 7, 8]).astype(int)  # Summer months
    
    return df

def create_airport_features(df):
    """
    Create airport-specific features including congestion metrics.
    
    Args:
        df (pd.DataFrame): Input dataframe with flight data
        
    Returns:
        pd.DataFrame: Dataframe with added airport features
    """
    print("Creating airport-based features...")
    
    # Airport congestion features
    # Daily departures by origin airport
    daily_departures = df.groupby(['Origin', 'FlightDate']).size().reset_index(name='DAILY_DEPARTURES')
    df = df.merge(daily_departures, on=['Origin', 'FlightDate'], how='left')
    
    # Hourly departures by origin airport and hour
    hourly_departures = df.groupby(['Origin', 'DEP_HOUR']).size().reset_index(name='HOURLY_DEPARTURES')
    df = df.merge(hourly_departures, on=['Origin', 'DEP_HOUR'], how='left')
    
    # Airport historical delay statistics
    airport_delay_stats = df.groupby('Origin')['DepDelay'].agg(['mean', 'std']).reset_index()
    airport_delay_stats.columns = ['Origin', 'AIRPORT_AVG_DELAY', 'AIRPORT_DELAY_STD']
    df = df.merge(airport_delay_stats, on='Origin', how='left')
    
    # Route-specific features
    route_delay_stats = df.groupby(['Origin', 'Dest'])['DepDelay'].agg(['mean', 'count']).reset_index()
    route_delay_stats.columns = ['Origin', 'Dest', 'ROUTE_AVG_DELAY', 'ROUTE_FREQUENCY']
    df = df.merge(route_delay_stats, on=['Origin', 'Dest'], how='left')
    
    # Airport size categories based on daily traffic
    airport_sizes = df.groupby('Origin')['DAILY_DEPARTURES'].mean().reset_index()
    airport_sizes['AIRPORT_SIZE'] = pd.cut(airport_sizes['DAILY_DEPARTURES'],
                                          bins=[0, 50, 200, 500, float('inf')],
                                          labels=['Small', 'Medium', 'Large', 'Hub'])
    df = df.merge(airport_sizes[['Origin', 'AIRPORT_SIZE']], on='Origin', how='left')
    
    return df

def create_airline_features(df):
    """
    Create airline-specific features.
    
    Args:
        df (pd.DataFrame): Input dataframe with flight data
        
    Returns:
        pd.DataFrame: Dataframe with added airline features
    """
    print("Creating airline-based features...")
    
    # Airline performance metrics
    airline_stats = df.groupby('Marketing_Airline_Network')['DepDelay'].agg(['mean', 'std', 'count']).reset_index()
    airline_stats.columns = ['Marketing_Airline_Network', 'AIRLINE_AVG_DELAY', 'AIRLINE_DELAY_STD', 'AIRLINE_FLIGHT_COUNT']
    df = df.merge(airline_stats, on='Marketing_Airline_Network', how='left')
    
    # Airline punctuality score (percentage of on-time flights)
    airline_punctuality = df.groupby('Marketing_Airline_Network').apply(
        lambda x: (x['DepDelay'] <= 15).mean()
    ).reset_index(name='AIRLINE_PUNCTUALITY')
    df = df.merge(airline_punctuality, on='Marketing_Airline_Network', how='left')
    
    return df

def create_weather_proxy_features(df):
    """
    Create proxy features for weather conditions using seasonal and time patterns.
    
    Args:
        df (pd.DataFrame): Input dataframe with flight data
        
    Returns:
        pd.DataFrame: Dataframe with added weather proxy features
    """
    print("Creating weather proxy features...")
    
    # Seasonal weather patterns
    df['WINTER_WEATHER_RISK'] = ((df['MONTH'].isin([12, 1, 2])) & 
                                 (df['Origin'].str.contains('|'.join(['ORD', 'DEN', 'MSP', 'DTW', 'BOS'])))).astype(int)
    
    df['SUMMER_STORM_RISK'] = ((df['MONTH'].isin([6, 7, 8])) & 
                               (df['Origin'].str.contains('|'.join(['ATL', 'DFW', 'IAH', 'MIA', 'TPA'])))).astype(int)
    
    # Time-based weather risk (afternoon storms more common)
    df['AFTERNOON_STORM_RISK'] = ((df['DEP_HOUR'].between(14, 18)) & 
                                  (df['MONTH'].isin([5, 6, 7, 8, 9]))).astype(int)
    
    return df

def create_target_variable(df, delay_threshold=15):
    """
    Create the target variable for delay prediction.
    
    Args:
        df (pd.DataFrame): Input dataframe with flight data
        delay_threshold (int): Minutes threshold for considering a flight delayed
        
    Returns:
        pd.DataFrame: Dataframe with target variable
    """
    print(f"Creating target variable (delay threshold: {delay_threshold} minutes)...")
    
    # Binary target: 1 if delayed by more than threshold minutes, 0 otherwise
    df['IS_DELAYED'] = (df['DepDelay'] > delay_threshold).astype(int)
    
    # Additional delay severity categories
    df['DELAY_CATEGORY'] = pd.cut(df['DepDelay'],
                                 bins=[-float('inf'), 0, 15, 60, 180, float('inf')],
                                 labels=['Early', 'OnTime', 'Minor_Delay', 'Major_Delay', 'Severe_Delay'])
    
    print(f"Delay distribution:")
    print(df['IS_DELAYED'].value_counts(normalize=True))
    
    return df

def encode_categorical_features(df):
    """
    Encode categorical features for machine learning.
    
    Args:
        df (pd.DataFrame): Input dataframe with categorical features
        
    Returns:
        pd.DataFrame: Dataframe with encoded features
    """
    print("Encoding categorical features...")
    
    # One-hot encode categorical features with limited cardinality
    categorical_features = ['TIME_PERIOD', 'AIRPORT_SIZE', 'DAY_OF_WEEK']
    
    for feature in categorical_features:
        if feature in df.columns:
            dummies = pd.get_dummies(df[feature], prefix=feature, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
    
    # Label encode high-cardinality categorical features
    from sklearn.preprocessing import LabelEncoder
    
    high_card_features = ['Origin', 'Dest', 'Marketing_Airline_Network']
    label_encoders = {}
    
    for feature in high_card_features:
        if feature in df.columns:
            le = LabelEncoder()
            df[f'{feature}_ENCODED'] = le.fit_transform(df[feature].astype(str))
            label_encoders[feature] = le
    
    return df, label_encoders

def create_interaction_features(df):
    """
    Create interaction features between important variables.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with interaction features
    """
    print("Creating interaction features...")
    
    # Airport-Hour interaction (captures airport-specific peak patterns)
    if 'Origin_ENCODED' in df.columns and 'DEP_HOUR' in df.columns:
        df['AIRPORT_HOUR_INTERACTION'] = df['Origin_ENCODED'] * df['DEP_HOUR']
    
    # Airline-Airport interaction
    if 'Marketing_Airline_Network_ENCODED' in df.columns and 'Origin_ENCODED' in df.columns:
        df['AIRLINE_AIRPORT_INTERACTION'] = df['Marketing_Airline_Network_ENCODED'] * df['Origin_ENCODED']
    
    # Congestion-Time interaction
    if 'HOURLY_DEPARTURES' in df.columns and 'IS_PEAK_HOUR' in df.columns:
        df['CONGESTION_PEAK_INTERACTION'] = df['HOURLY_DEPARTURES'] * df['IS_PEAK_HOUR']
    
    return df

def main():
    """
    Main function to run the complete feature engineering pipeline.
    """
    print("Starting Feature Engineering Pipeline...")
    print("=" * 50)
    
    # Load and preprocess data
    df = load_and_preprocess_data()
    
    # Apply feature engineering steps
    df = create_time_features(df)
    df = create_airport_features(df)
    df = create_airline_features(df)
    df = create_weather_proxy_features(df)
    df = create_target_variable(df)
    df, label_encoders = encode_categorical_features(df)
    df = create_interaction_features(df)
    
    # Fill missing values
    print("Handling missing values...")
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
    
    # Final data summary
    print("\n" + "=" * 50)
    print("Feature Engineering Complete!")
    print(f"Final dataset shape: {df.shape}")
    print(f"Number of features: {len(df.columns)}")
    print(f"Target variable distribution:")
    print(df['IS_DELAYED'].value_counts(normalize=True))
    
    # Save the engineered dataset
    output_path = 'engineered_flight_data.parquet'
    df.to_parquet(output_path, index=False)
    print(f"Engineered dataset saved to: {output_path}")
    
    return df

if __name__ == "__main__":
    engineered_df = main()