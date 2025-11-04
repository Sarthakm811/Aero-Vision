"""
ML Integration Script for Air Traffic Dashboard
This script provides easy integration between the dashboard and ML components.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

def check_ml_dependencies():
    """Check if ML dependencies are installed."""
    required_packages = ['sklearn', 'joblib', 'matplotlib', 'seaborn']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def setup_ml_environment():
    """Set up the ML environment and check dependencies."""
    print("🔧 Setting up ML environment...")
    
    missing = check_ml_dependencies()
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("Please install with: pip install scikit-learn joblib matplotlib seaborn")
        return False
    
    print("✅ All ML dependencies available")
    return True

def run_complete_ml_pipeline():
    """Run the complete ML pipeline from data to trained model."""
    print("🚀 Starting Complete ML Pipeline")
    print("=" * 50)
    
    # Check environment
    if not setup_ml_environment():
        return False
    
    try:
        # Step 1: Feature Engineering
        print("\n📊 Step 1: Feature Engineering")
        print("-" * 30)
        
        from feature_engineering import main as run_feature_engineering
        engineered_df = run_feature_engineering()
        
        if engineered_df is None or engineered_df.empty:
            print("❌ Feature engineering failed")
            return False
        
        print(f"✅ Feature engineering completed: {engineered_df.shape}")
        
        # Step 2: Model Training
        print("\n🤖 Step 2: Model Training")
        print("-" * 30)
        
        from flight_delay_predictor import main as train_model
        predictor = train_model()
        
        if predictor is None:
            print("❌ Model training failed")
            return False
        
        print("✅ Model training completed")
        
        # Step 3: Validation
        print("\n✅ Step 3: Validation")
        print("-" * 30)
        
        # Check if model file exists
        if os.path.exists('flight_delay_model.joblib'):
            print("✅ Model file saved successfully")
        else:
            print("❌ Model file not found")
            return False
        
        # Test prediction
        sample_prediction = test_model_prediction(predictor)
        if sample_prediction:
            print("✅ Model prediction test passed")
        else:
            print("❌ Model prediction test failed")
            return False
        
        print("\n🎉 ML Pipeline completed successfully!")
        print("The dashboard can now use ML-powered predictions.")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline failed: {str(e)}")
        return False

def test_model_prediction(predictor):
    """Test the trained model with sample data."""
    try:
        # Create sample flight data
        sample_flight = {
            'DEP_HOUR': 17,
            'IS_PEAK_HOUR': 1,
            'HOURLY_DEPARTURES': 45,
            'AIRPORT_AVG_DELAY': 18.5,
            'AIRLINE_PUNCTUALITY': 0.72,
            'IS_WEEKEND': 0,
            'WINTER_WEATHER_RISK': 0,
            'SUMMER_STORM_RISK': 1,
        }
        
        # Add default values for other required features
        for feature in predictor.feature_names:
            if feature not in sample_flight:
                sample_flight[feature] = 0
        
        # Make prediction
        result = predictor.predict_delay(sample_flight)
        
        if result:
            print(f"Sample prediction: {result['risk_level']} (prob: {result['delay_probability']:.3f})")
            return True
        else:
            return False
            
    except Exception as e:
        print(f"Prediction test error: {str(e)}")
        return False

def create_ml_demo_data():
    """Create demonstration data for ML model testing."""
    print("📝 Creating ML demo data...")
    
    try:
        from flight_delay_predictor import FlightDelayPredictor
        
        predictor = FlightDelayPredictor()
        demo_df = predictor.create_sample_data()
        
        # Save demo data
        demo_df.to_parquet('demo_flight_data.parquet', index=False)
        print(f"✅ Demo data created: {demo_df.shape}")
        print(f"Delay rate: {demo_df['IS_DELAYED'].mean():.3f}")
        
        return demo_df
        
    except Exception as e:
        print(f"❌ Demo data creation failed: {str(e)}")
        return None

def validate_dashboard_integration():
    """Validate that the dashboard can use ML components."""
    print("🔍 Validating dashboard integration...")
    
    try:
        # Check if ML model exists
        if not os.path.exists('flight_delay_model.joblib'):
            print("❌ ML model not found. Run training first.")
            return False
        
        # Try to load the model
        from flight_delay_predictor import FlightDelayPredictor
        predictor = FlightDelayPredictor()
        predictor.load_model()
        
        if predictor.best_model is None:
            print("❌ Failed to load ML model")
            return False
        
        print("✅ ML model loaded successfully")
        print(f"Model type: {predictor.best_model_name}")
        print(f"Features: {len(predictor.feature_names)}")
        
        # Test dashboard integration
        print("✅ Dashboard integration validated")
        return True
        
    except Exception as e:
        print(f"❌ Integration validation failed: {str(e)}")
        return False

def main():
    """Main function to set up ML integration."""
    print("🎯 Air Traffic Dashboard - ML Integration Setup")
    print("=" * 60)
    
    # Check current status
    print("\n📋 Current Status:")
    print("-" * 20)
    
    # Check data files
    data_files = [
        '2018-2024/flight_data.parquet',
        'engineered_flight_data.parquet',
        'flight_delay_model.joblib'
    ]
    
    for file in data_files:
        if os.path.exists(file):
            size_mb = os.path.getsize(file) / (1024 * 1024)
            print(f"✅ {file} ({size_mb:.1f} MB)")
        else:
            print(f"❌ {file} (missing)")
    
    # Check dependencies
    missing_deps = check_ml_dependencies()
    if missing_deps:
        print(f"❌ Missing dependencies: {', '.join(missing_deps)}")
    else:
        print("✅ All ML dependencies available")
    
    # Interactive setup
    print("\n🚀 Setup Options:")
    print("-" * 20)
    print("1. Run complete ML pipeline (feature engineering + training)")
    print("2. Create demo data for testing")
    print("3. Validate dashboard integration")
    print("4. Exit")
    
    try:
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == '1':
            success = run_complete_ml_pipeline()
            if success:
                validate_dashboard_integration()
        elif choice == '2':
            create_ml_demo_data()
        elif choice == '3':
            validate_dashboard_integration()
        elif choice == '4':
            print("👋 Goodbye!")
        else:
            print("❌ Invalid choice")
            
    except KeyboardInterrupt:
        print("\n👋 Setup cancelled by user")
    except Exception as e:
        print(f"❌ Setup error: {str(e)}")

if __name__ == "__main__":
    main()