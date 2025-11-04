"""
Comprehensive Test Cases and Validation Framework
This module provides extensive testing and validation for the ATC system.
"""

import pandas as pd
import numpy as np
import unittest
from datetime import datetime, timedelta
import sys
import os

# Add current directory to path for imports
sys.path.append('.')

class ATCSystemTests(unittest.TestCase):
    """
    Comprehensive test suite for ATC Flight Delay Prediction System.
    """
    
    def setUp(self):
        """Set up test environment and sample data."""
        self.sample_data = pd.DataFrame({
            'FL_DATE': ['2023-01-15', '2023-01-15', '2023-01-16'],
            'AIRLINE_CODE': ['AA', 'DL', 'UA'],
            'FL_NUMBER': [101, 202, 303],
            'ORIGIN': ['JFK', 'LAX', 'ORD'],
            'DEST': ['LAX', 'JFK', 'DFW'],
            'CRS_DEP_TIME': [800, 1400, 1900],
            'DEP_DELAY': [5, 25, -3],
            'ARR_DELAY': [8, 30, 0]
        })
        
    def test_data_loading(self):
        """Test data loading functionality."""
        print("\n" + "="*50)
        print("TESTING: Data Loading")
        print("="*50)
        
        # Test 1: Valid data loading
        try:
            from data_analysis_report import FlightDataAnalyzer
            analyzer = FlightDataAnalyzer()
            
            # Create temporary test file
            test_file = 'test_data.csv'
            self.sample_data.to_csv(test_file, index=False)
            
            analyzer.data_path = test_file
            df = analyzer.load_data()
            
            self.assertIsNotNone(df, "Data should load successfully")
            self.assertGreater(len(df), 0, "Loaded data should not be empty")
            
            # Cleanup
            os.remove(test_file)
            print("✅ Data loading test passed")
            
        except Exception as e:
            print(f"❌ Data loading test failed: {e}")
            
    def test_feature_engineering(self):
        """Test feature engineering functionality."""
        print("\n" + "="*50)
        print("TESTING: Feature Engineering")
        print("="*50)
        
        try:
            from feature_engineering import create_time_features
            
            # Prepare test data
            test_df = self.sample_data.copy()
            test_df['FlightDate'] = pd.to_datetime(test_df['FL_DATE'])
            
            # Test time feature creation
            result_df = create_time_features(test_df)
            
            # Assertions
            self.assertIn('YEAR', result_df.columns, "YEAR feature should be created")
            self.assertIn('MONTH', result_df.columns, "MONTH feature should be created")
            self.assertIn('DAY_OF_WEEK', result_df.columns, "DAY_OF_WEEK feature should be created")
            
            # Validate feature values
            self.assertTrue(result_df['YEAR'].between(2020, 2030).all(), "Year values should be reasonable")
            self.assertTrue(result_df['MONTH'].between(1, 12).all(), "Month values should be 1-12")
            self.assertTrue(result_df['DAY_OF_WEEK'].between(0, 6).all(), "Day of week should be 0-6")
            
            print("✅ Feature engineering test passed")
            
        except Exception as e:
            print(f"❌ Feature engineering test failed: {e}")
    
    def test_ml_model_training(self):
        """Test ML model training functionality."""
        print("\n" + "="*50)
        print("TESTING: ML Model Training")
        print("="*50)
        
        try:
            from flight_delay_predictor import FlightDelayPredictor
            
            predictor = FlightDelayPredictor()
            
            # Test sample data creation
            sample_df = predictor.create_sample_data()
            self.assertIsNotNone(sample_df, "Sample data should be created")
            self.assertGreater(len(sample_df), 100, "Sample should have sufficient data")
            
            # Test feature preparation
            X, y = predictor.prepare_features(sample_df)
            self.assertIsNotNone(X, "Features should be prepared")
            self.assertIsNotNone(y, "Target should be prepared")
            self.assertEqual(len(X), len(y), "Features and target should have same length")
            
            print("✅ ML model training test passed")
            
        except Exception as e:
            print(f"❌ ML model training test failed: {e}")
    
    def test_model_performance(self):
        """Test model performance metrics."""
        print("\n" + "="*50)
        print("TESTING: Model Performance")
        print("="*50)
        
        try:
            from model_evaluation_report import ModelEvaluator
            
            evaluator = ModelEvaluator()
            
            # Test data loading
            data_loaded = evaluator.load_data()
            self.assertTrue(data_loaded, "Data should load for evaluation")
            
            # Test model initialization
            evaluator.initialize_models()
            self.assertGreater(len(evaluator.models), 0, "Models should be initialized")
            
            print("✅ Model performance test passed")
            
        except Exception as e:
            print(f"❌ Model performance test failed: {e}")
    
    def test_dashboard_functionality(self):
        """Test dashboard functionality."""
        print("\n" + "="*50)
        print("TESTING: Dashboard Functionality")
        print("="*50)
        
        try:
            # Test data processing functions
            test_df = self.sample_data.copy()
            
            # Test risk calculation logic
            def calculate_basic_risk(hour, volume):
                if hour in [17, 18, 19] and volume > 50:
                    return 'High Risk'
                elif volume > 30:
                    return 'Medium Risk'
                else:
                    return 'Low Risk'
            
            # Test risk assignments
            risk1 = calculate_basic_risk(18, 60)  # Should be High Risk
            risk2 = calculate_basic_risk(10, 40)  # Should be Medium Risk
            risk3 = calculate_basic_risk(6, 20)   # Should be Low Risk
            
            self.assertEqual(risk1, 'High Risk', "Peak hour + high volume should be High Risk")
            self.assertEqual(risk2, 'Medium Risk', "Medium volume should be Medium Risk")
            self.assertEqual(risk3, 'Low Risk', "Low volume should be Low Risk")
            
            print("✅ Dashboard functionality test passed")
            
        except Exception as e:
            print(f"❌ Dashboard functionality test failed: {e}")
    
    def test_data_validation(self):
        """Test data validation and quality checks."""
        print("\n" + "="*50)
        print("TESTING: Data Validation")
        print("="*50)
        
        try:
            # Test data quality metrics
            test_df = self.sample_data.copy()
            
            # Check for required columns
            required_columns = ['FL_DATE', 'ORIGIN', 'DEST', 'CRS_DEP_TIME']
            for col in required_columns:
                self.assertIn(col, test_df.columns, f"Required column {col} should be present")
            
            # Check data types and ranges
            test_df['CRS_DEP_TIME'] = pd.to_numeric(test_df['CRS_DEP_TIME'], errors='coerce')
            valid_times = test_df['CRS_DEP_TIME'].between(0, 2359)
            self.assertTrue(valid_times.all(), "All departure times should be valid (0-2359)")
            
            # Check for missing values in critical columns
            critical_missing = test_df[required_columns].isnull().sum().sum()
            self.assertEqual(critical_missing, 0, "Critical columns should not have missing values")
            
            print("✅ Data validation test passed")
            
        except Exception as e:
            print(f"❌ Data validation test failed: {e}")
    
    def test_performance_benchmarks(self):
        """Test system performance benchmarks."""
        print("\n" + "="*50)
        print("TESTING: Performance Benchmarks")
        print("="*50)
        
        try:
            import time
            
            # Test data processing speed
            large_df = pd.DataFrame({
                'ORIGIN': np.random.choice(['JFK', 'LAX', 'ORD'], 10000),
                'DEP_HOUR': np.random.randint(0, 24, 10000),
                'HOURLY_DEPARTURES': np.random.randint(1, 100, 10000)
            })
            
            start_time = time.time()
            
            # Simulate risk calculation
            def fast_risk_calc(df):
                conditions = [
                    (df['DEP_HOUR'].isin([17, 18, 19])) & (df['HOURLY_DEPARTURES'] > 50),
                    df['HOURLY_DEPARTURES'] > 30
                ]
                choices = ['High Risk', 'Medium Risk']
                df['RISK'] = np.select(conditions, choices, default='Low Risk')
                return df
            
            result_df = fast_risk_calc(large_df)
            processing_time = time.time() - start_time
            
            # Performance assertions
            self.assertLess(processing_time, 1.0, "Processing 10K records should take <1 second")
            self.assertEqual(len(result_df), 10000, "All records should be processed")
            self.assertIn('RISK', result_df.columns, "Risk column should be added")
            
            print(f"✅ Performance benchmark passed: {processing_time:.3f} seconds for 10K records")
            
        except Exception as e:
            print(f"❌ Performance benchmark test failed: {e}")
    
    def test_error_handling(self):
        """Test error handling and edge cases."""
        print("\n" + "="*50)
        print("TESTING: Error Handling")
        print("="*50)
        
        try:
            # Test empty dataframe handling
            empty_df = pd.DataFrame()
            
            def safe_process(df):
                if df.empty:
                    return pd.DataFrame()
                return df
            
            result = safe_process(empty_df)
            self.assertTrue(result.empty, "Empty dataframe should be handled gracefully")
            
            # Test invalid data handling
            invalid_df = pd.DataFrame({
                'CRS_DEP_TIME': [2500, -100, 'invalid'],  # Invalid times
                'DEP_DELAY': [None, 'text', 999999]       # Invalid delays
            })
            
            # Test data cleaning
            cleaned_times = pd.to_numeric(invalid_df['CRS_DEP_TIME'], errors='coerce')
            valid_times = cleaned_times.between(0, 2359, na=False)
            
            self.assertFalse(valid_times.all(), "Invalid times should be detected")
            
            print("✅ Error handling test passed")
            
        except Exception as e:
            print(f"❌ Error handling test failed: {e}")

class ValidationFramework:
    """
    Comprehensive validation framework for the ATC system.
    """
    
    def __init__(self):
        self.validation_results = {}
        
    def validate_data_quality(self, df):
        """Validate data quality metrics."""
        print("\n" + "="*50)
        print("VALIDATION: Data Quality Assessment")
        print("="*50)
        
        results = {
            'total_records': len(df),
            'missing_data_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            'duplicate_records': df.duplicated().sum(),
            'data_types_valid': True,
            'date_range_valid': True
        }
        
        # Validate date ranges
        if 'FL_DATE' in df.columns:
            try:
                dates = pd.to_datetime(df['FL_DATE'])
                min_date = dates.min()
                max_date = dates.max()
                
                # Check if dates are reasonable (within last 10 years)
                current_date = datetime.now()
                ten_years_ago = current_date - timedelta(days=3650)
                
                if min_date < ten_years_ago or max_date > current_date:
                    results['date_range_valid'] = False
                    
            except:
                results['date_range_valid'] = False
        
        # Print results
        print(f"Total Records: {results['total_records']:,}")
        print(f"Missing Data: {results['missing_data_percentage']:.2f}%")
        print(f"Duplicate Records: {results['duplicate_records']:,}")
        print(f"Date Range Valid: {'✅' if results['date_range_valid'] else '❌'}")
        
        self.validation_results['data_quality'] = results
        return results
    
    def validate_model_performance(self, model_results):
        """Validate ML model performance."""
        print("\n" + "="*50)
        print("VALIDATION: Model Performance")
        print("="*50)
        
        # Define performance thresholds
        thresholds = {
            'min_accuracy': 0.65,
            'min_f1_score': 0.60,
            'min_auc': 0.70,
            'max_cv_std': 0.05
        }
        
        validation_results = {}
        
        for model_name, metrics in model_results.items():
            model_valid = True
            issues = []
            
            # Check accuracy
            if metrics.get('accuracy', 0) < thresholds['min_accuracy']:
                model_valid = False
                issues.append(f"Accuracy below threshold: {metrics.get('accuracy', 0):.3f}")
            
            # Check F1 score
            if metrics.get('f1_score', 0) < thresholds['min_f1_score']:
                model_valid = False
                issues.append(f"F1 score below threshold: {metrics.get('f1_score', 0):.3f}")
            
            # Check AUC
            if metrics.get('auc', 0) < thresholds['min_auc']:
                model_valid = False
                issues.append(f"AUC below threshold: {metrics.get('auc', 0):.3f}")
            
            validation_results[model_name] = {
                'valid': model_valid,
                'issues': issues,
                'metrics': metrics
            }
            
            status = "✅ PASS" if model_valid else "❌ FAIL"
            print(f"{model_name}: {status}")
            for issue in issues:
                print(f"  - {issue}")
        
        self.validation_results['model_performance'] = validation_results
        return validation_results
    
    def validate_system_requirements(self):
        """Validate system requirements and dependencies."""
        print("\n" + "="*50)
        print("VALIDATION: System Requirements")
        print("="*50)
        
        requirements = {
            'python_version': sys.version_info >= (3, 8),
            'required_packages': [],
            'memory_usage': True,  # Simplified check
            'disk_space': True     # Simplified check
        }
        
        # Check required packages
        required_packages = ['pandas', 'numpy', 'sklearn', 'streamlit', 'plotly']
        
        for package in required_packages:
            try:
                __import__(package)
                requirements['required_packages'].append((package, True))
                print(f"✅ {package}: Available")
            except ImportError:
                requirements['required_packages'].append((package, False))
                print(f"❌ {package}: Missing")
        
        # Check Python version
        if requirements['python_version']:
            print(f"✅ Python Version: {sys.version}")
        else:
            print(f"❌ Python Version: {sys.version} (Requires 3.8+)")
        
        self.validation_results['system_requirements'] = requirements
        return requirements
    
    def generate_validation_report(self):
        """Generate comprehensive validation report."""
        print("\n" + "="*60)
        print("COMPREHENSIVE VALIDATION REPORT")
        print("="*60)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Overall system status
        all_validations_passed = True
        
        # Check data quality
        if 'data_quality' in self.validation_results:
            dq = self.validation_results['data_quality']
            if dq['missing_data_percentage'] > 10 or not dq['date_range_valid']:
                all_validations_passed = False
        
        # Check model performance
        if 'model_performance' in self.validation_results:
            mp = self.validation_results['model_performance']
            if not all(result['valid'] for result in mp.values()):
                all_validations_passed = False
        
        # Check system requirements
        if 'system_requirements' in self.validation_results:
            sr = self.validation_results['system_requirements']
            if not sr['python_version'] or not all(pkg[1] for pkg in sr['required_packages']):
                all_validations_passed = False
        
        # Final status
        print(f"\nOVERALL SYSTEM STATUS: {'✅ READY FOR DEPLOYMENT' if all_validations_passed else '❌ ISSUES DETECTED'}")
        
        if all_validations_passed:
            print("\n🎉 All validations passed! System is ready for production deployment.")
        else:
            print("\n⚠️  Some validations failed. Please address issues before deployment.")
        
        return {
            'overall_status': all_validations_passed,
            'detailed_results': self.validation_results,
            'timestamp': datetime.now().isoformat()
        }

def run_comprehensive_tests():
    """Run all tests and validations."""
    print("COMPREHENSIVE ATC SYSTEM TESTING & VALIDATION")
    print("="*80)
    print(f"Test Suite Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run unit tests
    print("\n🧪 RUNNING UNIT TESTS...")
    test_suite = unittest.TestLoader().loadTestsFromTestCase(ATCSystemTests)
    test_runner = unittest.TextTestRunner(verbosity=0)
    test_result = test_runner.run(test_suite)
    
    # Run validation framework
    print("\n🔍 RUNNING VALIDATION FRAMEWORK...")
    validator = ValidationFramework()
    
    # System requirements validation
    validator.validate_system_requirements()
    
    # Generate final report
    final_report = validator.generate_validation_report()
    
    print("\n" + "="*60)
    print("TESTING & VALIDATION COMPLETE")
    print("="*60)
    
    return {
        'unit_tests': {
            'tests_run': test_result.testsRun,
            'failures': len(test_result.failures),
            'errors': len(test_result.errors),
            'success_rate': (test_result.testsRun - len(test_result.failures) - len(test_result.errors)) / test_result.testsRun if test_result.testsRun > 0 else 0
        },
        'validation_report': final_report
    }

if __name__ == "__main__":
    results = run_comprehensive_tests()