"""
Comprehensive Model Evaluation and Comparison Report
This module provides detailed evaluation of ML models for flight delay prediction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """
    Comprehensive model evaluation and comparison system.
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self):
        """Load and prepare data for model evaluation."""
        try:
            # Try to load engineered data
            from flight_delay_predictor import FlightDelayPredictor
            predictor = FlightDelayPredictor()
            df = predictor.create_sample_data()
            
            # Prepare features and target
            X, y = predictor.prepare_features(df)
            
            # Split data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            print(f"Data loaded: {len(X)} samples, {len(X.columns)} features")
            print(f"Training set: {len(self.X_train)} samples")
            print(f"Test set: {len(self.X_test)} samples")
            print(f"Class distribution: {y.value_counts().to_dict()}")
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def initialize_models(self):
        """Initialize different ML models for comparison."""
        self.models = {
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                random_state=42, max_iter=1000
            ),
            'Support Vector Machine': SVC(
                kernel='rbf', probability=True, random_state=42
            )
        }
        
        print(f"Initialized {len(self.models)} models for evaluation")
    
    def train_and_evaluate_models(self):
        """Train and evaluate all models."""
        print("\n" + "="*60)
        print("MODEL TRAINING AND EVALUATION")
        print("="*60)
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred),
                'recall': recall_score(self.y_test, y_pred),
                'f1_score': f1_score(self.y_test, y_pred),
                'auc': roc_auc_score(self.y_test, y_pred_proba) if y_pred_proba is not None else None
            }
            
            # Cross-validation scores
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='f1')
            metrics['cv_mean'] = cv_scores.mean()
            metrics['cv_std'] = cv_scores.std()
            
            # Confusion matrix
            cm = confusion_matrix(self.y_test, y_pred)
            
            # Store results
            self.results[name] = {
                'model': model,
                'metrics': metrics,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'confusion_matrix': cm,
                'cv_scores': cv_scores
            }
            
            # Print results
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1 Score: {metrics['f1_score']:.4f}")
            if metrics['auc']:
                print(f"  AUC: {metrics['auc']:.4f}")
            print(f"  CV F1: {metrics['cv_mean']:.4f} (±{metrics['cv_std']:.4f})")
    
    def generate_comparison_report(self):
        """Generate comprehensive model comparison report."""
        print("\n" + "="*60)
        print("MODEL COMPARISON REPORT")
        print("="*60)
        
        # Create comparison DataFrame
        comparison_data = []
        for name, result in self.results.items():
            metrics = result['metrics']
            comparison_data.append({
                'Model': name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1 Score': metrics['f1_score'],
                'AUC': metrics['auc'] if metrics['auc'] else 0,
                'CV F1 Mean': metrics['cv_mean'],
                'CV F1 Std': metrics['cv_std']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.round(4)
        
        print("\nPerformance Comparison:")
        print(comparison_df.to_string(index=False))
        
        # Find best models
        best_accuracy = comparison_df.loc[comparison_df['Accuracy'].idxmax(), 'Model']
        best_f1 = comparison_df.loc[comparison_df['F1 Score'].idxmax(), 'Model']
        best_auc = comparison_df.loc[comparison_df['AUC'].idxmax(), 'Model']
        
        print(f"\nBest Models:")
        print(f"  Highest Accuracy: {best_accuracy} ({comparison_df['Accuracy'].max():.4f})")
        print(f"  Highest F1 Score: {best_f1} ({comparison_df['F1 Score'].max():.4f})")
        print(f"  Highest AUC: {best_auc} ({comparison_df['AUC'].max():.4f})")
        
        return comparison_df
    
    def detailed_model_analysis(self, model_name):
        """Provide detailed analysis for a specific model."""
        if model_name not in self.results:
            print(f"Model {model_name} not found in results")
            return
        
        result = self.results[model_name]
        model = result['model']
        
        print(f"\n" + "="*60)
        print(f"DETAILED ANALYSIS: {model_name}")
        print("="*60)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, result['predictions']))
        
        # Confusion matrix
        print("\nConfusion Matrix:")
        cm = result['confusion_matrix']
        print(f"True Negatives: {cm[0,0]}")
        print(f"False Positives: {cm[0,1]}")
        print(f"False Negatives: {cm[1,0]}")
        print(f"True Positives: {cm[1,1]}")
        
        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            print("\nTop 10 Feature Importances:")
            feature_importance = pd.DataFrame({
                'feature': self.X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(feature_importance.head(10).to_string(index=False))
        
        # Cross-validation analysis
        cv_scores = result['cv_scores']
        print(f"\nCross-Validation Analysis:")
        print(f"  Mean F1 Score: {cv_scores.mean():.4f}")
        print(f"  Standard Deviation: {cv_scores.std():.4f}")
        print(f"  Min Score: {cv_scores.min():.4f}")
        print(f"  Max Score: {cv_scores.max():.4f}")
        print(f"  Score Range: {cv_scores.max() - cv_scores.min():.4f}")
    
    def model_stability_analysis(self):
        """Analyze model stability and variance."""
        print("\n" + "="*60)
        print("MODEL STABILITY ANALYSIS")
        print("="*60)
        
        stability_results = {}
        
        for name, result in self.results.items():
            cv_scores = result['cv_scores']
            stability_results[name] = {
                'mean_performance': cv_scores.mean(),
                'performance_std': cv_scores.std(),
                'coefficient_of_variation': cv_scores.std() / cv_scores.mean(),
                'stability_score': 1 - (cv_scores.std() / cv_scores.mean())  # Higher is more stable
            }
        
        # Sort by stability
        stability_df = pd.DataFrame(stability_results).T
        stability_df = stability_df.sort_values('stability_score', ascending=False)
        
        print("Model Stability Ranking (Higher = More Stable):")
        for idx, (model, row) in enumerate(stability_df.iterrows(), 1):
            print(f"{idx}. {model}")
            print(f"   Stability Score: {row['stability_score']:.4f}")
            print(f"   Performance: {row['mean_performance']:.4f} (±{row['performance_std']:.4f})")
            print(f"   Coefficient of Variation: {row['coefficient_of_variation']:.4f}")
            print()
        
        return stability_df
    
    def generate_recommendations(self):
        """Generate model selection recommendations."""
        print("\n" + "="*60)
        print("MODEL SELECTION RECOMMENDATIONS")
        print("="*60)
        
        # Get best models by different criteria
        comparison_data = []
        for name, result in self.results.items():
            metrics = result['metrics']
            comparison_data.append({
                'Model': name,
                'F1': metrics['f1_score'],
                'Accuracy': metrics['accuracy'],
                'AUC': metrics['auc'] if metrics['auc'] else 0,
                'CV_Mean': metrics['cv_mean'],
                'CV_Std': metrics['cv_std']
            })
        
        df = pd.DataFrame(comparison_data)
        
        # Recommendations
        best_overall = df.loc[df['F1'].idxmax(), 'Model']
        most_stable = df.loc[df['CV_Std'].idxmin(), 'Model']
        best_generalization = df.loc[df['CV_Mean'].idxmax(), 'Model']
        
        print("RECOMMENDATIONS:")
        print(f"\n1. BEST OVERALL PERFORMANCE: {best_overall}")
        print(f"   - Highest F1 Score: {df.loc[df['Model'] == best_overall, 'F1'].iloc[0]:.4f}")
        print(f"   - Recommended for: Production deployment with balanced precision/recall")
        
        print(f"\n2. MOST STABLE MODEL: {most_stable}")
        print(f"   - Lowest CV Standard Deviation: {df.loc[df['Model'] == most_stable, 'CV_Std'].iloc[0]:.4f}")
        print(f"   - Recommended for: Consistent performance across different data")
        
        print(f"\n3. BEST GENERALIZATION: {best_generalization}")
        print(f"   - Highest CV Mean: {df.loc[df['Model'] == best_generalization, 'CV_Mean'].iloc[0]:.4f}")
        print(f"   - Recommended for: New, unseen data")
        
        print(f"\nFINAL RECOMMENDATION:")
        if best_overall == best_generalization:
            print(f"✅ {best_overall} - Excellent choice for production")
            print("   Combines best performance with good generalization")
        else:
            print(f"⚖️  Consider trade-off between {best_overall} (performance) and {best_generalization} (generalization)")
        
        return {
            'best_overall': best_overall,
            'most_stable': most_stable,
            'best_generalization': best_generalization
        }
    
    def run_complete_evaluation(self):
        """Run complete model evaluation pipeline."""
        print("COMPREHENSIVE MODEL EVALUATION REPORT")
        print("="*80)
        print(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Load data
        if not self.load_data():
            return None
        
        # Initialize and train models
        self.initialize_models()
        self.train_and_evaluate_models()
        
        # Generate reports
        comparison_df = self.generate_comparison_report()
        stability_df = self.model_stability_analysis()
        recommendations = self.generate_recommendations()
        
        # Detailed analysis for best model
        best_model = recommendations['best_overall']
        self.detailed_model_analysis(best_model)
        
        print("\n" + "="*60)
        print("EVALUATION COMPLETE")
        print("="*60)
        
        return {
            'comparison': comparison_df,
            'stability': stability_df,
            'recommendations': recommendations,
            'detailed_results': self.results
        }

def main():
    """Run comprehensive model evaluation."""
    evaluator = ModelEvaluator()
    results = evaluator.run_complete_evaluation()
    return evaluator, results

if __name__ == "__main__":
    evaluator, results = main()