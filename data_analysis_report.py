"""
Comprehensive Data Analysis Report for ATC Flight Data
This module generates statistical analysis, data quality assessment, and exploratory data analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class FlightDataAnalyzer:
    """
    Comprehensive data analysis for flight operations data.
    """
    
    def __init__(self, data_path='2019-2023/flights_sample_3m.csv'):
        self.data_path = data_path
        self.df = None
        self.analysis_results = {}
        
    def load_data(self):
        """Load and prepare flight data for analysis."""
        print("Loading flight data for analysis...")
        self.df = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.df):,} flight records with {len(self.df.columns)} features")
        return self.df
    
    def data_quality_assessment(self):
        """Perform comprehensive data quality assessment."""
        print("\n" + "="*60)
        print("DATA QUALITY ASSESSMENT")
        print("="*60)
        
        # Basic statistics
        total_records = len(self.df)
        total_features = len(self.df.columns)
        
        # Missing values analysis
        missing_data = self.df.isnull().sum()
        missing_percentage = (missing_data / total_records) * 100
        
        quality_report = {
            'total_records': total_records,
            'total_features': total_features,
            'missing_data': missing_data[missing_data > 0],
            'missing_percentage': missing_percentage[missing_percentage > 0],
            'complete_records': self.df.dropna().shape[0],
            'completeness_rate': (self.df.dropna().shape[0] / total_records) * 100
        }
        
        print(f"Total Records: {total_records:,}")
        print(f"Total Features: {total_features}")
        print(f"Complete Records: {quality_report['complete_records']:,}")
        print(f"Data Completeness: {quality_report['completeness_rate']:.2f}%")
        
        if len(quality_report['missing_data']) > 0:
            print("\nMissing Data Summary:")
            for col, count in quality_report['missing_data'].items():
                pct = quality_report['missing_percentage'][col]
                print(f"  {col}: {count:,} ({pct:.2f}%)")
        
        self.analysis_results['data_quality'] = quality_report
        return quality_report
    
    def descriptive_statistics(self):
        """Generate comprehensive descriptive statistics."""
        print("\n" + "="*60)
        print("DESCRIPTIVE STATISTICS")
        print("="*60)
        
        # Numerical columns analysis
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        stats_summary = {
            'numerical_summary': self.df[numerical_cols].describe(),
            'categorical_summary': {},
            'delay_statistics': {},
            'airport_statistics': {}
        }
        
        # Delay analysis
        if 'DEP_DELAY' in self.df.columns:
            delay_col = 'DEP_DELAY'
        elif 'DepDelay' in self.df.columns:
            delay_col = 'DepDelay'
        else:
            delay_col = None
        
        if delay_col and delay_col in self.df.columns:
            delays = self.df[delay_col].dropna()
            stats_summary['delay_statistics'] = {
                'mean_delay': delays.mean(),
                'median_delay': delays.median(),
                'std_delay': delays.std(),
                'min_delay': delays.min(),
                'max_delay': delays.max(),
                'on_time_flights': (delays <= 15).sum(),
                'delayed_flights': (delays > 15).sum(),
                'delay_rate': (delays > 15).mean() * 100
            }
            
            print(f"Delay Analysis ({delay_col}):")
            print(f"  Mean Delay: {stats_summary['delay_statistics']['mean_delay']:.2f} minutes")
            print(f"  Median Delay: {stats_summary['delay_statistics']['median_delay']:.2f} minutes")
            print(f"  Delay Rate: {stats_summary['delay_statistics']['delay_rate']:.2f}%")
            print(f"  On-time Flights: {stats_summary['delay_statistics']['on_time_flights']:,}")
            print(f"  Delayed Flights: {stats_summary['delay_statistics']['delayed_flights']:,}")
        
        # Airport analysis
        if 'ORIGIN' in self.df.columns:
            airport_stats = self.df['ORIGIN'].value_counts().head(10)
            stats_summary['airport_statistics'] = {
                'total_airports': self.df['ORIGIN'].nunique(),
                'top_airports': airport_stats.to_dict(),
                'flights_per_airport': self.df.groupby('ORIGIN').size().describe()
            }
            
            print(f"\nAirport Analysis:")
            print(f"  Total Airports: {stats_summary['airport_statistics']['total_airports']}")
            print(f"  Top 5 Busiest Airports:")
            for airport, count in list(airport_stats.head(5).items()):
                print(f"    {airport}: {count:,} flights")
        
        # Categorical variables summary
        for col in categorical_cols[:5]:  # Limit to first 5 categorical columns
            if col in self.df.columns:
                value_counts = self.df[col].value_counts()
                stats_summary['categorical_summary'][col] = {
                    'unique_values': self.df[col].nunique(),
                    'most_common': value_counts.head(3).to_dict()
                }
        
        self.analysis_results['descriptive_stats'] = stats_summary
        return stats_summary
    
    def correlation_analysis(self):
        """Perform correlation analysis on numerical variables."""
        print("\n" + "="*60)
        print("CORRELATION ANALYSIS")
        print("="*60)
        
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) > 1:
            correlation_matrix = self.df[numerical_cols].corr()
            
            # Find high correlations
            high_corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_val = correlation_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:  # High correlation threshold
                        high_corr_pairs.append({
                            'var1': correlation_matrix.columns[i],
                            'var2': correlation_matrix.columns[j],
                            'correlation': corr_val
                        })
            
            correlation_results = {
                'correlation_matrix': correlation_matrix,
                'high_correlations': high_corr_pairs,
                'avg_correlation': correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
            }
            
            print(f"Average correlation: {correlation_results['avg_correlation']:.3f}")
            
            if high_corr_pairs:
                print("\nHigh Correlations (|r| > 0.7):")
                for pair in high_corr_pairs:
                    print(f"  {pair['var1']} ↔ {pair['var2']}: {pair['correlation']:.3f}")
            else:
                print("No high correlations found (|r| > 0.7)")
            
            self.analysis_results['correlation'] = correlation_results
            return correlation_results
        
        return None
    
    def temporal_analysis(self):
        """Analyze temporal patterns in flight data."""
        print("\n" + "="*60)
        print("TEMPORAL ANALYSIS")
        print("="*60)
        
        temporal_results = {}
        
        # Date analysis
        if 'FL_DATE' in self.df.columns:
            self.df['FL_DATE'] = pd.to_datetime(self.df['FL_DATE'])
            self.df['YEAR'] = self.df['FL_DATE'].dt.year
            self.df['MONTH'] = self.df['FL_DATE'].dt.month
            self.df['DAY_OF_WEEK'] = self.df['FL_DATE'].dt.dayofweek
            
            # Yearly trends
            yearly_flights = self.df.groupby('YEAR').size()
            temporal_results['yearly_trends'] = yearly_flights.to_dict()
            
            # Monthly patterns
            monthly_flights = self.df.groupby('MONTH').size()
            temporal_results['monthly_patterns'] = monthly_flights.to_dict()
            
            # Day of week patterns
            dow_flights = self.df.groupby('DAY_OF_WEEK').size()
            temporal_results['day_of_week_patterns'] = dow_flights.to_dict()
            
            print("Temporal Patterns:")
            print(f"  Data Range: {self.df['FL_DATE'].min()} to {self.df['FL_DATE'].max()}")
            print(f"  Peak Year: {yearly_flights.idxmax()} ({yearly_flights.max():,} flights)")
            print(f"  Peak Month: {monthly_flights.idxmax()} ({monthly_flights.max():,} flights)")
            print(f"  Busiest Day of Week: {dow_flights.idxmax()} ({dow_flights.max():,} flights)")
        
        # Time of day analysis
        if 'CRS_DEP_TIME' in self.df.columns:
            self.df['DEP_HOUR'] = (self.df['CRS_DEP_TIME'] // 100).clip(0, 23)
            hourly_flights = self.df.groupby('DEP_HOUR').size()
            temporal_results['hourly_patterns'] = hourly_flights.to_dict()
            
            peak_hour = hourly_flights.idxmax()
            print(f"  Peak Departure Hour: {peak_hour}:00 ({hourly_flights.max():,} flights)")
        
        self.analysis_results['temporal'] = temporal_results
        return temporal_results
    
    def delay_pattern_analysis(self):
        """Analyze delay patterns and causes."""
        print("\n" + "="*60)
        print("DELAY PATTERN ANALYSIS")
        print("="*60)
        
        delay_results = {}
        
        # Find delay column
        delay_col = None
        for col in ['DEP_DELAY', 'DepDelay', 'DELAY']:
            if col in self.df.columns:
                delay_col = col
                break
        
        if delay_col:
            # Delay categories
            delays = self.df[delay_col].dropna()
            delay_results['delay_distribution'] = {
                'early': (delays < 0).sum(),
                'on_time': ((delays >= 0) & (delays <= 15)).sum(),
                'minor_delay': ((delays > 15) & (delays <= 60)).sum(),
                'major_delay': ((delays > 60) & (delays <= 180)).sum(),
                'severe_delay': (delays > 180).sum()
            }
            
            # Delay by airport
            if 'ORIGIN' in self.df.columns:
                airport_delays = self.df.groupby('ORIGIN')[delay_col].agg(['mean', 'count']).round(2)
                airport_delays = airport_delays[airport_delays['count'] >= 100]  # Minimum flights
                delay_results['airport_delays'] = airport_delays.sort_values('mean', ascending=False).head(10)
            
            # Delay by time of day
            if 'DEP_HOUR' in self.df.columns:
                hourly_delays = self.df.groupby('DEP_HOUR')[delay_col].mean()
                delay_results['hourly_delays'] = hourly_delays.to_dict()
            
            print("Delay Distribution:")
            for category, count in delay_results['delay_distribution'].items():
                pct = (count / len(delays)) * 100
                print(f"  {category.replace('_', ' ').title()}: {count:,} ({pct:.1f}%)")
        
        self.analysis_results['delay_patterns'] = delay_results
        return delay_results
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive analysis report."""
        print("\n" + "="*80)
        print("COMPREHENSIVE FLIGHT DATA ANALYSIS REPORT")
        print("="*80)
        
        # Load data if not already loaded
        if self.df is None:
            self.load_data()
        
        # Run all analyses
        self.data_quality_assessment()
        self.descriptive_statistics()
        self.correlation_analysis()
        self.temporal_analysis()
        self.delay_pattern_analysis()
        
        # Generate summary
        print("\n" + "="*60)
        print("EXECUTIVE SUMMARY")
        print("="*60)
        
        summary = {
            'dataset_size': f"{len(self.df):,} flight records",
            'time_period': f"{self.df['FL_DATE'].min()} to {self.df['FL_DATE'].max()}" if 'FL_DATE' in self.df.columns else "Unknown",
            'data_quality': f"{self.analysis_results.get('data_quality', {}).get('completeness_rate', 0):.1f}% complete",
            'delay_rate': f"{self.analysis_results.get('descriptive_stats', {}).get('delay_statistics', {}).get('delay_rate', 0):.1f}% delayed flights",
            'airports_covered': self.analysis_results.get('descriptive_stats', {}).get('airport_statistics', {}).get('total_airports', 0)
        }
        
        print(f"Dataset: {summary['dataset_size']}")
        print(f"Time Period: {summary['time_period']}")
        print(f"Data Quality: {summary['data_quality']}")
        print(f"Overall Delay Rate: {summary['delay_rate']}")
        print(f"Airports Covered: {summary['airports_covered']}")
        
        return self.analysis_results

def main():
    """Run comprehensive data analysis."""
    analyzer = FlightDataAnalyzer()
    results = analyzer.generate_comprehensive_report()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("Results saved to analysis_results object")
    
    return analyzer, results

if __name__ == "__main__":
    analyzer, results = main()