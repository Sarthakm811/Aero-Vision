"""
Data Cleaning and Preprocessing Module
Handles data cleaning, validation, and feature engineering for ATC ML project.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sqlite3
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DataCleaner:
    """Handles all data cleaning and preprocessing operations."""
    
    def __init__(self):
        self.data_dir = Path('data')
        self.processed_dir = self.data_dir / 'processed'
        self.processed_dir.mkdir(exist_ok=True)
        
        # Data quality thresholds
        self.altitude_bounds = (0, 50000)  # feet
        self.speed_bounds = (0, 1000)     # knots
        self.lat_bounds = (20, 50)        # US continental bounds
        self.lon_bounds = (-130, -65)
    
    def load_raw_data(self):
        """Load raw flight and weather data."""
        print("📂 Loading raw data...")
        
        # Try to load from different sources
        flight_data = None
        weather_data = None
        
        # Check for simulated data first
        sim_flight_path = self.data_dir / 'simulated' / 'flight_trajectories.parquet'
        sim_weather_path = self.data_dir / 'simulated' / 'weather_data.parquet'
        
        if sim_flight_path.exists():
            flight_data = pd.read_parquet(sim_flight_path)
            print(f"✅ Loaded {len(flight_data)} simulated flight records")
        
        if sim_weather_path.exists():
            weather_data = pd.read_parquet(sim_weather_path)
            print(f"✅ Loaded {len(weather_data)} weather records")
        
        # Check for real data
        real_flight_path = self.data_dir / 'raw' / 'flight_data.parquet'
        if real_flight_path.exists():
            real_data = pd.read_parquet(real_flight_path)
            if flight_data is not None:
                flight_data = pd.concat([flight_data, real_data], ignore_index=True)
            else:
                flight_data = real_data
            print(f"✅ Added {len(real_data)} real flight records")
        
        if flight_data is None:
            raise FileNotFoundError("No flight data found. Run flight simulator first.")
        
        return flight_data, weather_data
    
    def clean_flight_data(self, df):
        """Clean and validate flight trajectory data."""
        print("🧹 Cleaning flight data...")
        
        initial_count = len(df)
        
        # Convert timestamp to datetime if needed
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Remove records with missing critical fields
        critical_fields = ['lat', 'lon', 'altitude', 'flight_id']
        df = df.dropna(subset=critical_fields)
        
        # Validate coordinate bounds
        df = df[
            (df['lat'].between(*self.lat_bounds)) &
            (df['lon'].between(*self.lon_bounds))
        ]
        
        # Validate altitude bounds
        df = df[df['altitude'].between(*self.altitude_bounds)]
        
        # Validate ground speed
        if 'ground_speed' in df.columns:
            df = df[df['ground_speed'].between(*self.speed_bounds)]
        
        # Remove duplicate records
        df = df.drop_duplicates(subset=['flight_id', 'timestamp'])
        
        # Sort by flight and timestamp
        df = df.sort_values(['flight_id', 'timestamp'])
        
        print(f"✅ Cleaned data: {initial_count} → {len(df)} records ({len(df)/initial_count*100:.1f}% retained)")
        
        return df
    
    def clean_weather_data(self, df):
        """Clean and validate weather data."""
        if df is None:
            print("⚠️  No weather data to clean")
            return None
        
        print("🌤️  Cleaning weather data...")
        
        initial_count = len(df)
        
        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Remove invalid weather values
        df = df[
            (df['temperature'].between(-50, 50)) &  # Celsius
            (df['wind_speed'].between(0, 200)) &    # km/h
            (df['visibility'].between(0, 50)) &     # km
            (df['pressure'].between(900, 1100))     # hPa
        ]
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['timestamp', 'location'])
        
        print(f"✅ Cleaned weather: {initial_count} → {len(df)} records")
        
        return df
    
    def engineer_features(self, flight_df, weather_df=None):
        """Create engineered features for ML models."""
        print("⚙️  Engineering features...")
        
        # Time-based features
        flight_df['hour'] = flight_df['timestamp'].dt.hour
        flight_df['day_of_week'] = flight_df['timestamp'].dt.dayofweek
        flight_df['month'] = flight_df['timestamp'].dt.month
        flight_df['is_weekend'] = flight_df['day_of_week'].isin([5, 6]).astype(int)
        flight_df['is_peak_hour'] = flight_df['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
        
        # Flight-specific features
        flight_df = self._add_flight_features(flight_df)
        
        # Airport congestion features
        flight_df = self._add_congestion_features(flight_df)
        
        # Weather features (if available)
        if weather_df is not None:
            flight_df = self._merge_weather_features(flight_df, weather_df)
        
        # Delay calculation (target variable)
        flight_df = self._calculate_delays(flight_df)
        
        print(f"✅ Feature engineering complete: {flight_df.shape[1]} features")
        
        return flight_df
    
    def _add_flight_features(self, df):
        """Add flight-specific engineered features."""
        
        # Calculate distance to destination for each point
        df['distance_to_dest'] = df.apply(
            lambda row: self._haversine_distance(
                row['lat'], row['lon'],
                self._get_airport_coords(row['destination'])
            ), axis=1
        )
        
        # Calculate flight progress (0 to 1)
        flight_groups = df.groupby('flight_id')
        df['flight_progress'] = flight_groups['distance_to_dest'].transform(
            lambda x: 1 - (x / x.iloc[0]) if len(x) > 0 and x.iloc[0] > 0 else 0
        )
        
        # Speed and altitude changes
        df['speed_change'] = flight_groups['ground_speed'].diff()
        df['altitude_change'] = flight_groups['altitude'].diff()
        
        # Flight duration so far
        df['flight_duration'] = flight_groups['timestamp'].transform(
            lambda x: (x - x.iloc[0]).dt.total_seconds() / 60  # minutes
        )
        
        return df
    
    def _add_congestion_features(self, df):
        """Add airport congestion features."""
        
        # Hourly departures by airport
        hourly_deps = df.groupby(['origin', 'hour']).size().reset_index(name='hourly_departures')
        df = df.merge(hourly_deps, on=['origin', 'hour'], how='left')
        
        # Airport traffic density (flights per hour in vicinity)
        df['airport_traffic'] = df.groupby(['origin', pd.Grouper(key='timestamp', freq='H')]).transform('size')
        
        return df
    
    def _merge_weather_features(self, flight_df, weather_df):
        """Merge weather data with flight data."""
        
        # Create time windows for weather matching
        weather_df['time_window'] = weather_df['timestamp'].dt.floor('H')
        flight_df['time_window'] = flight_df['timestamp'].dt.floor('H')
        
        # Merge weather at origin
        origin_weather = weather_df.rename(columns={
            'location': 'origin',
            'wind_speed': 'origin_wind_speed',
            'visibility': 'origin_visibility',
            'weather_condition': 'origin_weather'
        })
        
        flight_df = flight_df.merge(
            origin_weather[['origin', 'time_window', 'origin_wind_speed', 'origin_visibility', 'origin_weather']],
            on=['origin', 'time_window'],
            how='left'
        )
        
        # Fill missing weather with defaults
        flight_df['origin_wind_speed'] = flight_df['origin_wind_speed'].fillna(10)
        flight_df['origin_visibility'] = flight_df['origin_visibility'].fillna(15)
        flight_df['origin_weather'] = flight_df['origin_weather'].fillna('clear')
        
        return flight_df
    
    def _calculate_delays(self, df):
        """Calculate delay metrics for each flight."""
        
        if 'scheduled_departure' in df.columns and 'actual_departure' in df.columns:
            # Convert to datetime if needed
            df['scheduled_departure'] = pd.to_datetime(df['scheduled_departure'])
            df['actual_departure'] = pd.to_datetime(df['actual_departure'])
            
            # Calculate departure delay in minutes
            df['departure_delay'] = (df['actual_departure'] - df['scheduled_departure']).dt.total_seconds() / 60
            
            # Binary delay indicator (>15 minutes)
            df['is_delayed'] = (df['departure_delay'] > 15).astype(int)
        else:
            # Create synthetic delay based on congestion and weather
            df['departure_delay'] = (
                df.get('hourly_departures', 0) * 0.5 +
                df.get('origin_wind_speed', 0) * 0.3 +
                np.random.exponential(5, len(df))
            )
            df['is_delayed'] = (df['departure_delay'] > 15).astype(int)
        
        return df
    
    def _haversine_distance(self, lat1, lon1, dest_coords):
        """Calculate haversine distance between two points."""
        if dest_coords is None:
            return 0
        
        lat2, lon2 = dest_coords
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Earth radius in km
        r = 6371
        return r * c
    
    def _get_airport_coords(self, airport_code):
        """Get coordinates for airport code."""
        airports = {
            'JFK': (40.6413, -73.7781),
            'LAX': (33.9425, -118.4081),
            'ORD': (41.9742, -87.9073),
            'ATL': (33.6407, -84.4277),
            'DFW': (32.8998, -97.0403),
            'DEN': (39.8561, -104.6737),
            'SFO': (37.6213, -122.3790),
            'SEA': (47.4502, -122.3088),
            'MIA': (25.7959, -80.2870),
            'BOS': (42.3656, -71.0096)
        }
        return airports.get(airport_code)
    
    def save_processed_data(self, flight_df, weather_df=None):
        """Save processed data to files."""
        print("💾 Saving processed data...")
        
        # Save flight data
        flight_df.to_parquet(self.processed_dir / 'flights_cleaned.parquet', index=False)
        flight_df.to_csv(self.processed_dir / 'flights_cleaned.csv', index=False)
        
        if weather_df is not None:
            weather_df.to_parquet(self.processed_dir / 'weather_cleaned.parquet', index=False)
            weather_df.to_csv(self.processed_dir / 'weather_cleaned.csv', index=False)
        
        print(f"✅ Saved processed data to {self.processed_dir}")
    
    def create_database(self, flight_df, weather_df=None):
        """Create SQLite database for processed data."""
        print("🗄️  Creating SQLite database...")
        
        db_path = self.processed_dir / 'atc_data.db'
        
        with sqlite3.connect(db_path) as conn:
            flight_df.to_sql('flights', conn, if_exists='replace', index=False)
            
            if weather_df is not None:
                weather_df.to_sql('weather', conn, if_exists='replace', index=False)
            
            # Create indexes for better query performance
            conn.execute('CREATE INDEX IF NOT EXISTS idx_flight_id ON flights(flight_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON flights(timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_origin ON flights(origin)')
            
        print(f"✅ Database created: {db_path}")
    
    def generate_data_summary(self, flight_df, weather_df=None):
        """Generate and save data summary statistics."""
        print("📊 Generating data summary...")
        
        summary = {
            'total_flights': flight_df['flight_id'].nunique(),
            'total_records': len(flight_df),
            'date_range': {
                'start': flight_df['timestamp'].min().isoformat(),
                'end': flight_df['timestamp'].max().isoformat()
            },
            'airports': {
                'origins': flight_df['origin'].unique().tolist(),
                'destinations': flight_df['destination'].unique().tolist()
            },
            'delay_stats': {
                'avg_delay': flight_df['departure_delay'].mean(),
                'delayed_flights_pct': flight_df['is_delayed'].mean() * 100
            }
        }
        
        # Save summary
        import json
        with open(self.processed_dir / 'data_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("✅ Data summary saved")
        return summary
    
    def process_all_data(self):
        """Run the complete data processing pipeline."""
        print("🚀 Starting data processing pipeline...")
        
        # Load raw data
        flight_df, weather_df = self.load_raw_data()
        
        # Clean data
        flight_df = self.clean_flight_data(flight_df)
        if weather_df is not None:
            weather_df = self.clean_weather_data(weather_df)
        
        # Engineer features
        flight_df = self.engineer_features(flight_df, weather_df)
        
        # Save processed data
        self.save_processed_data(flight_df, weather_df)
        
        # Create database
        self.create_database(flight_df, weather_df)
        
        # Generate summary
        summary = self.generate_data_summary(flight_df, weather_df)
        
        print("✅ Data processing pipeline complete!")
        return flight_df, weather_df, summary

def main():
    """Run data cleaning when executed directly."""
    cleaner = DataCleaner()
    cleaner.process_all_data()

if __name__ == "__main__":
    main()