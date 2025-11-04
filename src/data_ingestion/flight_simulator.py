"""
Flight Data Simulator for ATC ML Project
Generates realistic flight trajectories and data for testing and demo purposes.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path
from geopy.distance import geodesic
import random

class FlightSimulator:
    """Simulates realistic flight data for ATC system testing."""
    
    def __init__(self):
        self.airports = {
            'JFK': {'lat': 40.6413, 'lon': -73.7781, 'name': 'John F. Kennedy International'},
            'LAX': {'lat': 33.9425, 'lon': -118.4081, 'name': 'Los Angeles International'},
            'ORD': {'lat': 41.9742, 'lon': -87.9073, 'name': 'Chicago O\'Hare International'},
            'ATL': {'lat': 33.6407, 'lon': -84.4277, 'name': 'Hartsfield-Jackson Atlanta'},
            'DFW': {'lat': 32.8998, 'lon': -97.0403, 'name': 'Dallas/Fort Worth International'},
            'DEN': {'lat': 39.8561, 'lon': -104.6737, 'name': 'Denver International'},
            'SFO': {'lat': 37.6213, 'lon': -122.3790, 'name': 'San Francisco International'},
            'SEA': {'lat': 47.4502, 'lon': -122.3088, 'name': 'Seattle-Tacoma International'},
            'MIA': {'lat': 25.7959, 'lon': -80.2870, 'name': 'Miami International'},
            'BOS': {'lat': 42.3656, 'lon': -71.0096, 'name': 'Boston Logan International'}
        }
        
        self.airlines = ['AA', 'UA', 'DL', 'WN', 'B6', 'AS', 'NK', 'F9', 'G4', 'SY']
        
        # Ensure data directories exist
        Path('data/raw').mkdir(parents=True, exist_ok=True)
        Path('data/processed').mkdir(parents=True, exist_ok=True)
        Path('data/simulated').mkdir(parents=True, exist_ok=True)
    
    def generate_flight_plan(self, origin, destination, departure_time):
        """Generate a realistic flight plan between two airports."""
        origin_coords = (self.airports[origin]['lat'], self.airports[origin]['lon'])
        dest_coords = (self.airports[destination]['lat'], self.airports[destination]['lon'])
        
        # Calculate great circle distance
        distance_km = geodesic(origin_coords, dest_coords).kilometers
        
        # Estimate flight time (average commercial speed ~800 km/h)
        flight_time_hours = distance_km / 800
        estimated_arrival = departure_time + timedelta(hours=flight_time_hours)
        
        # Generate waypoints along the route
        waypoints = self._generate_waypoints(origin_coords, dest_coords, int(flight_time_hours * 10))
        
        return {
            'origin': origin,
            'destination': destination,
            'departure_time': departure_time,
            'estimated_arrival': estimated_arrival,
            'distance_km': distance_km,
            'flight_time_hours': flight_time_hours,
            'waypoints': waypoints
        }
    
    def _generate_waypoints(self, start, end, num_points):
        """Generate intermediate waypoints between start and end coordinates."""
        waypoints = []
        
        for i in range(num_points + 1):
            ratio = i / num_points
            
            # Linear interpolation with some realistic deviation
            lat = start[0] + (end[0] - start[0]) * ratio
            lon = start[1] + (end[1] - start[1]) * ratio
            
            # Add some realistic flight path deviation
            lat += np.random.normal(0, 0.01)  # Small random deviation
            lon += np.random.normal(0, 0.01)
            
            # Altitude profile (takeoff, cruise, descent)
            if ratio < 0.2:  # Takeoff phase
                altitude = 1000 + (35000 * ratio / 0.2)
            elif ratio > 0.8:  # Descent phase
                altitude = 35000 * (1 - ratio) / 0.2
            else:  # Cruise phase
                altitude = 35000 + np.random.normal(0, 1000)
            
            altitude = max(1000, min(42000, altitude))  # Realistic altitude bounds
            
            waypoints.append({
                'lat': lat,
                'lon': lon,
                'altitude': altitude,
                'timestamp_offset': i * 6  # 6 minutes between waypoints
            })
        
        return waypoints
    
    def generate_flight_trajectory(self, flight_plan, flight_id):
        """Generate detailed trajectory data for a single flight."""
        trajectory_data = []
        base_time = flight_plan['departure_time']
        
        # Add some realistic delays
        actual_departure_delay = np.random.exponential(5)  # Average 5 minute delay
        if np.random.random() < 0.1:  # 10% chance of significant delay
            actual_departure_delay += np.random.exponential(30)
        
        actual_departure = base_time + timedelta(minutes=actual_departure_delay)
        
        for i, waypoint in enumerate(flight_plan['waypoints']):
            timestamp = actual_departure + timedelta(minutes=waypoint['timestamp_offset'])
            
            # Calculate ground speed (varies with altitude and phase)
            if i == 0:
                ground_speed = 150  # Taxi/takeoff speed
            elif i < len(flight_plan['waypoints']) * 0.2:
                ground_speed = 250 + (450 * i / (len(flight_plan['waypoints']) * 0.2))
            elif i > len(flight_plan['waypoints']) * 0.8:
                ground_speed = 700 - (400 * (i - len(flight_plan['waypoints']) * 0.8) / (len(flight_plan['waypoints']) * 0.2))
            else:
                ground_speed = 700 + np.random.normal(0, 50)
            
            ground_speed = max(100, min(900, ground_speed))
            
            # Calculate heading to next waypoint
            if i < len(flight_plan['waypoints']) - 1:
                next_waypoint = flight_plan['waypoints'][i + 1]
                heading = self._calculate_heading(
                    waypoint['lat'], waypoint['lon'],
                    next_waypoint['lat'], next_waypoint['lon']
                )
            else:
                heading = 0  # Final approach
            
            trajectory_data.append({
                'timestamp': timestamp,
                'flight_id': flight_id,
                'callsign': flight_id,
                'lat': waypoint['lat'],
                'lon': waypoint['lon'],
                'altitude': waypoint['altitude'],
                'ground_speed': ground_speed,
                'heading': heading,
                'origin': flight_plan['origin'],
                'destination': flight_plan['destination'],
                'scheduled_departure': flight_plan['departure_time'],
                'actual_departure': actual_departure,
                'departure_delay': actual_departure_delay,
                'phase': self._determine_flight_phase(i, len(flight_plan['waypoints']))
            })
        
        return trajectory_data
    
    def _calculate_heading(self, lat1, lon1, lat2, lon2):
        """Calculate heading between two points."""
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        dlon = lon2 - lon1
        y = np.sin(dlon) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
        
        heading = np.degrees(np.arctan2(y, x))
        return (heading + 360) % 360
    
    def _determine_flight_phase(self, waypoint_index, total_waypoints):
        """Determine flight phase based on waypoint position."""
        ratio = waypoint_index / total_waypoints
        
        if ratio < 0.1:
            return 'taxi'
        elif ratio < 0.2:
            return 'takeoff'
        elif ratio < 0.3:
            return 'climb'
        elif ratio < 0.7:
            return 'cruise'
        elif ratio < 0.9:
            return 'descent'
        else:
            return 'approach'
    
    def generate_weather_data(self, timestamp, location):
        """Generate realistic weather data for a given time and location."""
        # Seasonal patterns
        month = timestamp.month
        is_winter = month in [12, 1, 2]
        is_summer = month in [6, 7, 8]
        
        # Base weather conditions
        if is_winter:
            base_temp = np.random.normal(-5, 10)
            wind_speed = np.random.exponential(15)
            visibility = np.random.normal(8, 3) if np.random.random() > 0.3 else np.random.exponential(2)
        elif is_summer:
            base_temp = np.random.normal(25, 8)
            wind_speed = np.random.exponential(10)
            visibility = np.random.normal(15, 5)
        else:
            base_temp = np.random.normal(15, 12)
            wind_speed = np.random.exponential(12)
            visibility = np.random.normal(12, 4)
        
        # Weather events
        weather_condition = 'clear'
        if visibility < 5:
            weather_condition = 'fog'
        elif wind_speed > 25:
            weather_condition = 'windy'
        elif np.random.random() < 0.1:
            weather_condition = 'rain'
        elif np.random.random() < 0.05:
            weather_condition = 'storm'
        
        return {
            'timestamp': timestamp,
            'location': location,
            'temperature': base_temp,
            'wind_speed': max(0, wind_speed),
            'wind_direction': np.random.uniform(0, 360),
            'visibility': max(0.1, visibility),
            'weather_condition': weather_condition,
            'pressure': np.random.normal(1013.25, 20)
        }
    
    def generate_demo_dataset(self, num_flights=500, days=7):
        """Generate a complete demo dataset with flights and weather."""
        print(f"🔄 Generating {num_flights} flights over {days} days...")
        
        all_trajectories = []
        all_weather = []
        
        # Generate flights over the specified period
        start_date = datetime.now() - timedelta(days=days)
        
        for day in range(days):
            current_date = start_date + timedelta(days=day)
            daily_flights = num_flights // days
            
            for flight_num in range(daily_flights):
                # Random flight details
                origin = np.random.choice(list(self.airports.keys()))
                destination = np.random.choice([k for k in self.airports.keys() if k != origin])
                airline = np.random.choice(self.airlines)
                flight_number = f"{airline}{np.random.randint(100, 9999)}"
                
                # Random departure time during the day
                departure_hour = np.random.choice(range(24), p=self._get_hourly_distribution())
                departure_time = current_date.replace(
                    hour=departure_hour,
                    minute=np.random.randint(0, 60),
                    second=0,
                    microsecond=0
                )
                
                # Generate flight plan and trajectory
                flight_plan = self.generate_flight_plan(origin, destination, departure_time)
                trajectory = self.generate_flight_trajectory(flight_plan, flight_number)
                all_trajectories.extend(trajectory)
                
                # Generate weather data for origin and destination
                weather_origin = self.generate_weather_data(departure_time, origin)
                weather_dest = self.generate_weather_data(
                    departure_time + timedelta(hours=flight_plan['flight_time_hours']),
                    destination
                )
                all_weather.extend([weather_origin, weather_dest])
        
        # Convert to DataFrames
        flights_df = pd.DataFrame(all_trajectories)
        weather_df = pd.DataFrame(all_weather)
        
        # Save datasets
        flights_df.to_parquet('data/simulated/flight_trajectories.parquet', index=False)
        weather_df.to_parquet('data/simulated/weather_data.parquet', index=False)
        
        # Also save as CSV for easy inspection
        flights_df.to_csv('data/simulated/flight_simulation.csv', index=False)
        weather_df.to_csv('data/simulated/weather_simulation.csv', index=False)
        
        print(f"✅ Generated {len(flights_df)} trajectory points for {len(flights_df['flight_id'].unique())} flights")
        print(f"✅ Generated {len(weather_df)} weather observations")
        print(f"📁 Data saved to data/simulated/")
        
        return flights_df, weather_df
    
    def _get_hourly_distribution(self):
        """Get realistic hourly flight distribution (more flights during day)."""
        # Peak hours: 6-9 AM, 12-2 PM, 5-8 PM
        hourly_weights = [
            0.01, 0.01, 0.01, 0.01, 0.02, 0.03,  # 0-5 AM
            0.08, 0.12, 0.10, 0.08, 0.06, 0.05,  # 6-11 AM
            0.07, 0.08, 0.06, 0.05, 0.04, 0.08,  # 12-5 PM
            0.10, 0.08, 0.06, 0.04, 0.03, 0.02   # 6-11 PM
        ]
        return np.array(hourly_weights) / sum(hourly_weights)

def main():
    """Generate demo data when run directly."""
    simulator = FlightSimulator()
    simulator.generate_demo_dataset(num_flights=1000, days=7)

if __name__ == "__main__":
    main()