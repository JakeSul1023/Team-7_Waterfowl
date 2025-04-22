import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
import dask.dataframe as dd
import pyarrow.csv as pv
import multiprocessing as mp
from typing import Dict, List, Optional, Tuple, Union
from sklearn.model_selection import train_test_split
from sklearn.neighbors import BallTree
from datetime import datetime, timedelta
from shapely.geometry import Point as ShapelyPoint
from scipy.interpolate import griddata
import os
import json
import requests
import math
import warnings
from functools import partial

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

print("Libraries installed")

# ========================================================
# Data Loading and Preprocessing
# ========================================================

# --- Duck Class Declaration ---
class Duck:
    def __init__(self, duckID):
        self.duckID = duckID
        self.species = ''
        self.longs = []
        self.lats = []
        self.coord = []
        self.timestamps = []

    def importLoc(self, df):
        duck_data = df[df['tag-local-identifier'] == self.duckID]

        # If the column exists, grab the first species name; otherwise keep it empty
        self.species = duck_data['individual-taxon-canonical-name'].iloc[0] if 'individual-taxon-canonical-name' in duck_data.columns else ''
        self.timestamps = duck_data['timestamp'].tolist()

        self.longs = duck_data['longitude'].tolist()
        self.lats = duck_data['latitude'].tolist()

        self.coord = list(zip(self.longs, self.lats))

# --- Data Loading Function ---
def load_duck_data(filepath="MallardOneWeek.csv"):
    """Load and preprocess duck telemetry data from CSV file"""
    try:
        duck_df = pd.read_csv(filepath)
        print(f"Successfully loaded data from {filepath}")
    except FileNotFoundError:
        print(f"Error: File {filepath} not found. Please check the file path.")
        return None, None
    except pd.errors.EmptyDataError:
        print(f"Error: File {filepath} is empty.")
        return None, None
    except pd.errors.ParserError:
        print(f"Error: Unable to parse {filepath}. Please check the file format.")
        return None, None

    print(f"Columns in the dataset: {duck_df.columns.tolist()}")

    # Drop irrelevant columns
    columns_to_drop = [
        "battery-charge-percent", "battery-charging-current", "gps-time-to-fix", "individual-local-identifier",
        "orn:transmission-protocol", "tag-voltage", "sensor-type", "acceleration-raw-x", "gps:satellite-count",
        "acceleration-raw-y", "acceleration-raw-z", "ground-speed", "gls:light-level", "study-name"
    ]
    for column in columns_to_drop:
        if column in duck_df.columns:
            duck_df.drop(columns=[column], inplace=True)

    # Convert timestamp column to proper datetime format
    try:
        original_timestamps = duck_df['timestamp'].copy()
        timestamp_converted = False
        date_formats = [
            '%m/%d/%Y %I:%M:%S %p', '%m/%d/%Y %H:%M:%S', '%Y-%m-%d %H:%M:%S',
            '%m/%d/%Y %H:%M', '%H:%M:%S', '%M:%S.%f', '%Y-%m-%dT%H:%M:%S.%fZ',
            '%Y-%m-%dT%H:%M:%SZ'
        ]
        for date_format in date_formats:
            converted = pd.to_datetime(original_timestamps, format=date_format, errors='coerce')
            if converted.notna().sum() > len(converted) * 0.5:
                duck_df['timestamp'] = converted
                timestamp_converted = True
                break

        if not timestamp_converted:
            duck_df['timestamp'] = pd.to_datetime(original_timestamps, errors='coerce', infer_datetime_format=True)

        if duck_df['timestamp'].isna().sum() > len(duck_df) * 0.9:
            duck_df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(duck_df), freq='1min')

    except Exception:
        duck_df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(duck_df), freq='1min')

    # Standardize column names via mapping
    column_mapping = {
        'location-long': 'longitude',
        'location-lat': 'latitude',
        'external-temperature': 'temperature',
        'height-above-msl': 'altitude',
        'ground-speed': 'speed',
        'heading': 'direction'
    }
    duck_df.rename(columns={k: v for k, v in column_mapping.items() if k in duck_df.columns}, inplace=True)

    # Ensure essential columns exist
    for col in ['longitude', 'latitude', 'timestamp', 'tag-local-identifier']:
        if col not in duck_df.columns:
            duck_df[col] = np.nan

    # Create GeoDataFrame for spatial operations
    geometry = [ShapelyPoint(xy) for xy in zip(duck_df['longitude'], duck_df['latitude'])]
    duck_gdf = gpd.GeoDataFrame(duck_df, geometry=geometry, crs="EPSG:4326")

    # Group data by duck ID (tag-local-identifier)
    duck_trajectories = {id: group for id, group in duck_gdf.groupby('tag-local-identifier')}

    # Instantiate a Duck object for each unique duck ID using the cleaned GeoDataFrame
    uniqueIDs = list(duck_trajectories.keys())
    ducks = {}
    for duck_id in uniqueIDs:
        duck = Duck(duck_id)
        duck.importLoc(duck_gdf)
        ducks[duck_id] = duck

    return ducks, duck_gdf

# --- Example Helper Functions ---
def get_flyway_region():
    """Define the Mississippi Flyway region"""
    flyway_bbox = [-97.12, 30.52, -80.93, 48.97]
    return flyway_bbox

def get_all_timestamps(duck_trajectories):
    """Extract all unique timestamps from duck trajectories"""
    if not duck_trajectories:
        print("No duck trajectories available")
        return []
        
    all_times = []
    for duck_id, trajectory in duck_trajectories.items():
        if 'timestamp' in trajectory.columns:
            all_times.extend(trajectory['timestamp'].tolist())
    
    # Remove NaT values
    all_times = [t for t in all_times if pd.notna(t)]
    
    # Sort and remove duplicates
    unique_times = sorted(set(all_times))
    
    print("\n=== Timestamp Analysis ===")
    print(f"Total unique timestamps: {len(unique_times)}")
    if unique_times:
        print(f"Date range: {unique_times[0]} to {unique_times[-1]}")
    print(f"================")
    
    return unique_times
    
# Example usage:
if __name__ == "__main__":
    ducks, duck_gdf = load_duck_data("MallardOneWeek.csv")
    # Now you can use `ducks` and `duck_gdf` for further processing


# ========================================================
# OpenWeather API Data Collection
# ========================================================

def fetch_openweather_data(lat, lon, api_key="02de0c63b48bcd13d425a73caa22eb81"):
    """Fetch weather data from OpenWeather API for a given lat, lon"""
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        # Extract relevant weather data and add latitude and longitude to the data
        weather = {
            "temperature": data['main']['temp'],
            "humidity": data['main']['humidity'],
            "pressure": data['main']['pressure'],
            "wind_speed": data['wind']['speed'],
            "wind_deg": data['wind']['deg'],
            "weather_description": data['weather'][0]['description'],
            "latitude": lat,  
            "longitude": lon  
        }
        
        return weather
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None



def fetch_weather_forecast_for_point(lat, lon, api_key="02de0c63b48bcd13d425a73caa22eb81"):
    """Fetch weather forecast data from OpenWeather API for a given lat/lon"""
    weather_data = fetch_openweather_data(lat, lon, api_key)
    if weather_data:
        print(f"Weather data at point ({lat}, {lon}): {weather_data}")
    else:
        print(f"Failed to retrieve weather data for point ({lat}, {lon})")
    return weather_data


def fetch_weather_data(stations, start_date, end_date):
    """
    Fetch weather data for stations using OpenWeather API (in place of NOAA)
    Returns a list of weather data dictionaries, instead of a DataFrame
    """
    all_weather_data = []
    for station in stations:
        try:
            if 'latitude' not in station or 'longitude' not in station:
                continue

            lat = station['latitude']
            lon = station['longitude']
            
            weather_data = fetch_openweather_data(lat, lon)
            if weather_data:
                weather_data['station_id'] = station['id']
                weather_data['station_name'] = station.get('name', 'Unknown')
                weather_data['latitude'] = lat
                weather_data['longitude'] = lon
                all_weather_data.append(weather_data)

        except Exception as e:
            print(f"Error fetching weather data: {e}")
    
    # Return list of dictionaries
    return all_weather_data

def fetch_multiple_openweather_data(flyway_bbox, api_key="02de0c63b48bcd13d425a73caa22eb81"):
    """Fetch weather data from multiple points within the bounding box (flyway region)"""
    weather_data = []
    lon_min, lat_min, lon_max, lat_max = flyway_bbox
    
    # Sample points within the bounding box (you can adjust the step for finer granularity)
    lons = np.linspace(lon_min, lon_max, num=5)  # 5 points in longitude direction
    lats = np.linspace(lat_min, lat_max, num=5)  # 5 points in latitude direction
    
    # Fetch weather for each point in the grid
    for lon in lons:
        for lat in lats:
            data = fetch_openweather_data(lat, lon, api_key)
            if data:
                weather_data.append(data)
    
    return weather_data

def extract_weather_at_point(weather_rasters, timestamp, lat, lon):
    """Extract weather variables at a specific point and time."""
    if not weather_rasters:
        print("No weather rasters available")
        return {}

    available_times = sorted(set(t for _, t in weather_rasters.keys()))
    print(f"Available timestamps in weather rasters: {available_times}")

    closest_timestamp = min(available_times, key=lambda x: abs(x - timestamp))
    print(f"Using closest timestamp: {closest_timestamp} at {lat}, {lon}")

    weather_at_point = {}
    for param in ['temperature', 'humidity', 'pressure', 'wind_speed', 'wind_deg']:
        weather_data = weather_rasters.get((param, closest_timestamp))
        if weather_data is not None:
            try:
                # Use the 'nearest' method to extract the closest data point to the given lat/lon
                value = weather_data.sel(latitude=lat, longitude=lon, method='nearest').values
                weather_at_point[param] = value
            except Exception as e:
                print(f"Error extracting {param} at {closest_timestamp}: {e}")
        else:
            weather_at_point[param] = np.nan

    return weather_at_point

def extract_weather_for_single_timestamp(weather_rasters, timestamp, lat, lon, available_times):
    """
    Extract weather for a single timestamp from the weather rasters.

    Parameters:
        weather_rasters (dict): Dictionary of weather raster data
        timestamp (datetime): Single timestamp
        lat (float): Latitude of the point
        lon (float): Longitude of the point
        available_times (list): List of available timestamps in rasters

    Returns:
        dict: Weather data at the point for the given timestamp
    """
    if timestamp not in available_times:
        print(f"Timestamp {timestamp} is not available in weather rasters")
        return {}

    weather_data = {}

    # For each parameter (e.g., temperature, wind speed, etc.), extract data
    for param in ['temperature', 'humidity', 'pressure', 'wind_speed', 'wind_deg']:
        try:
            # Get the raster for the parameter at the given timestamp
            raster = weather_rasters.get((param, timestamp))
            
            if raster is not None:
                # Get the nearest point in the raster to the given lat, lon
                lat_idx = np.abs(raster.coords['latitude'] - lat).argmin()
                lon_idx = np.abs(raster.coords['longitude'] - lon).argmin()
                
                # Extract the value at the nearest point
                weather_data[param] = raster.isel(latitude=lat_idx, longitude=lon_idx).values
            else:
                weather_data[param] = None
        except Exception as e:
            print(f"Error extracting {param} at {timestamp}: {e}")
            weather_data[param] = None

    return weather_data

def calculate_weather_gradient(weather_rasters, timestamp, lat1, lon1, lat2, lon2, distance=50):
    """Calculate weather gradients along the direction of travel"""
    if not weather_rasters:
        print("No weather rasters available for gradient calculation")
        return {}
        
    # Calculate bearing
    bearing = calculate_bearing(lat1, lon1, lat2, lon2)
    bearing_rad = np.radians(bearing)
    
    # Convert starting point to radians
    lat1_rad, lon1_rad = np.radians([lat1, lon1])
    
    # Convert distance to angular distance (radians)
    angular_distance = distance / 6371.0  # Earth radius in km
    
    # Calculate destination point
    lat2_rad = np.arcsin(
        np.sin(lat1_rad) * np.cos(angular_distance) + 
        np.cos(lat1_rad) * np.sin(angular_distance) * np.cos(bearing_rad)
    )
    
    lon2_rad = lon1_rad + np.arctan2(
        np.sin(bearing_rad) * np.sin(angular_distance) * np.cos(lat1_rad),
        np.cos(angular_distance) - np.sin(lat1_rad) * np.sin(lat2_rad)
    )
    
    # Convert back to degrees
    ahead_lat, ahead_lon = np.degrees([lat2_rad, lon2_rad])
    
    # Get weather at both points using the extract_weather_at_point function
    current_weather = extract_weather_at_point(weather_rasters, timestamp, lat1, lon1)
    ahead_weather = extract_weather_at_point(weather_rasters, timestamp, ahead_lat, ahead_lon)
    
    # Calculate gradients
    gradients = {}
    
    for param in current_weather.keys():
        if param in ahead_weather:
            current_val = current_weather[param]
            ahead_val = ahead_weather[param]
            
            if not np.isnan(current_val) and not np.isnan(ahead_val):
                gradients[f"{param}_gradient"] = (ahead_val - current_val) / distance
            else:
                gradients[f"{param}_gradient"] = np.nan
    
    return gradients


# ========================================================
# Feature Engineering with Weather Data
# ========================================================

def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculate the bearing between two points in degrees"""
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Calculate bearing
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    initial_bearing = np.arctan2(y, x)
    
    # Convert to degrees
    initial_bearing = np.degrees(initial_bearing)
    bearing = (initial_bearing + 360) % 360
    
    return bearing

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points in kilometers"""
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c  # Earth's radius in kilometers
    
    return km

def engineer_features(duck_trajectories, weather_rasters, station_tree, stations):
    if not duck_trajectories:
        print("No duck trajectories available for feature engineering")
        return pd.DataFrame()

    features_list = []
    for duck_id, trajectory in duck_trajectories.items():
        trajectory = trajectory.sort_values('timestamp')

        for i in range(len(trajectory) - 1):
            current_point = trajectory.iloc[i]
            next_point = trajectory.iloc[i + 1]

            current_lat = current_point['latitude']
            current_lon = current_point['longitude']
            next_lat = next_point['latitude']
            next_lon = next_point['longitude']
            current_time = current_point['timestamp']
            next_time = next_point['timestamp']

            distance = haversine_distance(current_lat, current_lon, next_lat, next_lon)
            bearing = calculate_bearing(current_lat, current_lon, next_lat, next_lon)
            time_diff = (next_time - current_time).total_seconds() / 3600  # hours
            speed = distance / time_diff if time_diff > 0 else 0

            # Extract weather features for the current point
            current_weather = extract_weather_at_point(weather_rasters, current_time, current_lat, current_lon)
            if current_weather:
                feature = {
                    'duck_id': duck_id,
                    'timestamp': current_time,
                    'lat': current_lat,
                    'lon': current_lon,
                    'next_lat': next_lat,
                    'next_lon': next_lon,
                    'distance': distance,
                    'bearing': bearing,
                    'calculated_speed': speed,
                    'temperature': current_weather.get('temperature', np.nan),
                    'humidity': current_weather.get('humidity', np.nan),
                    'wind_speed': current_weather.get('wind_speed', np.nan),
                    'wind_deg': current_weather.get('wind_deg', np.nan),
                    'pressure': current_weather.get('pressure', np.nan),
                    'accel_x': current_point.get('acceleration-raw-x', 0),
                    'accel_y': current_point.get('acceleration-raw-y', 0),
                    'accel_z': current_point.get('acceleration-raw-z', 0),
                    'mag_x': current_point.get('mag:magnetic-field-raw-x', 0),
                    'mag_y': current_point.get('mag:magnetic-field-raw-y', 0),
                    'mag_z': current_point.get('mag:magnetic-field-raw-z', 0),
                    'hour_of_day': current_time.hour,
                    'day_of_year': current_time.timetuple().tm_yday
                }
                features_list.append(feature)

    features_df = pd.DataFrame(features_list)
    return features_df

def prepare_lstm_sequences(features_df, sequence_length=24, target_cols=['next_lat', 'next_lon']):
    """Prepare sequence data for LSTM model"""
    if features_df.empty:
        print("Features DataFrame is empty, cannot prepare sequences")
        return np.array([]), np.array([]), []

    # Group by duck_id
    grouped = features_df.groupby('duck_id')
    X_sequences = []
    y_targets = []
    metadata = []  # Store information about each sequence

    feature_cols = [col for col in features_df.columns 
                   if col not in target_cols + ['duck_id', 'timestamp', 'next_lat', 'next_lon']]
    
    print(f"\n=== Preparing LSTM Sequences ===")
    print(f"Using sequence length: {sequence_length}")
    
    # For each duck's trajectory
    for duck_id, group in grouped:
        group = group.sort_values('timestamp')

        # Handle non-numeric and missing data in the features
        for col in feature_cols:
            group[col] = pd.to_numeric(group[col], errors='coerce')

        # Fill missing values
        group_filled = group[feature_cols].fillna(method='ffill').fillna(0)

        if len(group_filled) >= sequence_length + 1:
            # Normalize features
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaler.fit(group_filled)
            group_normalized = group.copy()
            group_normalized[feature_cols] = scaler.transform(group_filled)

            # Create sequences
            for i in range(len(group_normalized) - sequence_length):
                seq_df = group_normalized.iloc[i:i + sequence_length]
                target_df = group_normalized.iloc[i + sequence_length]

                X_seq = seq_df[feature_cols].values
                y_target = target_df[target_cols].values

                X_sequences.append(X_seq)
                y_targets.append(y_target)

                metadata.append({
                    'duck_id': str(duck_id),
                    'start_time': seq_df['timestamp'].iloc[0],
                    'end_time': seq_df['timestamp'].iloc[-1],
                    'target_time': target_df['timestamp'],
                    'last_lat': float(seq_df['lat'].iloc[-1]),
                    'last_lon': float(seq_df['lon'].iloc[-1])
                })

    if not X_sequences:
        print("No sequences could be created")
        return np.array([]), np.array([]), []

    X = np.array(X_sequences, dtype=np.float32)
    y = np.array(y_targets, dtype=np.float32)

    print(f"Created {len(X)} sequences")
    return X, y, metadata

def build_lstm_model(input_shape, output_dim=2):
    """Build and compile an LSTM model for duck migration prediction with additional features"""
    model = Sequential()
    
    # LSTM layers with dropout for regularization
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    
    # Dense output layers with batch normalization
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(output_dim, activation='linear'))  # Linear activation for regression
    
    # Compile the model with advanced metrics
    model.compile(
        optimizer='adam', 
        loss='mse', 
        metrics=['mae', 'mape']  # Add mean absolute percentage error 
    )
    
    print("\n=== Enhanced LSTM Model Built ===")
    model.summary()
    
    return model

def train_lstm_model(X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
    """Train the LSTM model with early stopping and learning rate scheduling"""
    # Define model input shape
    input_shape = (X_train.shape[1], X_train.shape[2])
    output_dim = y_train.shape[1]
    
    # Ensure data is float32 for TensorFlow
    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    y_val = y_val.astype(np.float32)
    
    print(f"X_train dtype: {X_train.dtype}, y_train dtype: {y_train.dtype}")
    print(f"X_val dtype: {X_val.dtype}, y_val dtype: {y_val.dtype}")
    
    # Build the model
    model = build_lstm_model(input_shape, output_dim)
    
    # Define callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss', 
            patience=15, 
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            'best_duck_model.h5', 
            save_best_only=True, 
            monitor='val_loss',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            verbose=1,
            min_lr=0.0001
        )
    ]
    
    # Train the model with error handling
    try:
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n=== Model Training Complete ===")
        print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
        print(f"Final validation MAE: {history.history['val_mae'][-1]:.4f}")
        
        # Plot training history
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Mean Absolute Error Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('duck_model_training_history.png')
        plt.close()
        
        return model, history
    
    except Exception as e:
        print(f"Error during model training: {e}")
        print("Detailed error information:")
        import traceback
        traceback.print_exc()
        
        # Return a simple model that can at least make predictions
        print("Creating fallback model...")
        simple_model = Sequential()
        simple_model.add(LSTM(32, input_shape=input_shape))
        simple_model.add(Dense(output_dim))
        simple_model.compile(optimizer='adam', loss='mse')
        
        # Create mock history
        history = {'loss': [0], 'val_loss': [0], 'mae': [0], 'val_mae': [0]}
        
        return simple_model, history

def predict_duck_migration(model, X_test, metadata_test):
    """Make predictions on test data and visualize results"""
    # Make predictions
    predictions = model.predict(X_test)
    
    # Create a DataFrame for visualization
    pred_df = pd.DataFrame({
        'duck_id': [m['duck_id'] for m in metadata_test],
        'timestamp': [m['target_time'] for m in metadata_test],
        'last_lat': [m['last_lat'] for m in metadata_test],
        'last_lon': [m['last_lon'] for m in metadata_test],
        'pred_lat': predictions[:, 0],
        'pred_lon': predictions[:, 1]
    })
    
    # Calculate error metrics
    pred_df['lat_error'] = np.abs(pred_df['pred_lat'] - pred_df['last_lat'])
    pred_df['lon_error'] = np.abs(pred_df['pred_lon'] - pred_df['last_lon'])
    
    # Calculate distance error in kilometers
    distance_errors = []
    for i, row in pred_df.iterrows():
        error_km = haversine_distance(
            row['last_lat'], row['last_lon'],
            row['pred_lat'], row['pred_lon']
        )
        distance_errors.append(error_km)
    
    pred_df['distance_error_km'] = distance_errors
    
    # Generate visualizations
    visualize_predictions(pred_df)
    
    print("\n=== Migration Predictions ===")
    print(f"Made {len(pred_df)} predictions")
    print(f"Average error distance: {pred_df['distance_error_km'].mean():.2f} km")
    print(f"Median error distance: {pred_df['distance_error_km'].median():.2f} km")
    print(f"Max error distance: {pred_df['distance_error_km'].max():.2f} km")
    
    return pred_df

def visualize_extended_forecasts(forecasts_df):
    """Visualize extended forecasts on a map"""
    if forecasts_df.empty:
        return
        
    # Sample down if we have too many forecasts
    if len(forecasts_df) > 100:
        sample_df = forecasts_df.sample(n=100, random_state=42)
    else:
        sample_df = forecasts_df
    
    # Get unique duck IDs
    duck_ids = sample_df['duck_id'].unique()
    
    # Create a map
    plt.figure(figsize=(12, 10))
    
    # Use a different color for each duck
    colors = plt.cm.rainbow(np.linspace(0, 1, len(duck_ids)))
    color_map = dict(zip(duck_ids, colors))
    
    # Plot each duck's forecast path
    for duck_id in duck_ids:
        duck_data = sample_df[sample_df['duck_id'] == duck_id]
        
        # Sort by forecast day
        duck_data = duck_data.sort_values('forecast_day')
        
        # Get coordinates
        lats = [duck_data.iloc[0]['start_lat']] + duck_data['forecast_lat'].tolist()
        lons = [duck_data.iloc[0]['start_lon']] + duck_data['forecast_lon'].tolist()
        
        # Plot path
        plt.plot(lons, lats, '-o', color=color_map[duck_id], alpha=0.7, 
                 label=f"Duck {duck_id}" if duck_id == duck_ids[0] else "")
                 
        # Add day markers for first duck only
        if duck_id == duck_ids[0]:
            for i, (lat, lon) in enumerate(zip(lats[1:], lons[1:])):
                plt.text(lon, lat, f"Day {i+1}", fontsize=8)
    
    # Add legend with one example
    plt.legend(loc='best')
    
    plt.title('Duck Migration Extended Forecasts')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    plt.savefig('extended_forecasts_map.png')
    plt.close()
    
    print("Extended forecasts visualization saved")


def create_weather_raster_timeseries(weather_data, duck_trajectories, flyway_bbox, resolution=0.1):
    """
    Create a time series of weather rasters aligned with duck migration timestamps
    
    Parameters:
    - weather_data: List of dictionaries with weather observations from OpenWeather (must include 'longitude', 'latitude', and weather parameters)
    - duck_trajectories: Dictionary of GeoDataFrames with duck migration data (must include timestamps)
    - flyway_bbox: Bounding box [lon_min, lat_min, lon_max, lat_max]
    - resolution: Spatial resolution in degrees
    
    Returns:
    - Dictionary of xarray DataArrays, keyed by parameter and timestamp
    """
    # Check if weather_data contains required parameters (including coordinates)
    required_keys = ['temperature', 'humidity', 'pressure', 'wind_speed', 'wind_deg', 'longitude', 'latitude']
    
    # Ensure weather_data is a list of dictionaries
    if not isinstance(weather_data, list) or not all(isinstance(item, dict) for item in weather_data):
        print("Error: weather_data must be a list of dictionaries")
        return {}

    if not all(key in weather_data[0] for key in required_keys):  # Check only first item for required keys
        print("Error: Invalid weather data or missing required parameters (e.g., temperature, longitude, latitude)")
        return {}

    # Extract latitude, longitude and weather parameters from the list of dictionaries
    longitude = [w['longitude'] for w in weather_data]
    latitude = [w['latitude'] for w in weather_data]
    
    # Extract unique timestamps from duck trajectories
    unique_time_duck = get_all_timestamps(duck_trajectories)
    
    # If no duck timestamps, use weather data timestamps (if available)
    if not unique_time_duck and 'time' in weather_data:
        unique_time_duck = sorted(weather_data['time'].unique())
    
    # Limit to a few timestamps for demonstration
    if len(unique_time_duck) > 5:
        unique_time_duck = unique_time_duck[:5]
        
    if not unique_time_duck:
        print("No timestamps available for raster creation")
        return {}
    
    # Define parameters to interpolate (use what's available)
    params = ['temperature', 'humidity', 'pressure', 'wind_speed', 'wind_deg']
    
    # Define grid
    lon_min, lat_min, lon_max, lat_max = flyway_bbox
    grid_lon = np.arange(lon_min, lon_max, resolution)
    grid_lat = np.arange(lat_min, lat_max, resolution)
    grid_lon_mesh, grid_lat_mesh = np.meshgrid(grid_lon, grid_lat)
    
    # Initialize dictionary to store DataArrays
    raster_dict = {}
    
    # For each duck migration timestamp
    for t in unique_time_duck:
        # For each parameter
        for param in params:
            try:
                # Prepare the weather data for interpolation
                param_data = {'longitude': longitude, 'latitude': latitude, param: [w[param] for w in weather_data]}
                
                # Skip interpolation if there are fewer than 4 points
                if len(param_data['longitude']) < 4:
                    print(f"Skipping interpolation for {param} at {t} due to insufficient data points")
                    continue
                
                # Interpolate using griddata
                grid_values = griddata(
                    (param_data['longitude'], param_data['latitude']), param_data[param], 
                    (grid_lon_mesh, grid_lat_mesh),
                    method='linear', fill_value=np.nan
                )
                
                # Create DataArray
                da = xr.DataArray(
                    grid_values,
                    dims=['latitude', 'longitude'],
                    coords={
                        'latitude': grid_lat,
                        'longitude': grid_lon,
                        'timestamp': t
                    },
                    name=param
                )
                
                # Store in dictionary
                raster_dict[(param, t)] = da
                
            except Exception as e:
                print(f"Error interpolating {param} at {t}: {e}")
    
    print("\n=== Weather Rasters Created ===")
    print(f"Created {len(raster_dict)} raster layers")
    print(f"Parameters: {set(k[0] for k in raster_dict.keys())}")
    print(f"Time steps: {len(set(k[1] for k in raster_dict.keys()))}")
    
    return raster_dict


def visualize_predictions(pred_df):
    """Visualize prediction results"""
    if pred_df.empty:
        print("No predictions to visualize")
        return
    
    # Sample a subset for visualization if too many
    if len(pred_df) > 50:
        viz_df = pred_df.sample(50, random_state=42)
    else:
        viz_df = pred_df
    
    # Create a map of actual vs predicted locations
    plt.figure(figsize=(12, 10))
    
    # Plot actual locations
    plt.scatter(
        viz_df['last_lon'], 
        viz_df['last_lat'], 
        c='blue', 
        label='Actual Location',
        alpha=0.7
    )
    
    # Plot predicted locations
    plt.scatter(
        viz_df['pred_lon'], 
        viz_df['pred_lat'], 
        c='red', 
        label='Predicted Location',
        alpha=0.7
    )
    
    # Connect actual and predicted with lines
    for _, row in viz_df.iterrows():
        plt.plot(
            [row['last_lon'], row['pred_lon']], 
            [row['last_lat'], row['pred_lat']], 
            'k-', 
            alpha=0.3
        )
    
    plt.title('Duck Migration: Actual vs Predicted Locations')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.grid(True)
    plt.savefig('duck_prediction_map.png')
    plt.close()
    
    # Add error histogram
    plt.figure(figsize=(10, 6))
    plt.hist(pred_df['distance_error_km'], bins=20, alpha=0.7)
    plt.axvline(pred_df['distance_error_km'].mean(), color='r', linestyle='--', 
                label=f'Mean: {pred_df["distance_error_km"].mean():.2f} km')
    plt.axvline(pred_df['distance_error_km'].median(), color='g', linestyle='--', 
                label=f'Median: {pred_df["distance_error_km"].median():.2f} km')
    
    plt.title('Prediction Error Distribution')
    plt.xlabel('Error Distance (km)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.savefig('duck_prediction_error_distribution.png')
    plt.close()
    
    print("\n=== Prediction Visualizations Created ===")
    print("Saved visualizations to disk")

def analyze_feature_importance(model, feature_names):
    """Analyze feature importance by perturbing input features"""
    # This is a simple implementation that supports Sequential models
    # For more complex models, consider using SHAP or other explainability tools
    
    print("\n=== Feature Importance Analysis ===")
    print("Method: Input Perturbation (basic approach)")
    
    # Get model weights for the first layer
    first_layer_weights = model.layers[0].get_weights()[0]
    
    # Sum of absolute weights for each feature
    importance = np.sum(np.abs(first_layer_weights), axis=1)
    
    # Normalize importance
    importance = importance / np.sum(importance)
    
    # Create feature importance dataframe
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Print top features
    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10))
    
    # Visualize
    plt.figure(figsize=(12, 8))
    plt.barh(importance_df['Feature'].head(15), importance_df['Importance'].head(15))
    plt.xlabel('Relative Importance')
    plt.title('Feature Importance (Top 15 Features)')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    return importance_df

def analyze_migration_patterns(pred_df, duck_trajectories):
    """Analyze migration patterns from predictions and actual data"""
    if pred_df.empty:
        print("No predictions available for migration pattern analysis")
        return pd.DataFrame()
        
    print("\n=== Migration Pattern Analysis ===")
    
    # Group predictions by duck_id
    duck_predictions = pred_df.groupby('duck_id')
    
    # Analysis results
    analysis_results = []
    
    for duck_id, group in duck_predictions:
        # Skip if too few predictions
        if len(group) < 5:
            continue
        
        # Calculate average movement direction (bearing)
        bearings = []
        for i in range(len(group) - 1):
            lat1 = group.iloc[i]['last_lat']
            lon1 = group.iloc[i]['last_lon']
            lat2 = group.iloc[i+1]['last_lat']
            lon2 = group.iloc[i+1]['last_lon']
            
            bearing = calculate_bearing(lat1, lon1, lat2, lon2)
            bearings.append(bearing)
        
        avg_bearing = np.mean(bearings) if bearings else np.nan
        
        # Calculate average speed
        speeds = []
        for i in range(len(group) - 1):
            lat1 = group.iloc[i]['last_lat']
            lon1 = group.iloc[i]['last_lon']
            lat2 = group.iloc[i+1]['last_lat']
            lon2 = group.iloc[i+1]['last_lon']
            
            time1 = group.iloc[i]['timestamp']
            time2 = group.iloc[i+1]['timestamp']
            
            distance = haversine_distance(lat1, lon1, lat2, lon2)
            time_diff = (time2 - time1).total_seconds() / 3600  # hours
            
            if time_diff > 0:
                speed = distance / time_diff
                speeds.append(speed)
        
        avg_speed = np.mean(speeds) if speeds else np.nan
        
        # Calculate total distance
        total_distance = sum(
            haversine_distance(
                group.iloc[i]['last_lat'], 
                group.iloc[i]['last_lon'],
                group.iloc[i+1]['last_lat'], 
                group.iloc[i+1]['last_lon']
            )
            for i in range(len(group) - 1)
        )
        
        # Store results
        analysis_results.append({
            'duck_id': duck_id,
            'avg_bearing': avg_bearing,
            'avg_speed_kmh': avg_speed,
            'total_distance_km': total_distance,
            'prediction_count': len(group)
        })
    
    if not analysis_results:
        print("No migration patterns could be analyzed")
        return pd.DataFrame()
        
    # Convert to DataFrame
    analysis_df = pd.DataFrame(analysis_results)
    
    # Visualize migration directions
    plt.figure(figsize=(10, 10), dpi=100)
    ax = plt.subplot(111, projection='polar')
    
    # Convert bearings to radians for polar plot
    bearings_rad = np.radians(analysis_df['avg_bearing'])
    
    # Use speed for the length of the arrows
    speeds = analysis_df['avg_speed_kmh']
    normalized_speeds = speeds / speeds.max() if not speeds.empty and speeds.max() > 0 else speeds
    
    # Plot arrows
    for i, (bear, speed) in enumerate(zip(bearings_rad, normalized_speeds)):
        if not np.isnan(bear) and not np.isnan(speed):
            ax.arrow(bear, 0, 0, speed * 0.8, alpha=0.5, width=0.05,
                    head_width=0.1, head_length=0.1, color='blue')
    
    # Set the direction labels
    ax.set_xticklabels(['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE'])
    
    plt.title('Duck Migration Directions')
    plt.tight_layout()
    plt.savefig('migration_directions.png')
    plt.close()
    
    # Summarize the results
    print("\nMigration Pattern Summary:")
    print(f"Analysis performed on {len(analysis_df)} ducks")
    if not analysis_df.empty:
        print(f"Average migration speed: {analysis_df['avg_speed_kmh'].mean():.2f} km/h")
        print(f"Average migration distance: {analysis_df['total_distance_km'].mean():.2f} km")
    
    return analysis_df

def generate_extended_forecasts(model, X_test, metadata_test, days_ahead=5):
    """
    Generate extended forecasts for multiple days ahead
    
    Parameters:
        model: Trained LSTM model
        X_test: Test data sequences
        metadata_test: Metadata for test sequences
        days_ahead: Number of days to forecast ahead
        
    Returns:
        DataFrame with extended forecasts
    """
    if len(X_test) == 0:
        print("No test data for extended forecasts")
        return pd.DataFrame()
        
    print(f"Generating {days_ahead}-day forecasts for {len(X_test)} test sequences")
    
    # List to store all forecasts
    all_forecasts = []
    
    # For each test sequence
    for i in range(len(X_test)):
        # Get initial sequence and metadata
        sequence = X_test[i:i+1].copy()  # Keep batch dimension
        meta = metadata_test[i]
        
        # Get last known location
        current_lat = meta['last_lat']
        current_lon = meta['last_lon']
        base_time = meta['target_time']
        
        # Make forecasts for each day ahead
        for day in range(1, days_ahead + 1):
            # Predict the next position
            pred = model.predict(sequence)
            
            # Extract predicted lat/lon
            pred_lat = pred[0, 0]
            pred_lon = pred[0, 1]
            
            # Calculate distance moved
            distance = haversine_distance(current_lat, current_lon, pred_lat, pred_lon)
            
            # Calculate bearing
            bearing = calculate_bearing(current_lat, current_lon, pred_lat, pred_lon)
            
            # Store the forecast
            forecast = {
                'duck_id': meta['duck_id'],
                'base_timestamp': base_time,
                'forecast_day': day,
                'forecast_timestamp': base_time + timedelta(days=day),
                'start_lat': current_lat,
                'start_lon': current_lon,
                'forecast_lat': pred_lat,
                'forecast_lon': pred_lon,
                'distance_km': distance,
                'bearing': bearing
            }
            
            all_forecasts.append(forecast)
            
            # Update current position for next forecast
            current_lat = pred_lat
            current_lon = pred_lon
            
    if not all_forecasts:
        print("No forecasts generated")
        return pd.DataFrame()
        
    # Convert to DataFrame
    forecasts_df = pd.DataFrame(all_forecasts)
    
    print(f"Generated {len(forecasts_df)} forecast points")
    
    # Visualize forecasts on a map
    visualize_extended_forecasts(forecasts_df)
    
    return forecasts_df

def visualize_extended_forecasts(forecasts_df):
    """Visualize extended forecasts on a map"""
    if forecasts_df.empty:
        return
        
    # Sample down if we have too many forecasts
    if len(forecasts_df) > 100:
        sample_df = forecasts_df.sample(n=100, random_state=42)
    else:
        sample_df = forecasts_df
    
    # Get unique duck IDs
    duck_ids = sample_df['duck_id'].unique()
    
    # Create a map
    plt.figure(figsize=(12, 10))
    
    # Use a different color for each duck
    colors = plt.cm.rainbow(np.linspace(0, 1, len(duck_ids)))
    color_map = dict(zip(duck_ids, colors))
    
    # Plot each duck's forecast path
    for duck_id in duck_ids:
        duck_data = sample_df[sample_df['duck_id'] == duck_id]
        
        # Sort by forecast day
        duck_data = duck_data.sort_values('forecast_day')
        
        # Get coordinates
        lats = [duck_data.iloc[0]['start_lat']] + duck_data['forecast_lat'].tolist()
        lons = [duck_data.iloc[0]['start_lon']] + duck_data['forecast_lon'].tolist()
        
        # Plot path
        plt.plot(lons, lats, '-o', color=color_map[duck_id], alpha=0.7, 
                 label=f"Duck {duck_id}" if duck_id == duck_ids[0] else "")
                 
        # Add day markers for first duck only
        if duck_id == duck_ids[0]:
            for i, (lat, lon) in enumerate(zip(lats[1:], lons[1:])):
                plt.text(lon, lat, f"Day {i+1}", fontsize=8)
    
    # Add legend with one example
    plt.legend(loc='best')
    
    plt.title('Duck Migration Extended Forecasts')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    plt.savefig('extended_forecasts_map.png')
    plt.close()
    
    print("Extended forecasts visualization saved")

# ========================================================
# Main Execution
# ========================================================

def main():
    """Main execution function"""
    print("\n===== Starting Duck Migration Prediction System =====\n")
    
    # Load duck data
    duck_trajectories, duck_gdf = load_duck_data()
    if duck_trajectories is None:
        print("Error loading duck data. Exiting.")
        return
    
    # Define flyway region
    flyway_bbox = get_flyway_region()
    
    # Get all timestamps from duck trajectories
    all_timestamps = get_all_timestamps(duck_trajectories)
    if not all_timestamps:
        print("No timestamps found in duck data. Using current time range.")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        all_timestamps = [start_date, end_date]
    else:
        start_date = min(all_timestamps)
        end_date = max(all_timestamps)
    
    # Sample points for testing
    lat = 39.0
    lon = -90.0
    
    # Fetch OpenWeather forecast for a sample point
    try:
        print(f"Fetching OpenWeather forecast for point ({lat}, {lon})...")
        forecast_json = fetch_weather_forecast_for_point(lat, lon)  # Updated call to OpenWeather
        print("Successfully fetched OpenWeather forecast")
    except Exception as e:
        print(f"Error fetching OpenWeather forecast: {e}")
        forecast_json = None
    
    # Fetch weather data for stations
    try:
        print("Fetching weather data for stations...")
        weather_data = fetch_weather_data([{'id': 'test1', 'latitude': lat, 'longitude': lon}], start_date, end_date)
        if weather_data:
            print("Successfully fetched weather data")
        else:
            print("No weather data found")
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        weather_data = []
    
    # Proceed with feature engineering and LSTM model training if we have the necessary data
    if weather_data:
        engineered_features = engineer_features(duck_trajectories, weather_data, None, None)
        print(f"Features created: {len(engineered_features)}")
    
    print("\n===== Duck Migration Prediction System Complete =====")
