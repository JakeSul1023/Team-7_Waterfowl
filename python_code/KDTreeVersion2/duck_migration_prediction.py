print("------------------------------------------------------------")
print("------------------------------------------------------------------------------------------")
print("----------------------------------------------------------------------------------------------------------------------------------------")

import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import BallTree
from meteostat import Point, Daily, Stations, Hourly
from datetime import datetime, timedelta
from shapely.geometry import Point as ShapelyPoint
from scipy.interpolate import griddata
import os
import json
import requests
import rioxarray
import math
import pyproj
from pyproj import Transformer
from functools import partial
import dask.dataframe as dd
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

print("---------------------------------------------------------------------------------------------------------------------------------")
print("----------------------------------------------------------------------------------------------------")
print("------------------------------------------------------------------------")

print("Libraries installed")

# ========================================================
# Data Loading and Preprocessing
# ========================================================

def load_duck_data(filepath="ShortTermSetData(Aug-Sept).csv"):
    """Load and preprocess duck telemetry data from CSV file"""
    # Read CSV file
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
    
    # Display column names to verify data structure
    print(f"Columns in the dataset: {duck_df.columns.tolist()}")
    
    # Convert timestamp to datetime
    duck_df['timestamp'] = pd.to_datetime(duck_df['timestamp'])
    
    # Create GeoDataFrame for spatial operations
    geometry = [ShapelyPoint(xy) for xy in zip(duck_df['location-long'], duck_df['location-lat'])]
    duck_gdf = gpd.GeoDataFrame(duck_df, geometry=geometry, crs="EPSG:4326")
    
    # Group by individual duck IDs
    duck_trajectories = {id: group for id, group in duck_gdf.groupby('individual-local-identifier')}
    
    print("\n=== Duck Data Loaded ===")
    print(f"Total ducks: {len(duck_trajectories)}")
    print(f"Total observations: {len(duck_df)}")
    print(f"Sample duck trajectory:\n{next(iter(duck_trajectories.values())).head()}")
    
    return duck_trajectories, duck_gdf



def get_flyway_region():
    """Define the Mississippi Flyway region"""
    # Approximate bounding box of Mississippi Flyway
    # (longitude min, latitude min, longitude max, latitude max)
    bbox = [-98.0, 28.0, -82.0, 49.0]
    
    print("\n=== Flyway Region ===")
    print(f"Bounding box: {bbox}")
    
    return bbox


def get_all_timestamps(duck_trajectories):
    """Extract all unique timestamps from duck trajectories"""
    all_times = []
    for duck_id, trajectory in duck_trajectories.items():
        all_times.extend(trajectory['timestamp'].tolist())
    
    unique_times = sorted(set(all_times))
    
    print("\n=== Timestamp Analysis ===")
    print(f"Total unique timestamps: {len(unique_times)}")
    print(f"Date range: {unique_times[0]} to {unique_times[-1]}")
    
    return unique_times


# ========================================================
# Weather Data Collection
# ========================================================

def fetch_noaa_forecast_for_point(lat, lon, token):
    """
    Fetch forecast JSON from NOAA NWS API for a given point.
    
    Parameters:
      - lat, lon: Latitude and Longitude.
      - token: NOAA API token.
      
    Returns:
      - JSON object with forecast data.
    """
    # Query the points endpoint to get the forecast URL.
    points_url = f"https://api.weather.gov/points/{lat},{lon}"
    headers = {
        "User-Agent": "DuckMigrationApp/1.0",
        "token": token
    }
    response = requests.get(points_url, headers=headers)
    response.raise_for_status()  # Raise error if request fails
    data = response.json()
    
    forecast_url = data.get("properties", {}).get("forecast")
    if not forecast_url:
        raise ValueError("Forecast URL not found in NOAA response.")
    
    # Now fetch the forecast data using the forecast URL.
    forecast_response = requests.get(forecast_url, headers=headers)
    forecast_response.raise_for_status()
    forecast_json = forecast_response.json()
    
    print("Forecast JSON Fetched")
    return forecast_json



def fetch_weather_data(stations, start_date, end_date):
    """
    Fetch hourly weather data for multiple stations and date range.
    
    Parameters:
      - stations: list of station dictionaries with id, latitude, longitude 
                  OR a class/type that can be instantiated to yield either a list
                  or an instance with a 'stations' attribute.
      - start_date: datetime object for start of data collection.
      - end_date: datetime object for end of data collection.
      
    Returns:
      - DataFrame with hourly weather data.
    """
    # If 'stations' is a type, try to instantiate it and extract the list.
    if isinstance(stations, type):
        try:
            stations_instance = stations()  # Attempt to instantiate
        except Exception as e:
            raise ValueError("Error instantiating stations parameter: " + str(e))
        # If the instance itself is a list, use it.
        if isinstance(stations_instance, list):
            stations = stations_instance
        # Otherwise, if it has a 'stations' attribute, use that.
        elif hasattr(stations_instance, 'stations'):
            stations = stations_instance.stations
        else:
            raise ValueError(
                "The provided stations parameter is a type, but its instance is neither a list "
                "nor does it have a 'stations' attribute. Please provide a list of station dictionaries."
            )

    # Ensure that stations is a list. If not, try to convert it to a list.
    if not isinstance(stations, list):
        try:
            stations = list(stations)
        except Exception as e:
            raise ValueError("The provided stations parameter is not a list and cannot be converted to one: " + str(e))

    all_weather_data = []

    # Limit to first 3 stations for demonstration
    sample_stations = stations[:3] if len(stations) > 3 else stations

    for station in sample_stations:
        try:
            # Validate station structure
            if not isinstance(station, dict) or 'id' not in station:
                print(f"Skipping station due to missing 'id' key: {station}")
                continue

            # Create Meteostat Point for the station
            location = Point(
                station.get('latitude', 0),
                station.get('longitude', 0),
                station.get('elevation', 0)  # Default value to avoid KeyError
            )

            # Fetch hourly weather data
            data = Hourly(location, start_date, end_date).fetch()

            # Skip if no data
            if data.empty:
                print(f"No data available for station {station['id']}")
                continue

            # Add station info
            data['station_id'] = station['id']
            data['station_name'] = station.get('name', 'Unknown')
            data['latitude'] = station.get('latitude', None)
            data['longitude'] = station.get('longitude', None)

            all_weather_data.append(data)
            print(f"Fetched data for station {station['id']} - {station.get('name', 'Unknown')}")

        except Exception as e:
            station_id = station.get('id', 'Unknown')
            print(f"Error fetching data for station {station_id}: {e}")

    # Combine all station data
    if all_weather_data:
        combined_data = pd.concat(all_weather_data, ignore_index=True)
        print("\n=== Weather Data Fetched ===")
        print(f"Total records: {len(combined_data)}")
        print(f"Date range: {combined_data.index.min()} to {combined_data.index.max()}")
        print(f"Parameters: {list(combined_data.columns)}")
        return combined_data
    else:
        print("No weather data could be fetched.")
        return pd.DataFrame()


# Modify fetch_weather_forecast to be more resilient
def fetch_weather_forecast_robust(bbox, resolution=2.0, forecast_dates=None, token="your_token"):
    """More robust version of fetch_weather_forecast that handles errors gracefully"""
    # Unpack bounding box
    lon_min, lat_min, lon_max, lat_max = bbox
    
    # Create grid points with wider spacing
    lats = np.arange(lat_min, lat_max, resolution)
    lons = np.arange(lon_min, lon_max, resolution)
    
    records = []
    
    # Process forecast dates
    if forecast_dates is not None:
        forecast_dates_set = set(d.date() for d in forecast_dates)
    else:
        forecast_dates_set = None
    
    # Add delay between requests to avoid rate limiting
    request_delay = 1  # seconds
    
    # Loop over grid points
    for lat in lats:
        for lon in lons:
            try:
                print(f"Fetching forecast for point ({lat}, {lon})...")
                forecast_json = fetch_noaa_forecast_for_point(lat, lon, token)
                #time.sleep(request_delay)  # Add delay between requests
                
                periods = forecast_json.get("properties", {}).get("periods", [])
                if not periods:
                    print(f"No forecast periods returned for point ({lat}, {lon}).")
                    continue
                
                for period in periods:
                    try:
                        period_start = pd.to_datetime(period["startTime"])
                        if forecast_dates_set is not None:
                            if period_start.date() not in forecast_dates_set:
                                continue
                        record = {
                            "lat": lat,
                            "lon": lon,
                            "startTime": period_start,
                            "endTime": pd.to_datetime(period["endTime"]),
                            "temperature": period["temperature"],
                            "temperatureUnit": period["temperatureUnit"],
                            "windSpeed": period["windSpeed"],
                            "windDirection": period["windDirection"],
                            "shortForecast": period["shortForecast"]
                        }
                        records.append(record)
                    except Exception as e:
                        print(f"Error processing forecast period: {e}")
            except Exception as e:
                print(f"Error fetching forecast for point ({lat}, {lon}): {e}")
                #time.sleep(request_delay)  # Still add delay after errors
    
    if records:
        forecast_df = pd.DataFrame(records)
        print(f"\n=== NOAA Forecast Data Summary ===")
        print(f"Total forecast records: {len(forecast_df)}")
        print(f"Sample forecast data:\n{forecast_df.head(10)}")
        return forecast_df
    else:
        print("No forecast data could be generated.")
        return pd.DataFrame()

# ========================================================
# Spatial Data Processing
# ========================================================


#now good till here and down
# ===========================================================================================================
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------
# ===========================================================================================================

def create_weather_raster_timeseries(weather_data, load_duck_data, flyway_bbox, resolution=0.1):
    """
    Create a time series of weather rasters aligned with duck migration timestamps
    
    Parameters:
    - weather_data: DataFrame with weather observations (without timestamps)
    - duck_data: DataFrame with duck migration data (with timestamps)
    - bbox: Bounding box [lon_min, lat_min, lon_max, lat_max]
    - resolution: Spatial resolution in degrees
    
    Returns:
    - Dictionary of xarray DataArrays, keyed by parameter and timestamp
    """
    if weather_data.empty:
        raise ValueError("Weather data is empty")
    
    # Extract unique timestamps from duck data
    # Assuming duck_data has a 'timestamp' column
    if 'timestamp' not in load_duck_data.column:
        raise ValueError("Duck data must have a 'timestamp' column")
    
    # Get unique timestamps from duck data
    unique_times = load_duck_data['timestamp'].sort_values().unique()
    
    # For demonstration, limit to a few timestamps
    if len(unique_times) > 5:
        unique_times = unique_times[:5]
    
    # Define parameters to interpolate
    params = ['temp', 'rhum', 'prcp', 'wspd', 'wdir', 'pres']
    
    # Define grid
    lon_min, lat_min, lon_max, lat_max = bbox
    grid_lon = np.arange(lon_min, lon_max, resolution)
    grid_lat = np.arange(lat_min, lat_max, resolution)
    grid_lon_mesh, grid_lat_mesh = np.meshgrid(grid_lon, grid_lat)
    
    # Initialize dictionary to store DataArrays
    raster_dict = {}
    
    # For each duck migration timestamp
    for t in unique_times:
        # For each parameter
        for param in params:
            if param not in weather_data.columns:
                continue
            
            # Drop NaN values
            param_data = weather_data.dropna(subset=[param])
            
            if len(param_data) < 3:
                continue  # Skip if not enough valid points
            
            # Points and values for interpolation
            points = param_data[['longitude', 'latitude']].values
            values = param_data[param].values
            
            try:
                # Interpolate using IDW or another method
                from scipy.interpolate import griddata
                import numpy as np
                import xarray as xr
                
                grid_values = griddata(
                    points, values, (grid_lon_mesh, grid_lat_mesh),
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
    print(f"Time steps: {len(unique_times)}")
    
    return raster_dict


def build_kdtree_for_stations(stations):
    """Build a KD-tree for efficient nearest station lookup"""
    # Extract coordinates
    coords = np.array([[s['latitude'], s['longitude']] for s in stations])
    
    # Convert lat/lon to radians for haversine distance
    coords_rad = np.radians(coords)
    
    # Build KD-tree
    tree = BallTree(coords_rad, metric='haversine')
    
    print("\n=== KDTree Built ===")
    print(f"Tree built for {len(stations)} weather stations")
    
    return tree, coords



def find_nearest_stations(lat, lon, tree, stations, k=3):
    """Find k nearest weather stations to a location"""
    # Convert to radians
    lat_rad, lon_rad = np.radians([lat, lon])
    
    # Query the tree
    distances, indices = tree.query([[lat_rad, lon_rad]], k=k)
    
    # Convert distances to kilometers (Earth radius ≈ 6371 km)
    distances = distances[0] * 6371.0
    
    # Get station info
    nearest_stations = [
        {**stations[idx], 'distance': dist}
        for idx, dist in zip(indices[0], distances)
    ]
    
    print("\n=== Nearest Stations Found ===")
    print(f"Query location: ({lat}, {lon})")
    print(f"Found {len(nearest_stations)} nearest stations:")
    for i, station in enumerate(nearest_stations):
        print(f"  {i+1}: {station['name']} - {station['distance']:.2f} km")
    
    return nearest_stations



# ========================================================
# Feature Engineering
# ========================================================

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
    
    distance = km
    
    print("\n=== Haversine Distance Calculated ===")
    print(f"Point 1: ({lat1:.6f}°, {lon1:.6f}°)")
    print(f"Point 2: ({lat2:.6f}°, {lon2:.6f}°)")
    print(f"Distance: {distance:.2f} km")
    
    return distance


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
    
    print("\n=== Bearing Calculated ===")
    print(f"Point 1: ({np.degrees(lat1):.6f}°, {np.degrees(lon1):.6f}°)")
    print(f"Point 2: ({np.degrees(lat2):.6f}°, {np.degrees(lon2):.6f}°)")
    print(f"Bearing: {bearing:.2f}°")
    
    return bearing


def extract_weather_at_point(weather_rasters, timestamp, lat, lon):
    """Extract weather variables at a specific point and time"""
    # Find the closest timestamp in the rasters
    available_times = sorted(set(t for _, t in weather_rasters.keys()))
    if not available_times:
        print("No weather rasters available")
        return {}
        
    closest_time = min(available_times, key=lambda t: abs(t - timestamp))
    
    # Initialize results
    weather_at_point = {}
    
    # Extract each weather parameter
    for param in ['temp', 'rhum', 'prcp', 'wspd', 'wdir', 'pres']:
        try:
            raster = weather_rasters.get((param, closest_time))
            if raster is not None:
                # Get nearest grid cell value
                weather_at_point[param] = float(raster.sel(
                    latitude=lat, longitude=lon, method='nearest'
                ).values)
            else:
                weather_at_point[param] = np.nan
        except Exception as e:
            print(f"Error extracting {param}: {e}")
            weather_at_point[param] = np.nan
    
    print("\n=== Weather at Point Extracted ===")
    print(f"Location: ({lat:.6f}°, {lon:.6f}°)")
    print(f"Timestamp: {timestamp}")
    print(f"Closest time available: {closest_time}")
    print(f"Extracted weather data: {weather_at_point}")
    
    return weather_at_point


def calculate_weather_gradient(weather_rasters, timestamp, lat1, lon1, lat2, lon2, distance=50):
    """Calculate weather gradients along the direction of travel"""
    # Calculate bearing
    bearing = calculate_bearing(lat1, lon1, lat2, lon2)
    bearing_rad = np.radians(bearing)
    
    # Calculate a point ahead in the direction of travel
    # Uses the haversine formula in reverse
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    
    angular_distance = distance / 6371.0  # Convert km to radians
    
    lat2_rad = np.arcsin(
        np.sin(lat1_rad) * np.cos(angular_distance) + 
        np.cos(lat1_rad) * np.sin(angular_distance) * np.cos(bearing_rad)
    )
    
    lon2_rad = lon1_rad + np.arctan2(
        np.sin(bearing_rad) * np.sin(angular_distance) * np.cos(lat1_rad),
        np.cos(angular_distance) - np.sin(lat1_rad) * np.sin(lat2_rad)
    )
    
    ahead_lat = np.degrees(lat2_rad)
    ahead_lon = np.degrees(lon2_rad)
    
    # Get weather at current point and ahead point
    current_weather = extract_weather_at_point(weather_rasters, timestamp, lat1, lon1)
    ahead_weather = extract_weather_at_point(weather_rasters, timestamp, ahead_lat, ahead_lon)
    
    # Calculate gradients
    gradients = {}
    for param in current_weather:
        if not np.isnan(current_weather[param]) and not np.isnan(ahead_weather[param]):
            gradients[f"{param}_gradient"] = (ahead_weather[param] - current_weather[param]) / distance
        else:
            gradients[f"{param}_gradient"] = np.nan
    
    print("\n=== Weather Gradients Calculated ===")
    print(f"Start point: ({lat1:.6f}°, {lon1:.6f}°)")
    print(f"End point: ({lat2:.6f}°, {lon2:.6f}°)")
    print(f"Point ahead at {distance} km: ({ahead_lat:.6f}°, {ahead_lon:.6f}°)")
    print(f"Weather gradients: {gradients}")
    
    return gradients


def calculate_wind_assistance(wspd, wdir, travel_bearing):
    """
    Calculate wind assistance/resistance for duck movement
    
    Positive values indicate tailwind (assistance)
    Negative values indicate headwind (resistance)
    """
    # Calculate the angular difference between wind direction and travel bearing
    # Convert wind direction from meteorological to mathematical convention
    wdir_math = (270 - wdir) % 360
    
    # Calculate the bearing difference
    angle_diff = abs(((wdir_math - travel_bearing) + 180) % 360 - 180)
    
    # Calculate the wind component along the travel direction
    # cos(0°) = 1 (perfect tailwind), cos(180°) = -1 (perfect headwind)
    wind_component = wspd * np.cos(np.radians(angle_diff))
    
    print("\n=== Wind Assistance Calculated ===")
    print(f"Wind speed: {wspd:.1f} units")
    print(f"Wind direction: {wdir:.1f}°")
    print(f"Travel bearing: {travel_bearing:.1f}°")
    print(f"Wind assistance: {wind_component:.2f} units")
    print(f"Interpretation: {'Tailwind (helping)' if wind_component > 0 else 'Headwind (opposing)'}")
    
    return wind_component



def engineer_features(duck_trajectories, weather_rasters, station_tree, stations):
    """Create features for LSTM by combining duck trajectories with weather data"""
    # For demonstration, use a small subset of the data
    sample_duck_id = next(iter(duck_trajectories.keys()))
    sample_trajectory = duck_trajectories[sample_duck_id].iloc[:10]
    sample_trajectories = {sample_duck_id: sample_trajectory}
    
    features_list = []
    
    for duck_id, trajectory in sample_trajectories.items():
        # Sort trajectory by time
        trajectory = trajectory.sort_values('time')
        
        # For each point in the trajectory (except the last one)
        for i in range(len(trajectory) - 1):
            current_point = trajectory.iloc[i]
            next_point = trajectory.iloc[i + 1]
            
            current_lat = current_point['latitude']
            current_lon = current_point['longitude']
            next_lat = next_point['latitude']
            next_lon = next_point['longitude']
            current_time = current_point['time']
            next_time = next_point['time']
            
            # Calculate movement metrics
            distance = haversine_distance(current_lat, current_lon, next_lat, next_lon)
            bearing = calculate_bearing(current_lat, current_lon, next_lat, next_lon)
            time_diff = (next_time - current_time).total_seconds() / 3600  # hours
            speed = distance / time_diff if time_diff > 0 else 0
            
            # Find nearest weather stations
            nearest_stations = find_nearest_stations(
                current_lat, current_lon, station_tree, stations, k=3
            )
            
            # Extract weather at current location
            weather = extract_weather_at_point(
                weather_rasters, current_time, current_lat, current_lon
            )
            
            # Calculate weather gradients
            gradients = calculate_weather_gradient(
                weather_rasters, current_time,
                current_lat, current_lon, next_lat, next_lon
            )
            
            # Calculate wind assistance/resistance
            if not np.isnan(weather.get('wspd', np.nan)) and not np.isnan(weather.get('wdir', np.nan)):
                wind_assistance = calculate_wind_assistance(
                    weather['wspd'], weather['wdir'], bearing
                )
            else:
                wind_assistance = np.nan
            
            # Create feature dictionary
            feature = {
                'duck_id': duck_id,
                'timestamp': current_time,
                'lat': current_lat,
                'lon': current_lon,
                'next_lat': next_lat,
                'next_lon': next_lon,
                'distance': distance,
                'bearing': bearing,
                'speed': speed,
                'time_diff': time_diff,
                'wind_assistance': wind_assistance,
                'external_temp': current_point.get('external-temperature', np.nan),
                'height_above_msl': current_point.get('height-above-msl', np.nan),
                'ground_speed': current_point.get('ground-speed', np.nan),
                **weather,
                **gradients
            }
            
            features_list.append(feature)
    
    # Convert to DataFrame
    features_df = pd.DataFrame(features_list)
    
    # Drop rows with too many NaN values
    features_df = features_df.dropna(thresh=len(features_df.columns) - 5)
    
    print("\n=== Features Engineered ===")
    print(f"Created {len(features_df)} feature rows")
    print(f"Features: {list(features_df.columns)}")
    print(f"Sample features:\n{features_df.head(2)}")
    
    return features_df


def prepare_lstm_sequences(features_df, sequence_length=24, target_cols=['next_lat', 'next_lon']):
    """Prepare sequence data for LSTM model"""
    # Group by duck_id
    grouped = features_df.groupby('duck_id')
    
    X_sequences = []
    y_targets = []
    metadata = []  # Store information about each sequence
    
    # Feature columns (exclude target and metadata columns)
    feature_cols = [col for col in features_df.columns 
                   if col not in target_cols + ['duck_id', 'timestamp', 'next_lat', 'next_lon']]
    
    # Normalize features
    scaler = MinMaxScaler()
    
    # Use a smaller sequence length for demonstration
    demo_sequence_length = min(sequence_length, len(features_df) - 1)
    if demo_sequence_length < 2:
        demo_sequence_length = 2
    
    # For demonstration, create at least one sequence
    for duck_id, group in grouped:
        # Sort by timestamp
        group = group.sort_values('timestamp')
        
        # Fill NaN values for normalization
        group_filled = group[feature_cols].fillna(0)
        
        if len(group_filled) >= demo_sequence_length + 1:
            # Fit scaler
            scaler.fit(group_filled)
            group[feature_cols] = scaler.transform(group_filled)
            
            # Create at least one sequence
            seq_df = group.iloc[:demo_sequence_length]
            target_df = group.iloc[demo_sequence_length]
            
            X_seq = seq_df[feature_cols].values
            y_target = target_df[target_cols].values
            
            X_sequences.append(X_seq)
            y_targets.append(y_target)
            metadata.append({
                'duck_id': duck_id,
                'start_time': seq_df['timestamp'].iloc[0],
                'end_time': seq_df['timestamp'].iloc[-1],
                'target_time': target_df['timestamp'],
                'last_lat': seq_df['lat'].iloc[-1],
                'last_lon': seq_df['lon'].iloc[-1]
            })
    
    # Convert to numpy arrays
    X = np.array(X_sequences)
    y = np.array(y_targets)
