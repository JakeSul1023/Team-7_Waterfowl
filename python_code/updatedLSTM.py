import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
import dask.dataframe as dd
import pyarrow.csv as pv
import multiprocessing as mp
from typing import Dict, List, Optional, Tuple
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
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

print("Libraries installed")

# ========================================================
# Data Loading and Preprocessing
# ========================================================

def load_duck_data(filepath="WaterfowlData.csv"):
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
    try:
        # Try different timestamp formats
        date_formats = [
            '%m/%d/%Y %I:%M:%S %p',  # e.g. "11/30/2024 2:26:14 AM"
            '%H:%M:%S',               # e.g. "26:14.0" (hours can exceed 24)
            '%M:%S.%f'                # e.g. "26:14.0" (as minutes:seconds)
        ]
        
        for date_format in date_formats:
            try:
                duck_df['timestamp'] = pd.to_datetime(duck_df['timestamp'], format=date_format, errors='coerce')
                # If we get here with no errors and no NaT values, break the loop
                if not duck_df['timestamp'].isna().any():
                    break
            except Exception:
                continue
        
        # Check if conversion was successful
        if duck_df['timestamp'].isna().all():
            print("Warning: Could not parse timestamp column. Using default timestamps.")
            # Create artificial timestamps if parsing failed
            duck_df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(duck_df), freq='1min')
    except Exception as e:
        print(f"Error converting timestamps: {e}")
        # Create artificial timestamps
        duck_df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(duck_df), freq='1min')
    
    # Standardize column names
    column_mapping = {
        'location-long': 'longitude',
        'location-lat': 'latitude',
        'external-temperature': 'temperature',
        'height-above-msl': 'altitude',
        'ground-speed': 'speed',
        'heading': 'direction'
    }
    
    # Only rename columns that exist in the dataframe
    existing_columns = {col: new_col for col, new_col in column_mapping.items() 
                       if col in duck_df.columns}
    
    if existing_columns:
        duck_df = duck_df.rename(columns=existing_columns)
        print(f"Renamed columns: {existing_columns}")
    
    # Add any missing essential columns with default values
    essential_columns = ['longitude', 'latitude', 'timestamp']
    for col in essential_columns:
        if col not in duck_df.columns:
            print(f"Warning: Essential column '{col}' missing. Adding with default values.")
            duck_df[col] = np.nan
    
    # Create GeoDataFrame for spatial operations
    geometry = [ShapelyPoint(xy) for xy in zip(duck_df['longitude'], duck_df['latitude'])]
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
    flyway_bbox = [-97.12, 30.52, -80.93, 48.97]
    
    print("\n=== Flyway Region ===")
    print(f"Bounding box: {flyway_bbox}")
    
    return flyway_bbox


def get_all_timestamps(duck_trajectories):
    """Extract all unique timestamps from duck trajectories"""
    all_times = []
    for duck_id, trajectory in duck_trajectories.items():
        all_times.extend(trajectory['timestamp'].tolist())
    
    unique_times = sorted(set(all_times))
    
    print("\n=== Timestamp Analysis ===")
    print(f"Total unique timestamps: {len(unique_times)}")
    if unique_times:
        print(f"Date range: {unique_times[0]} to {unique_times[-1]}")
    print (f"================")
    
    return unique_times


# ========================================================
# Weather Data Collection
# ========================================================

# Removed the get_api_key function to use hardcoded key 

def get_nearest_noaa_location(lat, lon, token="pKsYMZQINiCtWEbiUdvJHImQDqYlZmhu"):
    """Get the nearest NOAA location for a given lat/lon pair"""
    url = f"https://api.weather.gov/points/{lat},{lon}"
    headers = {
        "User-Agent": "DuckMigrationApp/1.0",
        "token": token
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()

        props = data.get("properties", {})
        city = props.get("relativeLocation", {}).get("properties", {}).get("city")
        state = props.get("relativeLocation", {}).get("properties", {}).get("state")
        forecast_url = props.get("forecast", None)

        return {
            "nearest_city": f"{city}, {state}" if city and state else "Unknown",
            "grid_id": props.get("gridId"),
            "grid_x": props.get("gridX"),
            "grid_y": props.get("gridY"),
            "forecast_url": forecast_url
        }
    except Exception as e:
        print(f"Error fetching location metadata: {e}")
        return None


def fetch_noaa_forecast_for_point(lat, lon, token="pKsYMZQINiCtWEbiUdvJHImQDqYlZmhu"):
    """
    Fetch forecast JSON from NOAA NWS API for a given point.
    Also returns the nearest city/state name if available.
    
    Parameters:
      - lat, lon: Latitude and Longitude.
      - token: NOAA API token.
      
    Returns:
      - JSON object with forecast data.
    """
    points_url = f"https://api.weather.gov/points/{lat},{lon}"
    headers = {
        "User-Agent": "DuckMigrationApp/1.0",
        "token": token
    }

    try:
        response = requests.get(points_url, headers=headers)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Error querying NOAA points endpoint: {e}")

    props = data.get("properties", {})
    city = props.get("relativeLocation", {}).get("properties", {}).get("city")
    state = props.get("relativeLocation", {}).get("properties", {}).get("state")
    location_name = f"{city}, {state}" if city and state else "Unknown Location"

    forecast_url = props.get("forecast")
    if not forecast_url:
        raise ValueError(f"Forecast URL not found in NOAA response for point ({lat:.2f}, {lon:.2f})")
    
    print (f"========= Nearest NOAA Location ========")
    print(f"Nearest NOAA location: {location_name}")
    print(f"Fetching forecast for point ({lat:.2f}, {lon:.2f})...")

    try:
        forecast_response = requests.get(forecast_url, headers=headers)
        forecast_response.raise_for_status()
        forecast_json = forecast_response.json()
        print("Forecast JSON Fetched")
        return forecast_json
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"Forecast URL returned 404 or error: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to retrieve forecast from {forecast_url}: {e}")


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


def is_valid_noaa_point(lat, lon, token="pKsYMZQINiCtWEbiUdvJHImQDqYlZmhu"):
    """Check if a given lat/lon point has valid NOAA forecast data"""
    url = f"https://api.weather.gov/points/{lat},{lon}"
    headers = {
        "User-Agent": "DuckMigrationApp/1.0",
        "token": token
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        return "forecast" in data.get("properties", {})
    except requests.exceptions.HTTPError as e:
        print(f"[INVALID NOAA POINT] ({lat}, {lon}) - {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Checking point ({lat}, {lon}): {e}")
        return False


def fetch_weather_forecast_robust(flyway_bbox, resolution=0.25, forecast_dates=None, token="pKsYMZQINiCtWEbiUdvJHImQDqYlZmhu"):
    """More robust version of fetch_weather_forecast that handles errors gracefully"""
    # Unpack bounding box
    lon_min, lat_min, lon_max, lat_max = flyway_bbox
    
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
            # Slight jitter to avoid invalid edge zones
            lat_j = lat + np.random.uniform(-0.05, 0.05)
            lon_j = lon + np.random.uniform(-0.05, 0.05)

            # Validate the point before fetching
            if not is_valid_noaa_point(lat_j, lon_j, token):
                print(f"Skipping invalid NOAA point ({lat_j:.2f}, {lon_j:.2f})")
                print("Point may be on water. NOAA does not cover weather points on any body of water.")
                continue

            try:
                print(f"Fetching forecast for point ({lat_j:.2f}, {lon_j:.2f})...")
                forecast_json = fetch_noaa_forecast_for_point(lat_j, lon_j, token)

                periods = forecast_json.get("properties", {}).get("periods", [])
                if not periods:
                    print(f"No forecast periods returned for point ({lat_j:.2f}, {lon_j:.2f}).")
                    continue

                for period in periods:
                    try:
                        period_start = pd.to_datetime(period["startTime"])
                        if forecast_dates_set is not None:
                            if period_start.date() not in forecast_dates_set:
                                continue
                        record = {
                            "lat": lat_j,
                            "lon": lon_j,
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
                print(f"Error fetching forecast for point ({lat_j:.2f}, {lon_j:.2f}): {e}")

    
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

def create_weather_raster_timeseries(weather_data, duck_trajectories, flyway_bbox, resolution=0.1):
    """
    Create a time series of weather rasters aligned with duck migration timestamps
    
    Parameters:
    - weather_data: DataFrame with weather observations (must include 'longitude', 'latitude', and weather parameters)
    - duck_trajectories: Dictionary of GeoDataFrames with duck migration data (must include timestamps)
    - flyway_bbox: Bounding box [lon_min, lat_min, lon_max, lat_max]
    - resolution: Spatial resolution in degrees
    
    Returns:
    - Dictionary of xarray DataArrays, keyed by parameter and timestamp
    """
    if weather_data.empty:
        raise ValueError("Weather data is empty")
    
    # Extract unique timestamps from duck trajectories
    unique_time_duck = get_all_timestamps(duck_trajectories)
    
    # Limit to a few timestamps for demonstration
    if len(unique_time_duck) > 5:
        unique_time_duck = unique_time_duck[:5]
    
    # Define parameters to interpolate
    params = ['temp', 'rhum', 'prcp', 'wspd', 'wdir', 'pres']
    
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
    print(f"Time steps: {len(unique_time_duck)}")
    
    return raster_dict


def build_kdtree_for_stations(valid_stations):
    """Build a KD-tree for efficient nearest station lookup"""
    if not isinstance(valid_stations, list) or len(valid_stations) == 0:
        raise ValueError("Stations must be a non-empty list of dictionaries")
    
    # Extract coordinates
    coords = np.array([[s['latitude'], s['longitude']] for s in valid_stations])
    
    # Convert lat/lon to radians for haversine distance
    coords_rad = np.radians(coords)
    
    # Build KD-tree
    tree = BallTree(coords_rad, metric='haversine')
    
    print("\n=== KDTree Built ===")
    print(f"Tree built for {len(valid_stations)} weather stations")
    
    return tree, coords


def find_nearest_stations(lat, lon, tree, stations, k=3):
    """Find k nearest weather stations to a location"""
    if not isinstance(stations, list) or len(stations) == 0:
        raise ValueError("Stations must be a non-empty list of dictionaries")
    
    # Convert to radians
    lat_rad, lon_rad = np.radians([lat, lon])
    
    # Query the tree
    distances, indices = tree.query([[lat_rad, lon_rad]], k=k)
    
    # Convert distances to kilometers (Earth radius ≈ 6371 km)
    distances = distances[0] * 6371.0
    
    # Get station info
    nearest_stations = [
        {**stations[idx], 'distance': dist} if 'name' in stations[idx] else 
        {'name': 'Unknown', 'distance': dist}
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
    print(f"Point 1: ({np.degrees(lat1):.6f}°, {np.degrees(lon1):.6f}°)")
    print(f"Point 2: ({np.degrees(lat2):.6f}°, {np.degrees(lon2):.6f}°)")
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
    
    # Define expected column mappings with fallbacks
    column_mappings = {
        'lat': ['latitude', 'location-lat'],
        'lon': ['longitude', 'location-long'],
        'temp': ['temperature', 'external-temperature'],
        'alt': ['altitude', 'height-above-msl'],
        'speed': ['speed', 'ground-speed'],
        'direction': ['direction', 'heading']
    }
    
    for duck_id, trajectory in sample_trajectories.items():
        # Sort trajectory by time
        trajectory = trajectory.sort_values('timestamp')
        
        # Helper function to get column value with fallbacks
        def get_col_value(row, col_options):
            for col in col_options:
                if col in row.index and not pd.isna(row[col]):
                    return row[col]
            return np.nan
        
        # For each point in the trajectory (except the last one)
        for i in range(len(trajectory) - 1):
            current_point = trajectory.iloc[i]
            next_point = trajectory.iloc[i + 1]
            
            # Extract coordinates with fallbacks
            current_lat = get_col_value(current_point, column_mappings['lat'])
            current_lon = get_col_value(current_point, column_mappings['lon'])
            next_lat = get_col_value(next_point, column_mappings['lat'])
            next_lon = get_col_value(next_point, column_mappings['lon'])
            current_time = current_point['timestamp']
            next_time = next_point['timestamp']
            
            # Skip if essential data is missing
            if (np.isnan(current_lat) or np.isnan(current_lon) or 
                np.isnan(next_lat) or np.isnan(next_lon)):
                continue
                
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
            
            # Extract additional telemetry data with fallbacks
            temp = get_col_value(current_point, column_mappings['temp'])
            alt = get_col_value(current_point, column_mappings['alt'])
            measured_speed = get_col_value(current_point, column_mappings['speed'])
            
            # Add acceleration data if available
            accel_features = {}
            for accel_axis in ['acceleration-raw-x', 'acceleration-raw-y', 'acceleration-raw-z']:
                if accel_axis in current_point.index and not pd.isna(current_point[accel_axis]):
                    accel_features[accel_axis.replace('-raw', '')] = current_point[accel_axis]
            
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
                'calculated_speed': speed,
                'measured_speed': measured_speed,
                'time_diff': time_diff,
                'wind_assistance': wind_assistance,
                'external_temp': temp,
                'altitude': alt,
                **accel_features,
                **weather,
                **gradients
            }
            
            # Add any GPS quality metrics if available
            for gps_metric in ['gps:hdop', 'gps:satellite-count', 'gps-time-to-fix']:
                if gps_metric in current_point.index and not pd.isna(current_point[gps_metric]):
                    feature[gps_metric.replace(':', '_')] = current_point[gps_metric]
            
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
    demo_sequence_length = min(sequence_length, 10)  # Limit to 10 for demo
    if demo_sequence_length < 2:
        demo_sequence_length = 2
    
    print(f"\n=== Preparing LSTM Sequences ===")
    print(f"Using sequence length: {demo_sequence_length}")
    print(f"Target columns: {target_cols}")
    print(f"Feature columns: {feature_cols[:5]}... (total: {len(feature_cols)})")
    
    # For each duck's trajectory
    for duck_id, group in grouped:
        # Sort by timestamp
        group = group.sort_values('timestamp')
        
        # Fill NaN values for normalization
        group_filled = group[feature_cols].fillna(0)
        
        if len(group_filled) >= demo_sequence_length + 1:
            # Fit scaler on this duck's data
            scaler.fit(group_filled)
            group_normalized = group.copy()
            group_normalized[feature_cols] = scaler.transform(group_filled)
            
            # Create sequences
            for i in range(len(group_normalized) - demo_sequence_length):
                seq_df = group_normalized.iloc[i:i+demo_sequence_length]
                target_df = group_normalized.iloc[i+demo_sequence_length]
                
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
    
    print(f"Created {len(X)} sequences")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
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
    
    # Build the model
    model = build_lstm_model(input_shape, output_dim)
    
    # Learning rate scheduler
    def lr_scheduler(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return lr * tf.math.exp(-0.1)
    
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
        tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
    ]
    
    # Train the model
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
    print(pred_df.head())
    
    return pred_df


def visualize_predictions(pred_df):
    """Visualize prediction results"""
    # Sample a subset for visualization if too many
    if len(pred_df) > 50:
        viz_df = pred_df.sample(50)
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
    
    # Save the visualizations
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
    
    # Convert to DataFrame
    analysis_df = pd.DataFrame(analysis_results)
    
    # Visualize migration directions
    plt.figure(figsize=(10, 10), dpi=100)
    ax = plt.subplot(111, projection='polar')
    
    # Convert bearings to radians for polar plot
    bearings_rad = np.radians(analysis_df['avg_bearing'])
    
    # Use speed for the length of the arrows
    speeds = analysis_df['avg_speed_kmh']
    normalized_speeds = speeds / speeds.max() if speeds.max() > 0 else speeds
    
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
    
    print("\nMigration Pattern Summary:")
    print(f"Analysis performed on {len(analysis_df)} ducks")
    print(f"Average migration speed: {analysis_df['avg_speed_kmh'].mean():.2f} km/h")
    print(f"Average migration distance: {analysis_df['total_distance_km'].mean():.2f} km")
    
    # Predominant direction analysis
    direction_counts = {}
    for bearing in analysis_df['avg_bearing']:
        if np.isnan(bearing):
            continue
            
        # Convert bearing to cardinal direction
        if 22.5 <= bearing < 67.5:
            direction = 'NE'
        elif 67.5 <= bearing < 112.5:
            direction = 'E'
        elif 112.5 <= bearing < 157.5:
            direction = 'SE'
        elif 157.5 <= bearing < 202.5:
            direction = 'S'
        elif 202.5 <= bearing < 247.5:
            direction = 'SW'
        elif 247.5 <= bearing < 292.5:
            direction = 'W'
        elif 292.5 <= bearing < 337.5:
            direction = 'NW'
        else:
            direction = 'N'
            
        direction_counts[direction] = direction_counts.get(direction, 0) + 1
    
    predominant_direction = max(direction_counts.items(), key=lambda x: x[1])[0] if direction_counts else "Unknown"
    print(f"Predominant migration direction: {predominant_direction}")
    
    return analysis_df


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
    
    # Get timestamps for analysis
    all_timestamps = get_all_timestamps(duck_trajectories)
    
    # Set up date range for weather data
    if all_timestamps:
        start_date = all_timestamps[0].replace(hour=0, minute=0, second=0)
        end_date = all_timestamps[-1].replace(hour=23, minute=59, second=59)
    else:
        # Default to recent dates if no timestamps
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
    
    # Get nearby weather stations
    try:
        stations = Stations().nearby(flyway_bbox[1], flyway_bbox[0], 200)  # latitude, longitude, radius in km
        stations_list = stations.fetch().to_dict('records')
    except Exception as e:
        print(f"Error fetching stations: {e}")
        # Create dummy stations for testing
        stations_list = [
            {'id': 'test1', 'name': 'Test Station 1', 'latitude': flyway_bbox[1] + 1, 'longitude': flyway_bbox[0] + 1},
            {'id': 'test2', 'name': 'Test Station 2', 'latitude': flyway_bbox[1] - 1, 'longitude': flyway_bbox[0] - 1}
        ]
    
    # Build KD-tree for efficient station lookup
    station_tree, station_coords = build_kdtree_for_stations(stations_list)
    
    # Fetch weather data
    weather_data = fetch_weather_data(stations_list, start_date, end_date)
    
    # Fetch NOAA forecast data
    forecast_data = fetch_weather_forecast_robust(flyway_bbox, resolution=0.5, forecast_dates=all_timestamps)
    
    # Create weather rasters
    weather_rasters = create_weather_raster_timeseries(weather_data, duck_trajectories, flyway_bbox)
    
    # Engineer features
    features_df = engineer_features(duck_trajectories, weather_rasters, station_tree, stations_list)
    
    # Prepare sequences for LSTM
    X, y, metadata = prepare_lstm_sequences(features_df)
    
    # If we have sufficient data
    if len(X) >= 10:  # Minimum number for meaningful splits
        # Split data
        X_train, X_test, y_train, y_test, metadata_train, metadata_test = train_test_split(
            X, y, metadata, test_size=0.2, random_state=42
        )
        
        # Further split training data to get validation set
        X_train, X_val, y_train, y_val, metadata_train, metadata_val = train_test_split(
            X_train, y_train, metadata_train, test_size=0.2, random_state=42
        )
        
        # Train LSTM model
        model, history = train_lstm_model(X_train, y_train, X_val, y_val, epochs=50, batch_size=16)
        
        # Make predictions
        predictions = predict_duck_migration(model, X_test, metadata_test)
        
        # Get feature names for importance analysis
        feature_cols = [col for col in features_df.columns 
                       if col not in ['duck_id', 'timestamp', 'next_lat', 'next_lon']]
        
        # Analyze feature importance
        importance_df = analyze_feature_importance(model, feature_cols)
        
        # Analyze migration patterns
        analysis_df = analyze_migration_patterns(predictions, duck_trajectories)
        
        print("\n===== Duck Migration Prediction System Complete =====\n")
        
        return {
            'model': model,
            'history': history,
            'predictions': predictions,
            'duck_trajectories': duck_trajectories,
            'weather_rasters': weather_rasters,
            'feature_importance': importance_df,
            'migration_analysis': analysis_df
        }
    else:
        print("Not enough data to train the model. Need at least 10 sequences.")
        return {
            'duck_trajectories': duck_trajectories,
            'weather_rasters': weather_rasters
        }


if __name__ == "__main__":
    try:
        results = main()
        print("Program completed successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()