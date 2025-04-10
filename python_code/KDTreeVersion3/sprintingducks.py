from updatedagain import *  # Assuming the required functions are inside this module
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import os

# Set API token - ensure this is correct
TOKEN = "02de0c63b48bcd13d425a73caa22eb81"  # OpenWeather API token
DATA_PATH = "ShortTermSetData(Aug-Sept).csv"

def main():
    """Main function to run the duck migration prediction system"""
    print("\n===== Starting Duck Migration Prediction System =====\n")
    
    # Check if data file exists
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file '{DATA_PATH}' not found. Please check the file path.")
        return
    
    # Step 1: Load duck data
    print("Loading duck data...")
    duck_trajectories, duck_gdf = load_duck_data(DATA_PATH)
    if duck_trajectories is None or len(duck_trajectories) == 0:
        print("Error loading duck data. Exiting.")
        return
    
    # Step 2: Define flyway region
    flyway_bbox = get_flyway_region()
    
    # Step 3: Get all timestamps from duck trajectories
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
    
    # Step 4: Fetch OpenWeather forecast for a sample point
    try:
        print(f"Fetching OpenWeather forecast for point ({lat}, {lon})...")
        forecast_json = get_openweather_forecast(lat, lon, api_key=TOKEN)
        
        # Check if the forecast was retrieved successfully
        if forecast_json:
            print(f"Total forecast periods fetched: {len(forecast_json['forecast'])}")
        else:
            print("No forecast data retrieved.")
        
    except Exception as e:
        print(f"Error fetching OpenWeather forecast: {e}")
        forecast_json = None

    # Continue with the rest of your logic...

    
    # Step 5: Get weather stations
    try:
        print("Fetching weather stations...")
        stations_instance = Stations()
        stations_data = stations_instance.fetch()

        # Filter stations to ensure they have required fields
        valid_stations = []
        for station in stations_data.itertuples():
            # Convert to dictionary
            station_dict = {
                'id': getattr(station, 'id', f"station_{len(valid_stations)}"),  # Set default 'id' if missing
                'name': getattr(station, 'name', 'Unknown'),
                'latitude': getattr(station, 'latitude', None),
                'longitude': getattr(station, 'longitude', None),
                'elevation': getattr(station, 'elevation', 0)
            }

            # Only include stations with valid coordinates
            if station_dict['latitude'] is not None and station_dict['longitude'] is not None:
                valid_stations.append(station_dict)

        print(f"Found {len(valid_stations)} valid weather stations")
    except Exception as e:
        print(f"Error fetching stations: {e}")
        # Create dummy stations for testing
        valid_stations = [
            {'id': 'test1', 'name': 'Test Station 1', 'latitude': flyway_bbox[1] + 1, 'longitude': flyway_bbox[0] + 1, 'elevation': 0},
            {'id': 'test2', 'name': 'Test Station 2', 'latitude': flyway_bbox[1] - 1, 'longitude': flyway_bbox[0] - 1, 'elevation': 0}
        ]

    # Step 6: Build KD-tree for stations
    try:
        print("Building KD-tree for stations...")
        station_tree, station_coords = build_kdtree_for_stations(valid_stations)
        print("KD-tree built successfully")
    except Exception as e:
        print(f"Error building KD-tree: {e}")
        return
    
    # Step 7: Find nearest stations to a sample point
    try:
        nearest_stations = find_nearest_stations(lat, lon, station_tree, valid_stations, k=3)
        print(f"Found {len(nearest_stations)} nearest stations to point ({lat}, {lon})")
    except Exception as e:
        print(f"Error finding nearest stations: {e}")
    
    # Step 8: Fetch historical weather data (limit to 50 stations)
    try:
        print(f"Fetching weather data from {start_date} to {end_date}...")
        weather_data = fetch_weather_data(valid_stations[:50], start_date, end_date)  # Fetch weather data
        if weather_data.empty:
            print("Warning: No weather data returned")
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        weather_data = pd.DataFrame(columns=['temp', 'rhum', 'prcp', 'wspd', 'wdir', 'pres', 'station_id', 'latitude', 'longitude'])
    
    # Step 9: Fetch weather forecast (limited resolution to reduce API load)
    try:
        print("Fetching weather forecast...")
        future_dates = [datetime.now() + timedelta(days=i) for i in range(1, 3)]  # 2 days
        forecast_data = fetch_weather_forecast_robust(flyway_bbox, resolution=3.0, forecast_dates=future_dates, api_key=TOKEN)
        if forecast_data.empty:
            print("Warning: No forecast data returned")
    except Exception as e:
        print(f"Error fetching weather forecast: {e}")
        forecast_data = pd.DataFrame()
    
    # Step 10: Example calculations between two points
    point1_lat, point1_lon = 40.0, -90.0  # Near St. Louis, MO
    point2_lat, point2_lon = 42.0, -88.0  # Near Chicago, IL
    
    print("\nPerforming geographical calculations...")
    try:
        distance = haversine_distance(point1_lat, point1_lon, point2_lat, point2_lon)
        print(f"Distance between points: {distance:.2f} km")
        
        bearing = calculate_bearing(point1_lat, point1_lon, point2_lat, point2_lon)
        print(f"Bearing between points: {bearing:.2f} degrees")
    except Exception as e:
        print(f"Error in geographical calculations: {e}")
    
    # Step 11: Create weather rasters if we have data
    weather_rasters = {}
    if not weather_data.empty:
        try:
            print("\nCreating weather raster timeseries...")
            weather_rasters = create_weather_raster_timeseries(weather_data, duck_trajectories, flyway_bbox, resolution=0.5)
            print(f"Created {len(weather_rasters)} weather raster layers")
        except Exception as e:
            print(f"Error creating weather rasters: {e}")
    else:
        print("Skipping weather raster creation due to missing weather data")
    
    # Step 12: Extract weather at points if we have rasters
    if weather_rasters:
        try:
            print("\nExtracting weather at sample points...")
            sample_time = start_date + timedelta(days=1)  # A time within our data range
            point_weather = extract_weather_at_point(weather_rasters, sample_time, lat, lon)
            print(f"Weather at point ({lat}, {lon}): {point_weather}")
            
            weather_gradients = calculate_weather_gradient(
                weather_rasters, sample_time, point1_lat, point1_lon, point2_lat, point2_lon, distance=50
            )
            print(f"Weather gradients: {weather_gradients}")
        except Exception as e:
            print(f"Error extracting point weather data: {e}")
    else:
        print("Skipping weather extraction due to missing weather rasters")
    
    # Step 13: Engineer features if we have all required data
    if weather_rasters and duck_trajectories and valid_stations and station_tree:
        try:
            print("\nEngineering features for machine learning...")
            engineered_features = engineer_features(duck_trajectories, weather_rasters, station_tree, valid_stations)
            print(f"Created {len(engineered_features)} feature rows")
            
            # Proceed with model preparation if we have sufficient data
            if len(engineered_features) >= 10:
                print("\nPreparing sequence data for LSTM model...")
                try:
                    X, y, metadata = prepare_lstm_sequences(engineered_features)
                    print(f"Prepared {len(X)} sequences for model training")
                    print(f"Feature dimensions: {X.shape}")
                    print(f"Target dimensions: {y.shape}")
                    
                    # Perform model training
                    print("\nSplitting data for training...")
                    X_train, X_test, y_train, y_test, metadata_train, metadata_test = train_test_split(
                        X, y, metadata, test_size=0.2, random_state=42
                    )
                    
                    # Further split for validation
                    X_train, X_val, y_train, y_val, metadata_train, metadata_val = train_test_split(
                        X_train, y_train, metadata_train, test_size=0.2, random_state=42
                    )
                    
                    # Train model
                    print("\nTraining LSTM model...")
                    model, history = train_lstm_model(X_train, y_train, X_val, y_val, epochs=20)
                    
                    # Make predictions and save to CSV
                    print("\nGenerating duck migration predictions...")
                    predictions = predict_duck_migration(model, X_test, metadata_test)
                    
                    # Save predictions to CSV
                    predictions_file = "duck_migration_predictions.csv"
                    predictions.to_csv(predictions_file, index=False)
                    print(f"Predictions saved to {predictions_file}")
                    
                    # Generate extended forecasts if needed
                    print("\nGenerating extended forecasts...")
                    extended_predictions = generate_extended_forecasts(model, X_test, metadata_test, days_ahead=5)
                    extended_file = "duck_migration_extended_forecasts.csv"
                    extended_predictions.to_csv(extended_file, index=False)
                    print(f"Extended forecasts saved to {extended_file}")
                    
                except Exception as e:
                    print(f"Error in model training and prediction: {e}")
            else:
                print("Not enough feature data for model training (need at least 10 rows)")
        except Exception as e:
            print(f"Error in feature engineering: {e}")
    else:
        print("Skipping feature engineering due to missing prerequisite data")
    
    print("\n===== Duck Migration Prediction System Complete =====")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred in main execution: {e}")
        import traceback
        traceback.print_exc()
