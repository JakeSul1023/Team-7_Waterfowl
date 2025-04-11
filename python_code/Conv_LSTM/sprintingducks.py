# Import all functions from updatedLSTM module
from updatedagain import *

import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import os

# Set API token - consider using environment variables for sensitive tokens
TOKEN = "pKsYMZQINiCtWEbiUdvJHImQDqYlZmhu"
DATA_PATH = "ShortTermSetData(Aug-Sept).csv"

def main():
    """Main execution function"""
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
    
    # Step 3: Get timestamps for analysis
    all_timestamps = get_all_timestamps(duck_trajectories)
    
    # Set up date range for weather data based on duck data timestamps
    if all_timestamps:
        start_date = min(all_timestamps) - timedelta(days=1)  # Pad by 1 day before
        end_date = max(all_timestamps) + timedelta(days=1)    # Pad by 1 day after
        print(f"Weather data will cover: {start_date} to {end_date}")
    else:
        print("Warning: No valid timestamps found in duck data. Using default range.")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
    
    # Step 4: Get nearby weather stations
    try:
        print("Fetching weather stations...")
        stations = Stations().nearby(flyway_bbox[1], flyway_bbox[0], 200)  # latitude, longitude, radius in km
        stations_list = stations.fetch().to_dict('records')
    
        # Ensure stations have required fields
        valid_stations = []
        for station in stations_list:
            try:
                station_dict = {
                    'id': station.get('id', f"station_{len(valid_stations)}"),
                    'name': station.get('name', 'Unknown'),
                    'latitude': float(station.get('latitude', None)),
                    'longitude': float(station.get('longitude', None)),
                    'elevation': float(station.get('elevation', 0))
                }
                # Only include stations with valid, non-null coordinates
                if (station_dict['latitude'] is not None and 
                    station_dict['longitude'] is not None and 
                    not pd.isna(station_dict['latitude']) and 
                    not pd.isna(station_dict['longitude'])):
                    valid_stations.append(station_dict)
            except (TypeError, ValueError) as e:
                print(f"Skipping invalid station data: {station.get('id', 'unknown')} - {e}")
                continue
    
        print(f"Found {len(valid_stations)} valid weather stations")
        if not valid_stations:
            print("No valid stations found. Using dummy stations.")
            valid_stations = [
                {'id': 'test1', 'name': 'Test Station 1', 'latitude': flyway_bbox[1] + 1, 'longitude': flyway_bbox[0] + 1, 'elevation': 0},
                {'id': 'test2', 'name': 'Test Station 2', 'latitude': flyway_bbox[1] - 1, 'longitude': flyway_bbox[0] - 1, 'elevation': 0}
            ]
        stations_list = valid_stations
    except Exception as e:
        print(f"Error fetching stations: {e}")
        # Use dummy stations as fallback
        stations_list = [
            {'id': 'test1', 'name': 'Test Station 1', 'latitude': flyway_bbox[1] + 1, 'longitude': flyway_bbox[0] + 1, 'elevation': 0},
            {'id': 'test2', 'name': 'Test Station 2', 'latitude': flyway_bbox[1] - 1, 'longitude': flyway_bbox[0] - 1, 'elevation': 0}
        ]
        print(f"Using {len(stations_list)} dummy stations due to fetch error")

    # Step 5: Build KD-tree for efficient station lookup
    try:
        print("Building KD-tree for stations...")
        station_tree, station_coords = build_kdtree_for_stations(stations_list)
        print("KD-tree built successfully")
    except Exception as e:
        print(f"Error building KD-tree: {e}")
        return
    
    # Step 6: Fetch weather data
    try:
        print(f"Fetching weather data from {start_date} to {end_date}...")
        weather_data = fetch_weather_data(stations_list[:50], start_date, end_date)  # Limit to 50 stations
        if weather_data.empty:
            print("Warning: No weather data returned")
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        weather_data = pd.DataFrame(columns=['temp', 'rhum', 'prcp', 'wspd', 'wdir', 'pres', 'station_id', 'latitude', 'longitude'])
    
    # Step 7: Fetch NOAA forecast data
    try:
        print("Fetching weather forecast...")
        forecast_data = fetch_weather_forecast_robust(
            flyway_bbox, 
            resolution=0.5, 
            forecast_dates=[start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)],
            token=TOKEN
        )
        if forecast_data.empty:
            print("Warning: No forecast data returned")
    except Exception as e:
        print(f"Error fetching weather forecast: {e}")
        forecast_data = pd.DataFrame()
    
    # Step 8: Create weather rasters
    try:
        print("\nCreating weather raster timeseries...")
        weather_rasters = create_weather_raster_timeseries(weather_data, duck_trajectories, flyway_bbox, resolution=0.5)
        print(f"Created {len(weather_rasters)} weather raster layers")
    except Exception as e:
        print(f"Error creating weather rasters: {e}")
        weather_rasters = {}
    
    # Step 9: Engineer features
    try:
        print("\nEngineering features for machine learning...")
        features_df = engineer_features(duck_trajectories, weather_rasters, station_tree, stations_list)
        print(f"Created {len(features_df)} feature rows")
    except Exception as e:
        print(f"Error in feature engineering: {e}")
        features_df = pd.DataFrame()
    
    # Step 10: Prepare sequences for LSTM
    if not features_df.empty:
        try:
            print("\nPreparing sequence data for LSTM model...")
            X, y, metadata = prepare_lstm_sequences(features_df)
            print(f"Prepared {len(X)} sequences for model training")
            print(f"Feature dimensions: {X.shape}")
            print(f"Target dimensions: {y.shape}")
        except Exception as e:
            print(f"Error preparing LSTM sequences: {e}")
            X, y, metadata = [], [], []
    else:
        print("Skipping LSTM preparation due to missing features")
        X, y, metadata = [], [], []
    
    # Step 11: Train and predict if sufficient data
    if len(X) >= 10:  # Minimum number for meaningful splits
        try:
            # Split data
            print("\nSplitting data for training...")
            X_train, X_test, y_train, y_test, metadata_train, metadata_test = train_test_split(
                X, y, metadata, test_size=0.2, random_state=42
            )
            
            # Further split for validation
            X_train, X_val, y_train, y_val, metadata_train, metadata_val = train_test_split(
                X_train, y_train, metadata_train, test_size=0.2, random_state=42
            )
            
            # Train LSTM model
            print("\nTraining LSTM model...")
            model, history = train_lstm_model(X_train, y_train, X_val, y_val, epochs=50, batch_size=16)
            
            # Make predictions
            print("\nGenerating duck migration predictions...")
            predictions = predict_duck_migration(model, X_test, metadata_test)
            
            # Generate extended forecasts starting from the last timestamp
            last_timestamp = max(all_timestamps) if all_timestamps else datetime.now()
            print(f"Generating 7-day forecasts starting from {last_timestamp}")
            
            # Filter test data to include only sequences ending near the last timestamp
            latest_sequences = []
            latest_metadata = []
            for i, meta in enumerate(metadata_test):
                if 'end_time' in meta and abs((meta['end_time'] - last_timestamp).total_seconds()) < 24*3600:  # Within 1 day
                    latest_sequences.append(X_test[i])
                    latest_metadata.append(meta)
            
            extended_forecasts = pd.DataFrame()
            if latest_sequences:
                X_latest = np.array(latest_sequences)
                try:
                    extended_forecasts = generate_extended_forecasts(model, X_latest, latest_metadata, days_ahead=7)
                except Exception as e:
                    print(f"Error generating extended forecasts: {e}")
            else:
                print("No sequences found near the last timestamp for forecasting")
            
            # Save predictions to CSV
            predictions_file = "duck_migration_predictions.csv"
            predictions.to_csv(predictions_file, index=False)
            print(f"Predictions saved to {predictions_file}")
            
            extended_file = "duck_migration_extended_forecasts.csv"
            extended_forecasts.to_csv(extended_file, index=False)
            print(f"Extended forecasts saved to {extended_file}")
            
            # Get feature names for importance analysis
            feature_cols = [col for col in features_df.columns 
                          if col not in ['duck_id', 'timestamp', 'next_lat', 'next_lon']]
            
            # Analyze feature importance
            try:
                importance_df = analyze_feature_importance(model, feature_cols)
            except Exception as e:
                print(f"Error analyzing feature importance: {e}")
                importance_df = pd.DataFrame()
            
            # Analyze migration patterns
            try:
                analysis_df = analyze_migration_patterns(predictions, duck_trajectories)
            except Exception as e:
                print(f"Error analyzing migration patterns: {e}")
                analysis_df = pd.DataFrame()
            
        except Exception as e:
            print(f"Error in model training and prediction: {e}")
            return
        
        print("\n===== Duck Migration Prediction System Complete =====\n")
        
        return {
            'model': model,
            'history': history,
            'predictions': predictions,
            'duck_trajectories': duck_trajectories,
            'weather_rasters': weather_rasters,
            'feature_importance': importance_df,
            'migration_analysis': analysis_df,
            'extended_forecasts': extended_forecasts
        }
    else:
        print("Not enough data to train the model. Need at least 10 sequences.")
        return {
            'duck_trajectories': duck_trajectories,
            'weather_rasters': weather_rasters
        }

def generate_extended_forecasts(model, X_test, metadata_test, days_ahead=7):
    """
    Generate extended forecasts for 7 days ahead starting from the last timestamp
    
    Parameters:
        model: Trained LSTM model
        X_test: Test data sequences
        metadata_test: Metadata for test sequences
        days_ahead: Number of days to forecast ahead (default 7)
        
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
        
        # Get last known location and timestamp
        current_lat = meta.get('last_lat', None)
        current_lon = meta.get('last_lon', None)
        base_time = meta.get('end_time', meta.get('target_time', datetime.now()))  # Fallback to target_time or now
        
        if current_lat is None or current_lon is None:
            print(f"Warning: Missing coordinates for duck_id {meta.get('duck_id')}. Skipping.")
            continue
        
        # Make forecasts for each day ahead
        for day in range(1, days_ahead + 1):
            try:
                # Predict the next position
                pred = model.predict(sequence, verbose=0)
                
                # Extract predicted lat/lon
                pred_lat = pred[0, 0]
                pred_lon = pred[0, 1]
                
                # Calculate distance moved
                distance = haversine_distance(current_lat, current_lon, pred_lat, pred_lon)
                
                # Calculate bearing
                bearing = calculate_bearing(current_lat, current_lon, pred_lat, pred_lon)
                
                # Store the forecast
                forecast = {
                    'duck_id': meta.get('duck_id', 'unknown'),
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
                
                # Note: Keeping original sequence as per provided implementation
                # In production, update sequence with forecasted weather data
            except Exception as e:
                print(f"Error predicting for duck_id {meta.get('duck_id')} on day {day}: {e}")
                break
    
    if not all_forecasts:
        print("No forecasts generated")
        return pd.DataFrame()
        
    # Convert to DataFrame
    forecasts_df = pd.DataFrame(all_forecasts)
    
    print(f"Generated {len(forecasts_df)} forecast points")
    
    # Visualize forecasts on a map if function exists
    try:
        visualize_extended_forecasts(forecasts_df)
    except Exception as e:
        print(f"Error visualizing forecasts: {e}")
    
    return forecasts_df

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred in main execution: {e}")
        import traceback
        traceback.print_exc()