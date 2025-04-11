import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np

def compare_predictions(prediction_file, actual_file):
    # Load the prediction and actual data
    prediction_data = pd.read_csv(prediction_file)
    actual_data = pd.read_csv(actual_file)

    # Preprocess prediction data (rename columns if necessary)
    prediction_data = prediction_data[['duck_id', 'forecast_timestamp', 'forecast_lat', 'forecast_lon']]
    
    # Preprocess actual data (rename columns if necessary)
    actual_data = actual_data[['individual-local-identifier', 'timestamp', 'location-lat', 'location-long']]
    actual_data = actual_data.rename(columns={
        'individual-local-identifier': 'duck_id',
        'timestamp': 'forecast_timestamp',
        'location-lat': 'actual_lat',
        'location-long': 'actual_lon'
    })
    
    # Convert timestamps to datetime format for both prediction and actual data
    prediction_data['forecast_timestamp'] = pd.to_datetime(prediction_data['forecast_timestamp'])
    actual_data['forecast_timestamp'] = pd.to_datetime(actual_data['forecast_timestamp'])

    # Merge both datasets on 'duck_id' and 'forecast_timestamp' with a 5-minute tolerance
    merged_data = pd.merge_asof(
        prediction_data.sort_values('forecast_timestamp'),
        actual_data.sort_values('forecast_timestamp'),
        on='forecast_timestamp', 
        by='duck_id', 
        direction='nearest', 
        tolerance=pd.Timedelta('5min')
    )

    # Check if any data was merged successfully
    if merged_data.empty:
        print("No matching records found within the time tolerance.")
        return

    # Calculate the differences between predicted and actual coordinates
    merged_data['lat_diff'] = merged_data['forecast_lat'] - merged_data['actual_lat']
    merged_data['lon_diff'] = merged_data['forecast_lon'] - merged_data['actual_lon']

    # Calculate the Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
    mse_lat = mean_squared_error(merged_data['actual_lat'], merged_data['forecast_lat'])
    mse_lon = mean_squared_error(merged_data['actual_lon'], merged_data['forecast_lon'])
    rmse_lat = np.sqrt(mse_lat)
    rmse_lon = np.sqrt(mse_lon)

    # Print the results
    print(f"Mean Squared Error (Latitude): {mse_lat}")
    print(f"Root Mean Squared Error (Latitude): {rmse_lat}")
    print(f"Mean Squared Error (Longitude): {mse_lon}")
    print(f"Root Mean Squared Error (Longitude): {rmse_lon}")

    # Optionally, you can save the merged data with the differences for further inspection
    merged_data.to_csv("comparison_results.csv", index=False)
    print("Comparison results saved to 'comparison_results.csv'.")

# Example usage
prediction_file = 'path_to_prediction_file.csv'  # Replace with actual path
actual_file = 'path_to_actual_file.csv'  # Replace with actual path

compare_predictions(prediction_file, actual_file)
