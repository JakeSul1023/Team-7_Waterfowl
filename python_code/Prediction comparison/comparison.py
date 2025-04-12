import pandas as pd
from geopy.distance import geodesic

print("Loading actual and predicted duck migration data...")

# Load CSVs
actual_df = pd.read_csv("Mallard Connectivity(2).csv")
predicted_df = pd.read_csv("duck_migration_extended_forecasts.csv")

print("Cleaning and standardizing column names...")

# Normalize column names
actual_df.columns = actual_df.columns.str.strip().str.lower().str.replace(" ", "_")
predicted_df.columns = predicted_df.columns.str.strip().str.lower().str.replace(" ", "_")

# Rename columns for matching
actual_df = actual_df.rename(columns={'individual-local-identifier': 'duck_id'})
predicted_df = predicted_df.rename(columns={
    'forecast_timestamp': 'timestamp',
    'forecast_lat': 'predicted_lat',
    'forecast_lon': 'predicted_long'
})

print("Converting timestamps to datetime format...")
actual_df['timestamp'] = pd.to_datetime(actual_df['timestamp'], errors='coerce')
predicted_df['timestamp'] = pd.to_datetime(predicted_df['timestamp'], errors='coerce')

# Drop invalid timestamps
actual_df = actual_df.dropna(subset=['timestamp'])
predicted_df = predicted_df.dropna(subset=['timestamp'])

print("Identifying common ducks between actual and predicted datasets...")
common_ducks = set(actual_df['duck_id']).intersection(set(predicted_df['duck_id']))
print(f"Found {len(common_ducks)} matching ducks.")

# Select one duck to show a sample comparison
sample_duck_id = list(common_ducks)[0]
print(f"Analyzing duck: {sample_duck_id}")

# Filter data for this duck
actual_duck_data = actual_df[actual_df['duck_id'] == sample_duck_id].sort_values(by='timestamp')
predicted_duck_data = predicted_df[predicted_df['duck_id'] == sample_duck_id].sort_values(by='timestamp')

print("Matching each actual timestamp to the closest predicted timestamp and calculating distance...")

examples = []
for _, actual_row in actual_duck_data.iterrows():
    actual_time = actual_row['timestamp']
    
    # Find closest prediction
    predicted_duck_data = predicted_duck_data.copy()
    predicted_duck_data['time_diff'] = (predicted_duck_data['timestamp'] - actual_time).abs()
    closest_pred = predicted_duck_data.loc[predicted_duck_data['time_diff'].idxmin()]

    actual_coords = (actual_row['location-lat'], actual_row['location-long'])
    predicted_coords = (closest_pred['predicted_lat'], closest_pred['predicted_long'])

    examples.append({
        'duck_id': sample_duck_id,
        'actual_time': actual_time,
        'actual_lat': actual_coords[0],
        'actual_lon': actual_coords[1],
        'predicted_time': closest_pred['timestamp'],
        'predicted_lat': predicted_coords[0],
        'predicted_lon': predicted_coords[1],
        'time_diff_minutes': (closest_pred['timestamp'] - actual_time).total_seconds() / 60,
        'distance_km': geodesic(actual_coords, predicted_coords).kilometers
    })

# Create a DataFrame with results
example_df = pd.DataFrame(examples)

print("\n--- Sample Prediction Comparison ---")
print(example_df.head())

# Summary statistics
average_distance = example_df['distance_km'].mean()
within_10km = (example_df['distance_km'] <= 10).mean() * 100

print("\n--- Summary ---")
print(f"Total comparisons: {len(example_df)}")
print(f"Average distance error: {average_distance:.2f} km")
print(f"Predictions within 10 km of actual location: {within_10km:.2f}%")

# Save to CSV
example_df.to_csv("comparison_results.csv", index=False)
print("\nResults saved to 'comparison_results.csv'")
