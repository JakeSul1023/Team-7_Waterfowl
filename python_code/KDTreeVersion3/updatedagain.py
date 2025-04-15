import pandas as pd
from geopy.distance import geodesic
import numpy as np

print("Loading actual and predicted duck migration data...")

# Load CSVs
actual_df = pd.read_csv("Mallard Connectivity(2).csv")
predicted_df = pd.read_csv("duck_migration_extended_forecasts_14.csv")

print("Cleaning and standardizing column names...")

# Normalize column names
actual_df.columns = actual_df.columns.str.strip().str.lower().str.replace(" ", "_")
predicted_df.columns = predicted_df.columns.str.strip().str.lower().str.replace(" ", "_")

# Rename columns for consistency
predicted_df = predicted_df.rename(columns={
    'forecast_timestamp': 'timestamp',
    'forecast_lat': 'predicted_lat',
    'forecast_lon': 'predicted_long'
})
actual_df = actual_df.rename(columns={'tag-local-identifier': 'duck_id'})

print("Converting timestamps to datetime format...")
actual_df['timestamp'] = pd.to_datetime(actual_df['timestamp'], errors='coerce')
predicted_df['timestamp'] = pd.to_datetime(predicted_df['timestamp'], errors='coerce')

# Drop invalid timestamps
actual_df = actual_df.dropna(subset=['timestamp'])
predicted_df = predicted_df.dropna(subset=['timestamp'])

# Keep only necessary columns
actual_df = actual_df[['duck_id', 'timestamp', 'location-lat', 'location-long']]
predicted_df = predicted_df[['duck_id', 'timestamp', 'predicted_lat', 'predicted_long']]

# Sort both DataFrames by duck_id and timestamp
actual_df = actual_df.sort_values(by=['duck_id', 'timestamp'])
predicted_df = predicted_df.sort_values(by=['duck_id', 'timestamp'])

print("Matching each predicted timestamp to the closest actual timestamp within tolerance...")

# Perform nearest timestamp join (1-hour tolerance)
merged_df = pd.merge_asof(
    predicted_df,
    actual_df,
    by='duck_id',
    left_on='timestamp',
    right_on='timestamp',
    direction='nearest',
    tolerance=pd.Timedelta("1H")
)

# Drop rows without matches
merged_df = merged_df.dropna(subset=['location-lat', 'location-long'])

print(f"Total comparisons made: {len(merged_df)}")

# Calculate distances
merged_df['distance_km'] = merged_df.apply(
    lambda row: geodesic((row['location-lat'], row['location-long']),
                         (row['predicted_lat'], row['predicted_long'])).kilometers,
    axis=1
)

# Summary statistics
average_distance = merged_df['distance_km'].mean()
within_10_km = (merged_df['distance_km'] <= 10).mean() * 100

print("\n--- Summary ---")
print(f"Average distance error: {average_distance:.2f} km")
print(f"Percent within 10 km: {within_10_km:.2f}%")

# Optional: save results
merged_df.to_csv("lstm_comparison_results.csv", index=False)
print("Comparison results saved to 'lstm_comparison_results.csv'")
