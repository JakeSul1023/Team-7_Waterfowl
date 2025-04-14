import pandas as pd
from geopy.distance import geodesic
import numpy as np

print("Loading data...")

# Load files
actual_df = pd.read_csv("Mallard Connectivity(2).csv")
predicted_df = pd.read_csv("duck_migration_extended_forecasts-3 1.csv")

print("Cleaning column names and formatting timestamps...")

# Clean column names
actual_df.columns = actual_df.columns.str.strip().str.lower().str.replace(" ", "_")
predicted_df.columns = predicted_df.columns.str.strip().str.lower().str.replace(" ", "_")

# Rename forecast columns to be consistent
predicted_df = predicted_df.rename(columns={
    'forecast_timestamp': 'timestamp',
    'forecast_lat': 'predicted_lat',
    'forecast_lon': 'predicted_long'
})
actual_df = actual_df.rename(columns={'tag-local-identifier': 'duck_id'})

# Convert timestamps
predicted_df['timestamp'] = pd.to_datetime(predicted_df['timestamp'], errors='coerce')
actual_df['timestamp'] = pd.to_datetime(actual_df['timestamp'], errors='coerce')

# Drop missing data
predicted_df = predicted_df.dropna(subset=['timestamp', 'predicted_lat', 'predicted_long'])
actual_df = actual_df.dropna(subset=['timestamp', 'location-lat', 'location-long'])

# Ensure duck_id is string for consistent matching
predicted_df['duck_id'] = predicted_df['duck_id'].astype(str)
actual_df['duck_id'] = actual_df['duck_id'].astype(str)

print("Comparing predictions to closest actual timestamp per duck...")

# Group by duck ID
grouped_predicted = predicted_df.groupby('duck_id')
grouped_actual = actual_df.groupby('duck_id')

distances = []

for duck_id, pred_group in grouped_predicted:
    if duck_id not in grouped_actual.groups:
        continue  # skip unmatched ducks

    actual_group = grouped_actual.get_group(duck_id)

    for _, pred_row in pred_group.iterrows():
        # Find closest actual timestamp
        time_diffs = (actual_group['timestamp'] - pred_row['timestamp']).abs()
        closest_index = time_diffs.idxmin()
        actual_row = actual_group.loc[closest_index]

        pred_coords = (pred_row['predicted_lat'], pred_row['predicted_long'])
        actual_coords = (actual_row['location-lat'], actual_row['location-long'])
        dist_km = geodesic(pred_coords, actual_coords).kilometers
        distances.append(dist_km)

print("\n--- Comparison Summary ---")
print(f"Total comparisons made: {len(distances)}")
print(f"Average distance error: {np.mean(distances):.2f} km")
print(f"Percentage within 10 km: {(np.array(distances) <= 10).mean() * 100:.2f}%")

# Optional: save to file
results_df = pd.DataFrame({'distance_km': distances})
results_df.to_csv("lstm_comparison_results.csv", index=False)
print("\nSaved results to 'lstm_comparison_results.csv'")
