import pandas as pd
from geopy.distance import geodesic

# Load datasets
gnn_df = pd.read_csv("GNN_predictions_OneMonth.csv")
actual_df = pd.read_csv("Mallard Connectivity(2).csv")

# Standardize ID formats and timestamp parsing
gnn_df['duck_id'] = gnn_df['duck_id'].astype(str).str.strip()
actual_df['tag-local-identifier'] = actual_df['tag-local-identifier'].astype(str).str.strip()

gnn_df['forecast_timestamp'] = pd.to_datetime(gnn_df['forecast_timestamp'], errors='coerce')
actual_df['timestamp'] = pd.to_datetime(actual_df['timestamp'], errors='coerce')

# Rename columns for clarity
gnn_df.rename(columns={'forecast_lat': 'pred_lat', 'forecast_lon': 'pred_lon'}, inplace=True)
actual_df.rename(columns={'location-lat': 'actual_lat', 'location-long': 'actual_lon'}, inplace=True)

# Filter for ducks present in both datasets
common_ducks = set(gnn_df['duck_id']) & set(actual_df['tag-local-identifier'])
gnn_filtered = gnn_df[gnn_df['duck_id'].isin(common_ducks)].copy()
actual_filtered = actual_df[actual_df['tag-local-identifier'].isin(common_ducks)].copy()

# Compare actual vs predicted by closest timestamp
results = []

for duck_id in common_ducks:
    actual_duck = actual_filtered[actual_filtered['tag-local-identifier'] == duck_id].sort_values('timestamp')
    pred_duck = gnn_filtered[gnn_filtered['duck_id'] == duck_id].sort_values('forecast_timestamp')
    
    for _, actual_row in actual_duck.iterrows():
        actual_time = actual_row['timestamp']
        closest_pred = pred_duck.iloc[(pred_duck['forecast_timestamp'] - actual_time).abs().argsort()[:1]]
        
        if not closest_pred.empty:
            pred_row = closest_pred.iloc[0]
            distance_km = geodesic(
                (actual_row['actual_lat'], actual_row['actual_lon']),
                (pred_row['pred_lat'], pred_row['pred_lon'])
            ).km
            results.append({
                'duck_id': duck_id,
                'actual_time': actual_time,
                'actual_lat': actual_row['actual_lat'],
                'actual_lon': actual_row['actual_lon'],
                'pred_time': pred_row['forecast_timestamp'],
                'pred_lat': pred_row['pred_lat'],
                'pred_lon': pred_row['pred_lon'],
                'distance_error_km': distance_km
            })

# Create DataFrame of results
comparison_df = pd.DataFrame(results)

# Summary metrics
average_error = comparison_df['distance_error_km'].mean()
within_10km = (comparison_df['distance_error_km'] <= 10).mean() * 100

# Save results to CSV
comparison_df.to_csv("gnn_vs_actual_duck_comparison.csv", index=False)

# Print summary
print("--- GNN vs Actual Comparison Summary ---")
print(f"Total comparisons made: {len(comparison_df)}")
print(f"Average distance error: {average_error:.2f} km")
print(f"Percentage within 10 km: {within_10km:.2f}%")
