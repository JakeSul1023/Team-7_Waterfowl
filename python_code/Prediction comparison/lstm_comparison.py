import pandas as pd
import numpy as np
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
actual_df = pd.read_csv("Mallard Connectivity(2).csv")
predicted_df = pd.read_csv("duck_migration_predictions_lstm.csv")

# Standardize columns
actual_df.columns = actual_df.columns.str.lower().str.replace(" ", "_")
predicted_df.columns = predicted_df.columns.str.lower().str.replace(" ", "_")

# Rename for consistency
actual_df = actual_df.rename(columns={'tag-local-identifier': 'duck_id', 'location-lat': 'actual_lat', 'location-long': 'actual_lon'})
predicted_df = predicted_df.rename(columns={'pred_lat': 'predicted_lat', 'pred_lon': 'predicted_lon'})

# Parse timestamps
actual_df['timestamp'] = pd.to_datetime(actual_df['timestamp'], errors='coerce')
predicted_df['timestamp'] = pd.to_datetime(predicted_df['timestamp'], errors='coerce')

# Drop missing data
actual_df = actual_df.dropna(subset=['timestamp', 'actual_lat', 'actual_lon', 'duck_id'])
predicted_df = predicted_df.dropna(subset=['timestamp', 'predicted_lat', 'predicted_lon', 'duck_id'])

# Ensure ID consistency
actual_df['duck_id'] = actual_df['duck_id'].astype(str)
predicted_df['duck_id'] = predicted_df['duck_id'].astype(str)

# Grouping and comparison
grouped_actual = actual_df.groupby('duck_id')
grouped_predicted = predicted_df.groupby('duck_id')

results = []

for duck_id, pred_group in grouped_predicted:
    if duck_id not in grouped_actual.groups:
        continue
    actual_group = grouped_actual.get_group(duck_id)
    for _, pred_row in pred_group.iterrows():
        closest_idx = (actual_group['timestamp'] - pred_row['timestamp']).abs().idxmin()
        actual_row = actual_group.loc[closest_idx]
        pred_coords = (pred_row['predicted_lat'], pred_row['predicted_lon'])
        actual_coords = (actual_row['actual_lat'], actual_row['actual_lon'])
        dist_km = geodesic(pred_coords, actual_coords).kilometers
        results.append({
            'duck_id': duck_id,
            'timestamp': pred_row['timestamp'],
            'pred_lat': pred_row['predicted_lat'],
            'pred_lon': pred_row['predicted_lon'],
            'actual_lat': actual_row['actual_lat'],
            'actual_lon': actual_row['actual_lon'],
            'distance_error_km': dist_km
        })

# Convert to DataFrame
comparison_df = pd.DataFrame(results)

# Save to CSV
comparison_df.to_csv("lstm_comparison_results.csv", index=False)

# Summary stats
avg_error = comparison_df['distance_error_km'].mean()
within_10km = (comparison_df['distance_error_km'] <= 10).mean() * 100
print(f"\nTotal comparisons: {len(comparison_df)}")
print(f"Average distance error: {avg_error:.2f} km")
print(f"Predictions within 10 km: {within_10km:.2f}%")

# Histogram of errors
plt.figure(figsize=(10, 6))
sns.histplot(comparison_df['distance_error_km'], bins=50, kde=True)
plt.title("LSTM Prediction Error Distribution")
plt.xlabel("Distance Error (km)")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.savefig("lstm_error_histogram.png")
plt.show()

# Optional: Visualize predicted vs actual path for a duck
sample_duck = comparison_df['duck_id'].value_counts().idxmax()
duck_path = comparison_df[comparison_df['duck_id'] == sample_duck]

plt.figure(figsize=(10, 6))
plt.plot(duck_path['actual_lon'], duck_path['actual_lat'], marker='o', label="Actual Path", color='blue')
plt.plot(duck_path['pred_lon'], duck_path['pred_lat'], marker='x', label="Predicted Path", color='red')
plt.title(f"Path Comparison for Duck {sample_duck}")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"duck_{sample_duck}_path_comparison.png")
plt.show()
