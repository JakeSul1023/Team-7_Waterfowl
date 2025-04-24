import pandas as pd
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load datasets
gnn_df = pd.read_csv("M30_A06_P.csv")
actual_df = pd.read_csv("SpringTest.csv")

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
max_error = comparison_df['distance_error_km'].max()
min_error = comparison_df['distance_error_km'].min()
std_error = comparison_df['distance_error_km'].std()

# Save results to CSV
comparison_df.to_csv("gnn_vs_actual_duck_comparison.csv", index=False)

# Print summary
print("\n--- GNN vs Actual Comparison Summary ---")
print(f"Total comparisons made: {len(comparison_df)}")
print(f"Average distance error: {average_error:.2f} km")
print(f"Max error: {max_error:.2f} km")
print(f"Min error: {min_error:.2f} km")
print(f"Standard deviation: {std_error:.2f} km")
print(f"Percentage within 10 km: {within_10km:.2f}%")

# Custom colors
navy_green = "#2b4c3f"
black = "#000000"

# Set seaborn style
sns.set_style("whitegrid")
sns.set_palette([navy_green, black])

# Histogram with KDE overlay
plt.figure(figsize=(10, 6))
sns.histplot(comparison_df['distance_error_km'], bins=50, kde=False, color=navy_green, edgecolor='black')
sns.kdeplot(comparison_df['distance_error_km'], color=black, linewidth=2)
plt.title("Distribution of Prediction Errors (km)", fontsize=14, color=navy_green)
plt.xlabel("Distance Error (km)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig("error_distribution_histogram.png")
plt.show()

# Boxplot
plt.figure(figsize=(8, 4))
sns.boxplot(x=comparison_df['distance_error_km'], color=navy_green)
plt.title("Boxplot of Prediction Errors (km)", fontsize=14, color=navy_green)
plt.xlabel("Distance Error (km)", fontsize=12)
plt.tight_layout()
plt.savefig("error_boxplot.png")
plt.show()

# Bin edges and labels
bin_edges = [0, 10, 25, 50, 100, 250, 500, float("inf")]
bin_labels = ["0-10", "10-25", "25-50", "50-100", "100-250", "250-500", "500+"]

# Assign bins
comparison_df['error_bin'] = pd.cut(
    comparison_df['distance_error_km'],
    bins=bin_edges,
    labels=bin_labels,
    include_lowest=True
)

# Bar chart of bin frequencies
bin_counts = comparison_df['error_bin'].value_counts().sort_index()

plt.figure(figsize=(10, 6))
sns.barplot(x=bin_counts.index, y=bin_counts.values, color=navy_green, edgecolor=black)
plt.title("Frequency of Distance Errors by Bin", fontsize=14, color=navy_green)
plt.xlabel("Distance Error Bin (km)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig("error_bin_barplot.png")
plt.show()



# Optional: per-duck average error
duck_avg = comparison_df.groupby('duck_id')['distance_error_km'].mean().sort_values(ascending=False)
print("\nTop 5 Ducks with Highest Average Error:")
print(duck_avg.head())
