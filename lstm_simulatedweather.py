import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import os
import math
from datetime import datetime, timedelta
from matplotlib.patches import Ellipse
from scipy.ndimage import gaussian_filter

np.random.seed(42)
plt.style.use('ggplot')
sns.set(style="whitegrid")

def haversine_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c

def load_duck_data(file_path):
    print(f"Loading duck data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.rename(columns={
            'location-long': 'longitude', 'location-lat': 'latitude',
            'external-temperature': 'temperature', 'ground-speed': 'speed',
            'heading': 'direction', 'tag-local-identifier': 'tag_id',
            'individual-local-identifier': 'duck_id'
        })
        df = df.sort_values(['duck_id', 'timestamp'])
        df['time_diff'] = df.groupby('duck_id')['timestamp'].diff().dt.total_seconds() / 3600
        df['prev_lat'] = df.groupby('duck_id')['latitude'].shift(1)
        df['prev_lon'] = df.groupby('duck_id')['longitude'].shift(1)
        df['distance'] = df.apply(
            lambda row: haversine_distance(row['prev_lat'], row['prev_lon'], 
                                          row['latitude'], row['longitude']) 
            if not (pd.isna(row['prev_lat']) or pd.isna(row['prev_lon'])) else np.nan, 
            axis=1
        )
        df['day_of_year'] = df['timestamp'].dt.day_of_year
        df['hour'] = df['timestamp'].dt.hour
        df['is_daytime'] = ((df['hour'] >= 6) & (df['hour'] <= 18)).astype(int)
        print(f"Processed {len(df)} records for {df['duck_id'].nunique()} ducks")
        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def create_simulated_weather_grids(df, grid_size=50, days=30):
    min_lon, max_lon = df['longitude'].min() - 1, df['longitude'].max() + 1
    min_lat, max_lat = df['latitude'].min() - 1, df['latitude'].max() + 1
    bounds = (min_lon, min_lat, max_lon, max_lat)
    
    start_date = df['timestamp'].min().date()
    weather_dates = [start_date + timedelta(days=i) for i in range(days)]
    
    weather_data = {
        'bounds': bounds, 'dates': weather_dates, 'grid_size': grid_size,
        'temperature': {}, 'precipitation': {}, 'wind_u': {}, 'wind_v': {}
    }
    
    for date in weather_dates:
        day_of_year = date.timetuple().tm_yday
        seasonal_factor = math.sin((day_of_year - 15) / 365 * 2 * math.pi) * 15
        
        temp_grid = np.zeros((grid_size, grid_size))
        precip_grid = np.zeros((grid_size, grid_size))
        wind_u_grid = np.zeros((grid_size, grid_size))
        wind_v_grid = np.zeros((grid_size, grid_size))
        
        for y in range(grid_size):
            for x in range(grid_size):
                lon = min_lon + (max_lon - min_lon) * x / grid_size
                lat = min_lat + (max_lat - min_lat) * y / grid_size
                
                base_temp = 15
                lat_factor = -1 * (lat - min_lat) / (max_lat - min_lat) * 10
                random_factor = np.random.normal(0, 2)
                temp = base_temp + seasonal_factor + lat_factor + random_factor
                temp_grid[y, x] = temp
                
                if date.day % 5 == 0:
                    if np.random.random() < 0.7:
                        precip_grid[y, x] = np.random.gamma(1, 5)
                else:
                    if np.random.random() < 0.2:
                        precip_grid[y, x] = np.random.gamma(0.5, 2)
                
                season = "summer" if seasonal_factor > 0 else "winter"
                if season == "summer":
                    wind_u_grid[y, x] = np.random.normal(0, 5)
                    wind_v_grid[y, x] = np.random.normal(10, 5)
                else:
                    wind_u_grid[y, x] = np.random.normal(0, 5)
                    wind_v_grid[y, x] = np.random.normal(-10, 5)
        
        temp_grid = gaussian_filter(temp_grid, sigma=1)
        precip_grid = gaussian_filter(precip_grid, sigma=2)
        wind_u_grid = gaussian_filter(wind_u_grid, sigma=3)
        wind_v_grid = gaussian_filter(wind_v_grid, sigma=3)
        
        date_str = date.strftime('%Y%m%d')
        weather_data['temperature'][date_str] = temp_grid
        weather_data['precipitation'][date_str] = precip_grid
        weather_data['wind_u'][date_str] = wind_u_grid
        weather_data['wind_v'][date_str] = wind_v_grid
    
    print(f"Generated {len(weather_dates)} days of simulated weather data")
    return weather_data

def visualize_weather_grids(weather_data, output_dir="weather_visuals"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    grid_size = weather_data['grid_size']
    min_lon, min_lat, max_lon, max_lat = weather_data['bounds']
    dates = weather_data['dates']
    
    lons = np.linspace(min_lon, max_lon, grid_size)
    lats = np.linspace(min_lat, max_lat, grid_size)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    sample_dates = dates[::5]
    
    for date in sample_dates:
        date_str = date.strftime('%Y%m%d')
        fig, axes = plt.subplots(2, 2, figsize=(18, 15))
        
        ax = axes[0, 0]
        temp_grid = weather_data['temperature'][date_str]
        im = ax.pcolormesh(lon_grid, lat_grid, temp_grid, cmap='coolwarm', shading='auto')
        plt.colorbar(im, ax=ax, label='Temperature (°C)')
        ax.set_title(f'Temperature - {date.strftime("%Y-%m-%d")}')
        
        ax = axes[0, 1]
        precip_grid = weather_data['precipitation'][date_str]
        im = ax.pcolormesh(lon_grid, lat_grid, precip_grid, cmap='Blues', shading='auto')
        plt.colorbar(im, ax=ax, label='Precipitation (mm)')
        ax.set_title(f'Precipitation - {date.strftime("%Y-%m-%d")}')
        
        ax = axes[1, 0]
        wind_u = weather_data['wind_u'][date_str]
        wind_v = weather_data['wind_v'][date_str]
        
        wind_speed = np.sqrt(wind_u**2 + wind_v**2)
        im = ax.pcolormesh(lon_grid, lat_grid, wind_speed, cmap='YlGnBu', shading='auto')
        plt.colorbar(im, ax=ax, label='Wind Speed (m/s)')
        
        step = max(1, grid_size // 15)
        ax.quiver(lon_grid[::step, ::step], lat_grid[::step, ::step], 
                 wind_u[::step, ::step], wind_v[::step, ::step],
                 angles='xy', scale_units='inches', scale=15)
        
        ax.set_title(f'Wind - {date.strftime("%Y-%m-%d")}')
        
        ax = axes[1, 1]
        im = ax.contourf(lon_grid, lat_grid, temp_grid, cmap='coolwarm', levels=15)
        plt.colorbar(im, ax=ax, label='Temperature (°C)')
        
        precip_mask = precip_grid > 2.0
        ax.scatter(lon_grid[precip_mask], lat_grid[precip_mask], 
                  c='blue', s=1, alpha=0.5, marker='.')
        
        ax.quiver(lon_grid[::step, ::step], lat_grid[::step, ::step], 
                 wind_u[::step, ::step], wind_v[::step, ::step],
                 angles='xy', scale_units='inches', scale=15, color='black')
        
        ax.set_title(f'Combined Weather - {date.strftime("%Y-%m-%d")}')
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"weather_{date_str}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    print(f"Created weather visualizations for {len(sample_dates)} days")

def extract_simulated_weather(df, weather_data):
    print("Extracting simulated weather data at duck locations...")
    
    df_with_weather = df.copy()
    
    df_with_weather['sim_temp'] = np.nan
    df_with_weather['sim_precip'] = np.nan
    df_with_weather['sim_wind_u'] = np.nan
    df_with_weather['sim_wind_v'] = np.nan
    df_with_weather['sim_wind_speed'] = np.nan
    
    min_lon, min_lat, max_lon, max_lat = weather_data['bounds']
    grid_size = weather_data['grid_size']
    
    for idx, row in df_with_weather.iterrows():
        record_date = row['timestamp'].date()
        date_str = record_date.strftime('%Y%m%d')
        
        if date_str not in weather_data['temperature']:
            continue
        
        lon, lat = row['longitude'], row['latitude']
        
        if lon < min_lon or lon > max_lon or lat < min_lat or lat > max_lat:
            continue
        
        x_idx = int((lon - min_lon) / (max_lon - min_lon) * (grid_size - 1))
        y_idx = int((lat - min_lat) / (max_lat - min_lat) * (grid_size - 1))
        
        try:
            df_with_weather.at[idx, 'sim_temp'] = weather_data['temperature'][date_str][y_idx, x_idx]
            df_with_weather.at[idx, 'sim_precip'] = weather_data['precipitation'][date_str][y_idx, x_idx]
            df_with_weather.at[idx, 'sim_wind_u'] = weather_data['wind_u'][date_str][y_idx, x_idx]
            df_with_weather.at[idx, 'sim_wind_v'] = weather_data['wind_v'][date_str][y_idx, x_idx]
            
            df_with_weather.at[idx, 'sim_wind_speed'] = math.sqrt(
                df_with_weather.at[idx, 'sim_wind_u']**2 + 
                df_with_weather.at[idx, 'sim_wind_v']**2
            )
        except Exception as e:
            print(f"Error extracting weather at ({lon}, {lat}): {str(e)}")
    
    print(f"Added simulated weather data to {len(df_with_weather)} duck locations")
    return df_with_weather

def visualize_duck_weather_relationships(df, save_path="duck_weather_relationships.png"):
    weather_cols = ['sim_temp', 'sim_precip', 'sim_wind_speed']
    if not all(col in df.columns for col in weather_cols):
        print("No simulated weather data available for visualization")
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 15))
    
    ax = axes[0, 0]
    temp_dist = df[['sim_temp', 'distance']].dropna()
    if len(temp_dist) > 0:
        dist_upper = np.percentile(temp_dist['distance'], 95)
        filtered_data = temp_dist[temp_dist['distance'] <= dist_upper]
        
        sns.scatterplot(x='sim_temp', y='distance', data=filtered_data, 
                      alpha=0.5, edgecolor=None, ax=ax)
        sns.regplot(x='sim_temp', y='distance', data=filtered_data, 
                   scatter=False, color='red', ax=ax)
        
        ax.set_title('Movement Distance vs. Temperature')
    
    ax = axes[0, 1]
    precip_dist = df[['sim_precip', 'distance']].dropna()
    if len(precip_dist) > 0:
        dist_upper = np.percentile(precip_dist['distance'], 95)
        precip_upper = np.percentile(precip_dist['sim_precip'], 95)
        filtered_data = precip_dist[
            (precip_dist['distance'] <= dist_upper) & 
            (precip_dist['sim_precip'] <= precip_upper)
        ]
        
        sns.scatterplot(x='sim_precip', y='distance', data=filtered_data, 
                      alpha=0.5, edgecolor=None, ax=ax)
        sns.regplot(x='sim_precip', y='distance', data=filtered_data, 
                   scatter=False, color='red', ax=ax)
        
        ax.set_title('Movement Distance vs. Precipitation')
    
    ax = axes[1, 0]
    wind_dist = df[['sim_wind_speed', 'distance']].dropna()
    if len(wind_dist) > 0:
        dist_upper = np.percentile(wind_dist['distance'], 95)
        filtered_data = wind_dist[wind_dist['distance'] <= dist_upper]
        
        sns.scatterplot(x='sim_wind_speed', y='distance', data=filtered_data, 
                      alpha=0.5, edgecolor=None, ax=ax)
        sns.regplot(x='sim_wind_speed', y='distance', data=filtered_data, 
                   scatter=False, color='red', ax=ax)
        
        ax.set_title('Movement Distance vs. Wind Speed')
    
    ax = axes[1, 1]
    weather_dist = df[['sim_temp', 'sim_precip', 'sim_wind_speed', 'distance']].dropna()
    if len(weather_dist) > 0:
        dist_upper = np.percentile(weather_dist['distance'], 95)
        filtered_data = weather_dist[weather_dist['distance'] <= dist_upper]
        
        scatter = ax.scatter(
            filtered_data['sim_wind_speed'], 
            filtered_data['distance'],
            c=filtered_data['sim_temp'],
            s=filtered_data['sim_precip'] + 5,
            alpha=0.7,
            cmap='coolwarm'
        )
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Temperature (°C)')
        
        handles, labels = [], []
        for precip in [0, 5, 10, 20]:
            handles.append(plt.scatter([], [], s=precip+5, c='gray'))
            labels.append(f'{precip} mm')
        ax.legend(handles, labels, title='Precipitation', loc='upper right')
        
        ax.set_title('Combined Weather Effects on Movement')
    
    plt.tight_layout()
    
    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Weather relationship visualization saved to {save_path}")
    except Exception as e:
        print(f"Error saving figure: {str(e)}")
    
    return fig

def visualize_movement_patterns(df, save_path="movement_patterns.png"):
    fig, axes = plt.subplots(2, 2, figsize=(18, 15))
    
    ax = axes[0, 0]
    duck_ids = df['duck_id'].unique()
    sample_size = min(5, len(duck_ids))
    sample_duck_ids = np.random.choice(duck_ids, sample_size, replace=False)
    
    for duck_id in sample_duck_ids:
        duck_data = df[df['duck_id'] == duck_id].sort_values('timestamp')
        ax.plot(duck_data['longitude'], duck_data['latitude'], 'o-', alpha=0.7, 
               linewidth=1, markersize=4, label=f"Duck {duck_id}")
    
    ax.set_title('Duck Movement Trajectories')
    ax.legend()
    
    ax = axes[0, 1]
    distance_data = df['distance'].dropna()
    if len(distance_data) > 0:
        upper_limit = np.percentile(distance_data, 95)
        filtered_distances = distance_data[distance_data <= upper_limit]
        
        sns.histplot(filtered_distances, bins=30, kde=True, ax=ax)
        
        mean_dist = filtered_distances.mean()
        median_dist = filtered_distances.median()
        ax.axvline(mean_dist, color='red', linestyle='--', 
                  label=f'Mean: {mean_dist:.2f} km')
        ax.axvline(median_dist, color='green', linestyle='--', 
                  label=f'Median: {median_dist:.2f} km')
        
        ax.set_title('Distribution of Movement Distances')
        ax.legend()
    
    ax = axes[1, 0]
    if 'hour' in df.columns and 'distance' in df.columns:
        hourly_stats = df.groupby('hour')['distance'].agg(['mean', 'count']).reset_index()
        
        if len(hourly_stats) > 0:
            ax.bar(hourly_stats['hour'], hourly_stats['mean'], 
                  yerr=hourly_stats['mean'] / np.sqrt(hourly_stats['count']),
                  capsize=5, alpha=0.7)
            
            ax.set_title('Movement Distance by Hour of Day')
            ax.axvspan(0, 6, alpha=0.2, color='navy', label='Night')
            ax.axvspan(6, 18, alpha=0.1, color='yellow', label='Day')
            ax.axvspan(18, 24, alpha=0.2, color='navy')
            ax.legend()
    
    ax = axes[1, 1]
    if 'day_of_year' in df.columns and 'distance' in df.columns:
        daily_stats = df.groupby('day_of_year')['distance'].mean().reset_index()
        
        if len(daily_stats) > 10:
            ax.plot(daily_stats['day_of_year'], daily_stats['distance'], 
                   'o-', alpha=0.7, linewidth=1)
            
            from scipy.signal import savgol_filter
            if len(daily_stats) > 10:
                window_size = min(15, len(daily_stats) - (len(daily_stats) % 2) - 1)
                if window_size > 3:
                    try:
                        yhat = savgol_filter(daily_stats['distance'], window_size, 3)
                        ax.plot(daily_stats['day_of_year'], yhat, 'r-', linewidth=2)
                    except Exception as e:
                        print(f"Error creating trend line: {str(e)}")
            
            ax.set_title('Seasonal Movement Patterns')
            
            seasons = [(0, 'Winter'), (79, 'Spring'), (171, 'Summer'), (265, 'Fall'), (355, 'Winter')]
            for i in range(len(seasons)-1):
                start_day = seasons[i][0]
                end_day = seasons[i+1][0]
                season_name = seasons[i][1]
                
                colors = {'Winter': 'lightblue', 'Spring': 'lightgreen', 
                         'Summer': 'yellow', 'Fall': 'orange'}
                
                ax.axvspan(start_day, end_day, alpha=0.2, color=colors[season_name])
    
    plt.tight_layout()
    
    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Movement patterns visualization saved to {save_path}")
    except Exception as e:
        print(f"Error saving figure: {str(e)}")
    
    return fig

def create_migration_prediction_visuals(df, weather_data, save_path="migration_prediction.png"):
    fig, axes = plt.subplots(2, 2, figsize=(18, 15))
    
    min_lon, min_lat, max_lon, max_lat = weather_data['bounds']
    grid_size = weather_data['grid_size']
    
    lons = np.linspace(min_lon, max_lon, grid_size)
    lats = np.linspace(min_lat, max_lat, grid_size)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    sample_date = weather_data['dates'][len(weather_data['dates'])//2]
    date_str = sample_date.strftime('%Y%m%d')
    
    ax = axes[0, 0]
    temp_grid = weather_data['temperature'][date_str]
    im = ax.pcolormesh(lon_grid, lat_grid, temp_grid, cmap='coolwarm', alpha=0.7, shading='auto')
    plt.colorbar(im, ax=ax, label='Temperature (°C)')
    
    closest_date = min(df['timestamp'].dt.date.unique(), 
                     key=lambda x: abs((x - sample_date).days))
    day_ducks = df[df['timestamp'].dt.date == closest_date]
    
    ax.scatter(day_ducks['longitude'], day_ducks['latitude'], 
              color='black', s=30, marker='o', label='Duck Locations')
    
    ax.set_title(f'Duck Locations with Temperature - {sample_date.strftime("%Y-%m-%d")}')
    ax.legend()
    
    ax = axes[0, 1]
    precip_grid = weather_data['precipitation'][date_str]
    im = ax.pcolormesh(lon_grid, lat_grid, precip_grid, cmap='Blues', alpha=0.7, shading='auto')
    plt.colorbar(im, ax=ax, label='Precipitation (mm)')
    
    wind_u = weather_data['wind_u'][date_str]
    wind_v = weather_data['wind_v'][date_str]
    
    step = max(1, grid_size // 15)
    ax.quiver(lon_grid[::step, ::step], lat_grid[::step, ::step], 
             wind_u[::step, ::step], wind_v[::step, ::step],
             color='black', alpha=0.7, scale=300)
    
    day_ducks_with_prev = day_ducks.dropna(subset=['prev_lat', 'prev_lon'])
    
    if len(day_ducks_with_prev) > 0:
        ax.quiver(day_ducks_with_prev['prev_lon'], day_ducks_with_prev['prev_lat'],
                 day_ducks_with_prev['longitude'] - day_ducks_with_prev['prev_lon'],
                 day_ducks_with_prev['latitude'] - day_ducks_with_prev['prev_lat'],
                 color='red', scale=0.1, width=0.005, label='Actual Movement')
    
    ax.set_title(f'Movement Vectors with Weather - {sample_date.strftime("%Y-%m-%d")}')
    ax.legend()
    
    ax = axes[1, 0]
    risk_grid = np.zeros_like(temp_grid)
    temp_optimal = 15
    risk_grid += np.abs(temp_grid - temp_optimal) / 10
    risk_grid += precip_grid / 5
    
    season = "spring" if sample_date.month >= 3 and sample_date.month <= 6 else "fall"
    preferred_v = 1 if season == "spring" else -1
    
    wind_alignment = wind_v * preferred_v
    risk_grid -= wind_alignment / 5
    risk_grid = np.maximum(0, risk_grid)
    
    im = ax.pcolormesh(lon_grid, lat_grid, risk_grid, cmap='YlOrRd', alpha=0.7, shading='auto')
    plt.colorbar(im, ax=ax, label='Migration Risk')
    
    if len(day_ducks) > 0:
        sample_size = min(3, len(day_ducks))
        start_ducks = day_ducks.sample(sample_size)
        
        for i, (_, duck) in enumerate(start_ducks.iterrows()):
            path_lon = [duck['longitude']]
            path_lat = [duck['latitude']]
            
            for step in range(10):
                curr_lon, curr_lat = path_lon[-1], path_lat[-1]
                
                x_idx = int((curr_lon - min_lon) / (max_lon - min_lon) * (grid_size - 1))
                y_idx = int((curr_lat - min_lat) / (max_lat - min_lat) * (grid_size - 1))
                
                x_idx = max(0, min(x_idx, grid_size-1))
                y_idx = max(0, min(y_idx, grid_size-1))
                
                min_risk = float('inf')
                best_move = (0, 0)
                
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        new_x = x_idx + dx
                        new_y = y_idx + dy
                        
                        if new_x < 0 or new_x >= grid_size or new_y < 0 or new_y >= grid_size:
                            continue
                        
                        loc_risk = risk_grid[new_y, new_x]
                        
                        directional_bias = -dy if season == "spring" else dy
                        loc_risk += directional_bias
                        
                        if loc_risk < min_risk:
                            min_risk = loc_risk
                            best_move = (dx, dy)
                
                move_lon = best_move[0] * (max_lon - min_lon) / grid_size
                move_lat = best_move[1] * (max_lat - min_lat) / grid_size
                
                path_lon.append(curr_lon + move_lon)
                path_lat.append(curr_lat + move_lat)
            
            ax.plot(path_lon, path_lat, 'o-', linewidth=2, markersize=5,
                   label=f'Simulated Path {i+1}')
    
    ax.set_title('Simulated Optimal Migration Paths')
    ax.legend()
    
    ax = axes[1, 1]
    ax.grid(True, alpha=0.3)
    
    if len(day_ducks) > 0:
        sample_duck = day_ducks.iloc[0]
        
        current_lon, current_lat = sample_duck['longitude'], sample_duck['latitude']
        ax.scatter(current_lon, current_lat, s=100, color='blue', 
                  marker='o', label='Current Position')
        
        duck_history = df[df['duck_id'] == sample_duck['duck_id']].sort_values('timestamp')
        
        if len(duck_history) >= 5:
            recent_history = duck_history.tail(5)
            avg_lon_change = recent_history['longitude'].diff().mean()
            avg_lat_change = recent_history['latitude'].diff().mean()
            
            if pd.isna(avg_lon_change) or pd.isna(avg_lat_change):
                avg_lon_change = 0.05
                avg_lat_change = 0.05
            
            forecast_days = [1, 3, 7, 14]
            for days in forecast_days:
                forecast_lon = current_lon + avg_lon_change * days
                forecast_lat = current_lat + avg_lat_change * days
                
                uncertainty = 0.05 * np.sqrt(days)
                
                ax.scatter(forecast_lon, forecast_lat, 
                          s=50, color='red', alpha=0.7,
                          label=f"{days}-Day Forecast")
                
                ellipse = Ellipse(
                    (forecast_lon, forecast_lat),
                    width=uncertainty * 2,
                    height=uncertainty * 2,
                    alpha=0.2,
                    color='red'
                )
                ax.add_patch(ellipse)
                
                ax.text(forecast_lon, forecast_lat, f"{days}d",
                       ha='center', va='center', fontsize=10)
    
    ax.set_title('Position Forecast with Uncertainty')
    ax.legend()
    
    plt.tight_layout()
    
    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Migration prediction visualization saved to {save_path}")
    except Exception as e:
        print(f"Error saving figure: {str(e)}")
    
    return fig

def main():
    file_path = r"C:\Users\brean\Desktop\lstm\Mallard Connectivity_Recent_Data.csv"
    
    output_dir = r"C:\Users\brean\Desktop\lstm"
    
    
    df = load_duck_data(file_path)
    
    if df is None or len(df) == 0:
        print("No data to analyze. Please check the file path.")
        return
    
    visualize_movement_patterns(df, save_path=os.path.join(output_dir, "movement_patterns.png"))
    
    weather_data = create_simulated_weather_grids(df, grid_size=50, days=30)
    
    visualize_weather_grids(weather_data, output_dir=os.path.join(output_dir, "weather_grids"))
    
    df_with_weather = extract_simulated_weather(df, weather_data)
    
    visualize_duck_weather_relationships(df_with_weather, 
                                       save_path=os.path.join(output_dir, "weather_relationships.png"))
    
    create_migration_prediction_visuals(df_with_weather, weather_data,
                                      save_path=os.path.join(output_dir, "migration_prediction.png"))
    
    print("\nAnalysis complete! All visualizations saved to the 'mallard_analysis_output' directory.")

if __name__ == "__main__":
    main()
