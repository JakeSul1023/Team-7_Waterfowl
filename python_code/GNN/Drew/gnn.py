# Author: Drew burkhalter
# Date: 03-30-2025 (Updated 04-16-2025)
# Goal: Model migration as a graph problem using historical stopover sites with weather influence.

import pandas as pd
import numpy as np
import networkx as nx
import requests
import random
from datetime import datetime, timedelta
from functools import lru_cache
import time

OPENWEATHER_API_KEY = "02de0c63b48bcd13d425a73caa22eb81"

@lru_cache(maxsize=10000)
def get_openweather_historical(lat, lon, timestamp):
    """
    Fetch historical or future weather data with mock fallback (cached).
    """
    lat = round(lat, 3)
    lon = round(lon, 3)
    timestamp = str(timestamp)
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather"
        params = {"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY, "units": "metric"}
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        weather_info = {
            "latitude": lat,
            "longitude": lon,
            "temperature": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "pressure": data["main"]["pressure"],
            "wind_speed": data["wind"]["speed"],
            "wind_direction": data["wind"].get("deg", random.uniform(0, 360)),
            "weather_description": data["weather"][0]["description"],
            "timestamp": timestamp,
            "weather_score": 1.0
        }
        print(f"✅ Retrieved weather for ({lat}, {lon}) at {timestamp}: {weather_info['weather_description']}, {weather_info['temperature']}°C")
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to retrieve weather for ({lat}, {lon}) at {timestamp}: {e}")
        print("⚠️ Using mock historical weather.")
        ts = pd.to_datetime(timestamp)
        month = ts.month
        base_temp = 20 - (lat - 40) * 0.3
        seasonal_adj = 5 if month in [8, 9] else 0
        temperature = max(10, min(30, base_temp + seasonal_adj + random.uniform(-3, 3)))
        humidity = min(100, max(40, 60 + (temperature - 20) * 2 + random.uniform(-15, 15)))
        pressure = random.uniform(995, 1025)
        wind_speed = random.uniform(2, 8)
        wind_direction = random.uniform(0, 360)
        weather_score = 1.0
        if not (15 <= temperature <= 25):
            weather_score *= 0.6
        if not (3 <= wind_speed <= 6):
            weather_score *= 0.7
        if not (135 <= wind_direction <= 225):
            weather_score *= 0.5
        weather_info = {
            "latitude": lat,
            "longitude": lon,
            "temperature": temperature,
            "humidity": humidity,
            "pressure": pressure,
            "wind_speed": wind_speed,
            "wind_direction": wind_direction,
            "weather_description": random.choice(["clear sky", "scattered clouds", "light rain"]),
            "timestamp": timestamp,
            "weather_score": weather_score
        }
        print(f"📈 Mocked weather for ({lat}, {lon}) at {timestamp}: {weather_info['weather_description']}, {weather_info['temperature']}°C, Wind={wind_speed:.1f}m/s at {wind_direction:.0f}°, Score={weather_score:.2f}")
    return weather_info

class Duck:
    def __init__(self, duckID):
        self.duckID = duckID
        self.long = []
        self.lat = []
        self.coord = []
        self.timestamps = []
        self.weather_data = []
    
    def importLoc(self, df):
        duck_data = df[df['tag-local-identifier'] == self.duckID]
        self.long = duck_data['location-long'].values
        self.lat = duck_data['location-lat'].values
        self.coord = np.column_stack((self.long, self.lat))
        self.timestamps = duck_data['timestamp'].values
    
    def fetch_weather_data(self, bounds=(24.0, 70.0, -141.0, -52.0), future_timestamps=None):
        if not self.coord.size or not self.timestamps.size:
            raise AttributeError("Duck object is missing 'coord' or 'timestamps'. Did you run importLoc()?")
        min_lat, max_lat, min_lon, max_lon = bounds
        results = []
        coords_ts = set(zip(self.coord[:, 0], self.coord[:, 1], self.timestamps))
        if future_timestamps:
            last_coord = self.coord[-1] if self.coord.size else (0, 0)
            coords_ts.update((last_coord[0], last_coord[1], ts) for ts in future_timestamps)
        print(f"Fetching weather for {len(coords_ts)} locations for Duck {self.duckID}...")
        for lon, lat, timestamp in coords_ts:
            if not (min_lat <= lat <= max_lat and min_lon <= lon <= max_lon):
                print(f"Skipping out-of-bounds location: ({lat}, {lon})")
                continue
            weather = get_openweather_historical(lat, lon, timestamp)
            if weather is None:
                print(f"Skipping invalid weather data for ({lat}, {lon}) at {timestamp}")
                continue
            weather_entry = {
                'duck_id': self.duckID,
                'latitude': lat,
                'longitude': lon,
                'temperature': weather.get('temperature'),
                'pressure': weather.get('pressure'),
                'wind_speed': weather.get('wind_speed'),
                'humidity': weather.get('humidity'),
                'wind_direction': weather.get('wind_direction'),
                'timestamp': timestamp,
                'weather_score': weather.get('weather_score', 0.7)
            }
            results.append(weather_entry)
        self.weather_data = results
        return results

    def get_weather_dataframe(self):
        return pd.DataFrame(self.weather_data) if self.weather_data else pd.DataFrame()

def countDucks(df):
    uniqueIDs = df['tag-local-identifier'].unique()
    return len(uniqueIDs), uniqueIDs.tolist()

def selectDucks(totalDucks, duckList):
    print("Total Number of unique duck IDs imported: ", totalDucks)
    portion = int(input("How many ducks would you like to model with? "))
    return random.sample(duckList, min(portion, totalDucks))

def create_nodes(ducks):
    nodes = set()
    for duck in ducks.values():
        rounded_coords = {(round(lon, 5), round(lat, 5)) for lon, lat in duck.coord}
        nodes.update(rounded_coords)
    return list(nodes)

def normalize_factor(val1, val2, factor):
    ranges = {'temperature': 60, 'pressure': 100, 'wind_speed': 20, 'humidity': 100, 'wind_direction': 360}
    threshold = ranges.get(factor, 5) * 0.05
    diff = abs(val1 - val2)
    if factor == 'wind_direction':
        diff = min(diff, 360 - diff)
    if diff <= threshold:
        return 1.0
    return max(0, min(1, np.exp(-diff / (ranges.get(factor, 10) * 0.3))))

def calculate_edge_weights(edges, weather_data, available_factors, factor_weights):
    """
    Vectorized edge weight calculation with distance penalty.
    """
    if not weather_data:
        return np.full(len(edges), 0.5)
    weather_df = pd.DataFrame(weather_data)
    weights = np.zeros(len(edges))
    for i, (node1, node2) in enumerate(edges):
        dist = np.sqrt((node1[0] - node2[0])**2 + (node1[1] - node2[1])**2)
        dist_weight = max(0.1, 1 - dist / 0.01)
        node1_weather = weather_df[(weather_df['longitude'].round(5) == node1[0]) & 
                                  (weather_df['latitude'].round(5) == node1[1])]
        node2_weather = weather_df[(weather_df['longitude'].round(5) == node2[0]) & 
                                  (weather_df['latitude'].round(5) == node2[1])]
        if node1_weather.empty or node2_weather.empty:
            weights[i] = 0.5 * dist_weight
            continue
        factor_values = []
        for factor in available_factors:
            try:
                val1 = node1_weather[factor].iloc[0]
                val2 = node2_weather[factor].iloc[0]
                normalized = normalize_factor(val1, val2, factor)
                factor_values.append(normalized)
            except (IndexError, KeyError):
                continue
        try:
            score1 = node1_weather['weather_score'].iloc[0]
            score2 = node2_weather['weather_score'].iloc[0]
            score_weight = (score1 + score2) / 2
        except (IndexError, KeyError):
            score_weight = 0.7
        edge_weight = sum(factor_weights[factor] * value for factor, value in zip(available_factors, factor_values)) if factor_values else 0.5
        weights[i] = max(0, min(1, edge_weight * score_weight * dist_weight))
    return weights

def add_edge_weights(G, edge_count, weather_data, ducks, available_factors, factor_weights):
    max_count = max(edge_count.values()) if edge_count else 1
    edges = list(edge_count.keys())
    freq_weights = np.array([count / max_count for count in edge_count.values()])
    weather_weights = calculate_edge_weights(edges, weather_data, available_factors, factor_weights)
    final_weights = 0.5 * freq_weights + 0.5 * weather_weights
    for (node1, node2), weight in zip(edges, final_weights):
        G.add_edge(node1, node2, weight=weight)
    print(f"Added {len(edges)} real edges")
    synthetic_edge_count = 0
    nodes = np.array(list(G.nodes))
    for node1 in nodes:
        node1_tuple = tuple(node1)
        neighbor_count = len(list(G.neighbors(node1_tuple)))
        if neighbor_count >= 10:
            continue
        distances = np.sqrt(np.sum((nodes - node1) ** 2, axis=1))
        close_nodes = nodes[(0 < distances) & (distances <= 0.001)]
        for node2 in close_nodes[:10 - neighbor_count]:
            node2_tuple = tuple(node2)
            if node2_tuple in G.nodes and neighbor_count < 10:
                weather_weight = calculate_edge_weights([(node1_tuple, node2_tuple)], weather_data, available_factors, factor_weights)[0]
                final_weight = 0.5 * 0.1 + 0.5 * weather_weight
                G.add_edge(node1_tuple, node2_tuple, weight=final_weight)
                G.add_edge(node2_tuple, node1_tuple, weight=final_weight)
                synthetic_edge_count += 1
                neighbor_count += 1
    print(f"Added {synthetic_edge_count} synthetic edges")

def predict_next_location(G, current_location, nodes, duck_id):
    current_rounded = (round(current_location[0], 5), round(current_location[1], 5))
    if current_rounded in G.nodes:
        neighbors = list(G.neighbors(current_rounded))
        if neighbors:
            weights = [G[current_rounded][n].get('weight', 1) for n in neighbors]
            total_weight = sum(weights)
            if total_weight == 0:
                return random.choice(neighbors)
            probs = [w / total_weight for w in weights]
            chosen = random.choices(neighbors, weights=probs, k=1)[0]
            dist = np.sqrt((chosen[0] - current_rounded[0])**2 + (chosen[1] - current_rounded[1])**2)
            if dist > 0.01:
                print(f"Warning: Duck {duck_id} large jump at {current_rounded} to {chosen}, dist={dist*111:.2f}km")
            return chosen
    nodes_with_neighbors = [n for n in G.nodes if list(G.neighbors(n))]
    if not nodes_with_neighbors:
        return None
    distances = np.sqrt(((np.array(nodes_with_neighbors) - current_location) ** 2).sum(axis=1))
    nearest_node = nodes_with_neighbors[np.argmin(distances)]
    neighbors = list(G.neighbors(nearest_node))
    weights = [G[nearest_node][n].get('weight', 1) for n in neighbors]
    total_weight = sum(weights)
    if total_weight == 0:
        return random.choice(neighbors)
    probs = [w / total_weight for w in weights]
    return random.choices(neighbors, weights=probs, k=1)[0]

def create_edges(ducks):
    edges = []
    edge_count = {}
    duck_edge_map = {}
    for duck in ducks.values():
        coords = np.round(duck.coord, 5)
        for i in range(len(coords) - 1):
            node1 = tuple(coords[i])
            node2 = tuple(coords[i + 1])
            if node1 != node2:
                edge = (node1, node2)
                edge_count[edge] = edge_count.get(edge, 0) + 1
                duck_edge_map[edge] = duck_edge_map.get(edge, set()).union({duck.duckID})
    return list(edge_count.keys()), duck_edge_map, edge_count

def predict_future_locations(G, duck, nodes, start_time, hours=168):
    """
    Predict locations for the next 7 days (168 hours) without real-time edge updates.
    """
    predictions = []
    current_location = duck.coord[-1] if duck.coord.size else (0, 0)
    current_time = pd.to_datetime(start_time)
    for _ in range(hours):
        predicted = predict_next_location(G, current_location, nodes, duck.duckID)
        if predicted is None:
            print(f"No neighbors for Duck {duck.duckID} at {current_time}, stopping predictions.")
            break
        predictions.append((current_time, predicted))
        current_location = predicted
        current_time += timedelta(hours=1)
    return predictions

def compare_predictions(ducks, predictions, test_df):
    """
    Compare predicted locations to actual locations in test CSV.
    Correct if distance < 10m (~0.0001 degrees).
    """
    correct = 0
    total = 0
    duck_accuracies = {}
    comparison_results = []
    test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
    
    for duck_id, duck in ducks.items():
        duck_correct = 0
        duck_total = 0
        duck_predictions = predictions.get(duck_id, [])
        if not duck_predictions:
            print(f"No predictions for Duck {duck_id}")
            continue
        
        test_data = test_df[test_df['tag-local-identifier'] == duck_id].sort_values('timestamp')
        if test_data.empty:
            print(f"No test data for Duck {duck_id}")
            continue
        
        base_timestamp = pd.to_datetime(duck.timestamps[-1]) if duck.timestamps.size else pd.Timestamp.now()
        start_loc = duck.coord[-1] if duck.coord.size else (0, 0)
        start_lon, start_lat = start_loc
        
        for pred_time, pred_loc in duck_predictions:
            time_diffs = abs(test_data['timestamp'] - pred_time)
            if time_diffs.min() > pd.Timedelta(hours=1):
                print(f"Warning: No test data within 1h of {pred_time} for Duck {duck_id}")
                continue
            closest_idx = time_diffs.idxmin()
            actual = test_data.loc[closest_idx]
            actual_loc = (round(actual['location-long'], 5), round(actual['location-lat'], 5))
            
            dist = np.sqrt((pred_loc[0] - actual_loc[0])**2 + (pred_loc[1] - actual_loc[1])**2)
            is_correct = dist < 0.0001
            
            comparison_results.append({
                'duck_id': duck_id,
                'base_timestamp': base_timestamp,
                'forecast_timestamp': pred_time,
                'start_lat': start_lat,
                'start_lon': start_lon,
                'forecast_lat': pred_loc[1],
                'forecast_lon': pred_loc[0],
                'distance': dist,
                'correct': is_correct
            })
            
            if is_correct:
                correct += 1
                duck_correct += 1
            total += 1
            duck_total += 1
        
        duck_accuracies[duck_id] = (duck_correct / duck_total) * 100 if duck_total > 0 else 0
    
    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"\nPrediction Accuracy: {accuracy:.2f}%")
    for duck_id, acc in duck_accuracies.items():
        print(f"Duck {duck_id} Accuracy: {acc:.2f}%")
    
    comparison_df = pd.DataFrame(comparison_results)
    print("\nComparison Results (first 10):")
    print(comparison_df[['duck_id', 'base_timestamp', 'forecast_timestamp', 'start_lat', 'start_lon', 
                        'forecast_lat', 'forecast_lon', 'distance', 'correct']].head(10))
    
    return accuracy, comparison_df

if __name__ == "__main__":
    start = time.time()
    df = pd.read_csv("ShortTermSetData(Aug-Sept).csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    total, uniqueIDs = countDucks(df)
    sampleDucks = selectDucks(total, uniqueIDs)
    ducks = {duck_id: Duck(duck_id) for duck_id in sampleDucks}
    for duck in ducks.values():
        duck.importLoc(df)
    
    print("Fetching weather data for sample ducks...")
    all_weather_data = []
    for duck_id, duck in ducks.items():
        duck_weather = duck.fetch_weather_data()
        all_weather_data.extend(duck_weather)
        weather_df = duck.get_weather_dataframe()
        if not weather_df.empty:
            print(f"Weather data for Duck {duck_id} (first 5 rows):")
            print(weather_df[['latitude', 'longitude', 'temperature', 'wind_speed', 'timestamp']].head())
    
    G = nx.DiGraph()
    nodes = create_nodes(ducks)
    G.add_nodes_from(nodes)
    edges, duck_edge_map, edge_count = create_edges(ducks)
    
    factor_weights = {
        'pressure': 0.2,
        'wind_speed': 0.2,
        'humidity': 0.2,
        'temperature': 0.2,
        'wind_direction': 0.2
    }
    available_factors = list(factor_weights.keys())
    
    print("Building graph with weather-influenced weights...")
    add_edge_weights(G, edge_count, all_weather_data, ducks, available_factors, factor_weights)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    neighbor_counts = [len(list(G.neighbors(n))) for n in G.nodes]
    print(f"Average neighbors per node: {np.mean(neighbor_counts):.2f}, Max: {max(neighbor_counts, default=0)}")
    
    print("\nPredicting next 7 days (168 hours) for each duck...")
    predictions = {}
    start_time = max([pd.to_datetime(duck.timestamps[-1]) for duck in ducks.values()]) + timedelta(hours=1)
    for duck_id, duck in ducks.items():
        duck_predictions = predict_future_locations(G, duck, nodes, start_time)
        predictions[duck_id] = duck_predictions
        print(f"\nDuck {duck_id} Predictions (first 10):")
        for ts, loc in duck_predictions[:10]:
            print(f"  {ts}: ({loc[0]}, {loc[1]})")
    
    print("\nLoading test data from Mallard Connectivity(2).csv...")
    test_df = pd.read_csv("Mallard Connectivity(2).csv")
    test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
    
    print("Comparing predictions to actual locations...")
    accuracy, comparison_df = compare_predictions(ducks, predictions, test_df)
    
    comparison_df = comparison_df[['duck_id', 'base_timestamp', 'forecast_timestamp', 
                                   'start_lat', 'start_lon', 'forecast_lat', 
                                   'forecast_lon', 'distance', 'correct']]
    comparison_df.to_csv("prediction_comparison.csv", index=False)
    print("\nComparison results saved to prediction_comparison.csv")
    print(f"Runtime: {time.time() - start:.2f}s")
