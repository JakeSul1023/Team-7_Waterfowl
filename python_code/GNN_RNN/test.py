import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import requests
from sklearn.preprocessing import StandardScaler
from math import radians, sin, cos, sqrt, atan2

# Haversine distance (km)
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

# Weather API functions
def get_weather(lat, lon):
    url = f"https://api.weather.gov/points/{lat},{lon}"
    headers = {"User-Agent": "WaterfowlMigration", "Accept": "application/json"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            print("URL success")
            data = response.json()
            forecast_url = data['properties']['forecast']
            forecast_response = requests.get(forecast_url, headers=headers, timeout=10)
            if forecast_response.status_code == 200:
                print("Forecast success")
                return forecast_response.json()
            else:
                print("Forecast fail")
                return None
    except Exception as e:
        print(f"Error retrieving weather: {e}")
        return None

def process_weather(weather_raw):
    if not weather_raw:
        return {}
    periods = weather_raw['properties']['periods']
    first_period = periods[0] if periods else {}
    return {
        'temperature': first_period.get('temperature', np.nan),
        'wind_speed': int(first_period.get('windSpeed', '0 mph').split()[0]),
        'precipitation_probability': first_period.get('probabilityOfPrecipitation', {}).get('value', np.nan),
    }
 
# Load and preprocess data
def load_data(file_path, num_ducks=5, miles_limit=50):
    print("Loading CSV data")
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    print(f"Initial rows: {len(df)}")

    # Get the first 5 duck IDs
    duck_ids = df['tag-local-identifier'].unique()[:num_ducks]
    print(f"Selected duck IDs: {duck_ids}")
    df = df[df['tag-local-identifier'].isin(duck_ids)].copy()
    print(f"Rows after filtering to {num_ducks} ducks: {len(df)}")

    # Calculate cumulative distance
    df['distance_km'] = 0.0
    df['cumulative_distance_km'] = 0.0
    for duck in duck_ids:
        mask = df['tag-local-identifier'] == duck
        duck_df = df[mask].copy()
        duck_df['prev_lat'] = duck_df['location-lat'].shift(1)
        duck_df['prev_long'] = duck_df['location-long'].shift(1)
        duck_df['distance_km'] = duck_df.apply(
            lambda row: haversine_distance(row['location-lat'], row['location-long'],
                                           row['prev_lat'], row['prev_long'])
            if pd.notna(row['prev_lat']) else 0, axis=1
        )
        duck_df['cumulative_distance_km'] = duck_df['distance_km'].cumsum()
        duck_df['cumulative_distance_miles'] = duck_df['cumulative_distance_km'] * 0.621371
        df.loc[mask, 'cumulative_distance_miles'] = duck_df['cumulative_distance_miles']

    df_limited = df[df['cumulative_distance_miles'] <= miles_limit].copy()
    print(f"Rows after 50-mile filter: {len(df_limited)}")

    # Aggregate by day and duck
    df_limited = df_limited.groupby([pd.Grouper(key='timestamp', freq='D'), 'tag-local-identifier']).last().reset_index()
    print(f"Rows after daily aggregation: {len(df_limited)}")

    # Weather columns
    weather_columns = ['temperature', 'wind_speed', 'precipitation_probability']
    for col in weather_columns:
        df_limited[col] = np.nan
    
    # Fetches weather
    print(f"Starting weather fetch for {len(df_limited)} rows")
    for i, (index, row) in enumerate(df_limited.iterrows()):
        print(f"Processing row {index} at {row['timestamp']} for duck {row['tag-local-identifier']}")
        weather_raw = get_weather(row['location-lat'], row['location-long'])
        if weather_raw:
            weather_info = process_weather(weather_raw)
            for col, value in weather_info.items():
                if col in df_limited.columns:
                    df_limited.at[index, col] = value

    # Fills in missing weather data
    for col in weather_columns:
        if df_limited[col].isna().all():
            df_limited[col] = df_limited[col].fillna(0)
        else:
            df_limited[col] = df_limited[col].fillna(df_limited[col].mean())
    
    df_duck = df_limited.copy()
    print(f"Final rows for {num_ducks} ducks: {len(df_duck)}")
    
    df_duck['orig_lat'] = df_duck['location-lat']
    df_duck['orig_long'] = df_duck['location-long']
    
    # Scalers with variance check
    scalers_full = {}
    scalers_coords = {}
    features = ['location-lat', 'location-long', 'temperature', 'wind_speed', 'precipitation_probability']
    coord_features = ['location-lat', 'location-long']
    for duck in duck_ids:
        mask = df_duck['tag-local-identifier'] == duck
        duck_data = df_duck.loc[mask]
        print(f"Duck {duck} stats:")
        for feat in features:
            print(f"  {feat}: mean={duck_data[feat].mean():.2f}, std={duck_data[feat].std():.2f}, nan_count={duck_data[feat].isna().sum()}")
        
        scaler_full = StandardScaler()
        scaler_coords = StandardScaler()
        if duck_data[features].std().min() == 0:
            print(f"Warning: Duck {duck} has zero variance in some features. Replacing with unscaled data.")
            df_duck.loc[mask, features] = duck_data[features].fillna(0).values
        else:
            df_duck.loc[mask, features] = scaler_full.fit_transform(duck_data[features])
            scalers_full[duck] = scaler_full
        
        if duck_data[coord_features].std().min() == 0:
            print(f"Warning: Duck {duck} has zero variance in coordinates. Replacing with unscaled data.")
            df_duck.loc[mask, coord_features] = duck_data[coord_features].fillna(0).values
        else:
            df_duck.loc[mask, coord_features] = scaler_coords.fit_transform(duck_data[coord_features])
            scalers_coords[duck] = scaler_coords
    
    feature_values = df_duck[features].values
    inputs = feature_values[:-1]
    targets = df_duck[coord_features].values[1:]
    
    return inputs, targets, df_duck[['timestamp'] + features].values, scalers_full, scalers_coords, duck_ids, df_duck

# RNN Model with Dropout
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(RNNModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length + 1):
        seq = data[i:i + seq_length]
        if len(seq) == seq_length:
            sequences.append(seq)
    sequences = np.array(sequences)
    print(f"Created {len(sequences)} sequences of length {seq_length}")
    return sequences

def main():
    file_path = 'GNN_inital_test.csv'
    batch_size = 5
    seq_length = 5
    input_size = 5
    miles_limit = 50

    print("Starting data load")
    inputs, targets, coords, scalers_full, scalers_coords, duck_ids, df_duck = load_data(file_path, num_ducks=batch_size, miles_limit=miles_limit)
    print("Data load complete")

    input_sequences = create_sequences(inputs, seq_length)
    target_sequences = create_sequences(targets, seq_length)

    if len(input_sequences) == 0 or len(target_sequences) == 0:
        print("Error: No valid sequences created. Check data length and sequence length.")
        return

    input_tensor = torch.tensor(input_sequences, dtype=torch.float32)
    target_tensor = torch.tensor(target_sequences, dtype=torch.float32)[:, -1, :]

    # Initialize model
    hidden_size = 128
    num_layers = 2
    model = RNNModel(input_size, hidden_size, num_layers, dropout=0.2)

    # Training
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    num_epochs = 200
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(input_tensor)
        loss = criterion(outputs, target_tensor)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Prediction
    model.eval()
    all_predictions = []
    with torch.no_grad():
        last_positions = []
        last_positions_normalized = []
        print("Last known positions (original):")
        for duck in duck_ids:
            duck_data = df_duck[df_duck['tag-local-identifier'] == duck]
            if len(duck_data) == 0:
                last_pos = np.zeros(2)
                last_pos_normalized = np.zeros(input_size)
            else:
                last_row = duck_data.tail(1)
                last_pos = last_row[['orig_lat', 'orig_long']].values[0]
                last_pos_normalized = last_row[['location-lat', 'location-long', 'temperature', 'wind_speed', 'precipitation_probability']].values[0]
            last_positions.append(last_pos)
            last_positions_normalized.append(last_pos_normalized)
            timestamp = last_row['timestamp'].values[0]
            print(f"  Duck {duck}: Latitude = {last_pos[0]:.4f}, Longitude = {last_pos[1]:.4f}, Timestamp = {timestamp}")
        
        last_positions = np.array(last_positions)
        last_positions_normalized = np.array(last_positions_normalized)
        all_predictions.append(last_positions)
        
        current_sequence = torch.tensor(np.repeat(last_positions_normalized[np.newaxis, :, :], seq_length, axis=0).transpose(1, 0, 2), dtype=torch.float32)

        for day in range(3):  # Predict 3 days ahead
            outputs = model(current_sequence)
            pred_normalized = outputs.numpy()
            inverse_pred = np.zeros((batch_size, 2))
            for i, duck in enumerate(duck_ids):
                inverse_pred[i] = scalers_coords[duck].inverse_transform([pred_normalized[i]])[0]
            
            # No distance constraint—let it predict freely
            all_predictions.append(inverse_pred)
            pred_with_weather = np.concatenate([pred_normalized, last_positions_normalized[:, 2:]], axis=1)
            new_seq = torch.cat((current_sequence[1:], torch.tensor(pred_with_weather, dtype=torch.float32).unsqueeze(0)), dim=0)
            current_sequence = new_seq

    # Print predictions
    all_predictions = np.array(all_predictions)
    for day, pred in enumerate(all_predictions, 1):
        print(f"Day {day}:")
        for i, duck_id in enumerate(duck_ids):
            print(f"  Duck {duck_id}: Latitude = {pred[i, 0]:.4f}, Longitude = {pred[i, 1]:.4f}")
        print("=" * 40)

if __name__ == "__main__":
    main()