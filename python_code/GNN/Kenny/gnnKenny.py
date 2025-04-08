#Author: Revel Etheridge
#Date: 03-30-2025
#Model 4: Graph Neural Network (GNN) for Network-Based Migration Prediction 
#Goal: Model migration as a graph problem using historical stopover sites. 
#Why? Ducks tend to follow structured migration paths, which can be modeled as a graph of stopovers rather than just sequential time-series data. 

#Approach: 
#Nodes = Historical stopover locations. 
#Edges = Migration connections between locations (weighted by frequency). 
#GNN predicts the most probable next node (stopover location). 
#Use Case: Useful for network-based decision-making, such as conservation planning. 
#Assigned Team Members: Revel Etheridge 

#Library declarations
import pandas as pd
import matplotlib.pyplot as plt
import random
import networkx as nx
import numpy as np
import requests
import time

#Stored API Token
OPENWEATHER_API_KEY = "02de0c63b48bcd13d425a73caa22eb81"

# OpenWeather API integration functions
def get_openweather_forecast(lat, lon, api_key=OPENWEATHER_API_KEY):
    """
    Fetch weather data from OpenWeather API for a given latitude and longitude.
    Parameters:
    - lat: Latitude (rounded to 3 decimal places)
    - lon: Longitude (rounded to 3 decimal places)
    - api_key: OpenWeather API key
    Returns:
    - Dictionary containing weather data.
    """
    lat = round(lat, 3)
    lon = round(lon, 3)
    
    url = f"https://api.openweathermap.org/data/2.5/weather"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": api_key,
        "units": "metric"  # Use "imperial" for Fahrenheit
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        weather_info = {
            "latitude": lat,
            "longitude": lon,
            "temperature": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "pressure": data["main"]["pressure"],
            "weather_description": data["weather"][0]["description"],
            "wind_speed": data["wind"]["speed"],
            "timestamp": pd.Timestamp.now()  # Adding timestamp for when data was collected
        }
        print(f"✅ Retrieved weather for ({lat}, {lon}): {weather_info['weather_description']}, {weather_info['temperature']}°C")
        return weather_info
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to retrieve weather for ({lat}, {lon}): {e}")
        return None

def fetch_openweather_forecast_for_points(lat_lon_list, api_key=OPENWEATHER_API_KEY):
    """
    Fetch weather forecasts for multiple latitude/longitude points.
    Parameters:
    - lat_lon_list: List of (latitude, longitude) tuples
    - api_key: OpenWeather API key
    Returns:
    - List of weather data dictionaries.
    """
    weather_data = []
    for lat, lon in lat_lon_list:
        weather_info = get_openweather_forecast(lat, lon, api_key)
        if weather_info:
            weather_data.append(weather_info)
        time.sleep(1.2)  # Adding a small delay to respect API rate limits
    return weather_data

def is_valid_openweather_point(lat, lon, api_key=OPENWEATHER_API_KEY):
    """
    Check if OpenWeather API can return valid data for a given lat/lon.
    Parameters:
    - lat: Latitude
    - lon: Longitude
    - api_key: OpenWeather API key
    Returns:
    - True if a valid forecast is found, False otherwise.
    """
    lat = round(lat, 3)
    lon = round(lon, 3)
    url = f"https://api.openweathermap.org/data/2.5/weather"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": api_key
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return True
    except requests.exceptions.HTTPError:
        return False
    except Exception as e:
        print(f"Error checking OpenWeather point ({lat}, {lon}): {e}")
        return False

def process_weather_data(weather_data_list):
    """
    Process a list of weather data dictionaries into a pandas DataFrame.
    Parameters:
    - weather_data_list: List of dictionaries containing weather data
    Returns:
    - Pandas DataFrame with processed weather data
    """
    if not weather_data_list:
        print("No weather data to process")
        return pd.DataFrame()
    
    # Convert list of dictionaries to DataFrame
    weather_df = pd.DataFrame(weather_data_list)
    return weather_df

#Class Declaration with enhanced weather integration
class Duck():
    """
    Duck class to store location data and associated weather information.
    """
    #Variable initializations
    def __init__(self, duckID):
        self.duckID = duckID
        self.long = []
        self.lat = []
        self.coord = []
        self.timestamps = []
        self.weather_data = []  # Store weather data for each location
    
    def importLoc(self, df):
        """
        Import location data from DataFrame.
        """
        #Saving duck ID's
        duck_data = df[df['tag-local-identifier'] == self.duckID]
       
        #Saving longitudes
        self.long = duck_data['location-long'].tolist()

        #Saving latitudes
        self.lat = duck_data['location-lat'].tolist()

        #Combining coordinates
        self.coord = list(zip(self.long, self.lat))
        
        # If timestamps are available in the data
        if 'timestamp' in duck_data.columns:
            self.timestamps = duck_data['timestamp'].tolist()
    
    def fetch_weather_data(self, limit=None, bounds=None):
        """
        Fetch weather data for this duck's coordinates.
        
        Parameters:
            limit (int, optional): Maximum number of coordinates to fetch weather for.
            bounds (tuple, optional): Bounding box (min_lat, max_lat, min_lon, max_lon) to filter locations.
        """
        if not hasattr(self, 'coord'):
            raise AttributeError("Duck object is missing 'coord'. Did you run importLoc()?")

        if not hasattr(self, 'duckID'):
            raise AttributeError("Duck object is missing 'duckID'.")

        # Default to continental U.S. bounds if none provided
        if bounds is None:
            bounds = (24.396308, 49.384358, -125.0, -66.93457)  # (min_lat, max_lat, min_lon, max_lon)

        min_lat, max_lat, min_lon, max_lon = bounds

        weather_cache = {}
        results = []

        coords = self.coord if limit is None else self.coord[:limit]

        for idx, (lon, lat) in enumerate(coords):
            # Filter out-of-bounds coordinates
            if not (min_lat <= lat <= max_lat and min_lon <= lon <= max_lon):
                print(f"Skipping out-of-bounds location for Duck {self.duckID}: ({lat}, {lon})")
                continue

            key = (round(lat, 4), round(lon, 4))  # rounding to avoid float precision issues

            if key in weather_cache:
                weather = weather_cache[key]
            else:
                weather = get_openweather_forecast(lat, lon)
                if weather:
                    weather_cache[key] = weather

            if weather:
                weather_entry = {
                    'duck_id': self.duckID,
                    'index': idx,
                    'latitude': lat,
                    'longitude': lon,
                    'temperature': weather.get('temperature'),
                    'pressure': weather.get('pressure'),
                    'wind_speed': weather.get('wind_speed'),
                    'humidity': weather.get('humidity'),
                    'snow': weather.get('snow'),
                    'conditions': weather.get('conditions')
                }
                results.append(weather_entry)

        self.weather_data = results
        return results


    def get_weather_dataframe(self):
        """
        Convert the duck's weather data to a pandas DataFrame.
        Returns:
        - Pandas DataFrame with all weather data for this duck
        """
        if not self.weather_data:
            print(f"No weather data available for Duck {self.duckID}. Call fetch_weather_data() first.")
            return pd.DataFrame()
        
        return pd.DataFrame(self.weather_data)


#Function to count the number of unique ducks present in the data set
def countDucks(df):
    #Pulling entire column of duck id's (including repeats)
    pooledIDs = df['tag-local-identifier'].tolist()
    
    #Placeholder for list of unique duck IDs
    uniqueIDs = list(set(pooledIDs))

    return len(uniqueIDs), uniqueIDs


#Function to allow developer to choose amount of ducks to choose from
def selectDucks(totalDucks, duckList):
    #Asking user for number of desired modeling units
    print("Total Number of unique duck IDs imported: ", totalDucks)
    portion = int(input("How many ducks would you like to model with? "))
    sampleList = random.sample(duckList, portion)

    return sampleList


#Function to create a node for each duck present in the data set
def create_nodes(ducks):
    nodes = set()  
    for duck in ducks.values():
        nodes.update(duck.coord) 
    return list(nodes)


#Function to normalize factor weights (edges can only hold values betwene 0 and 1)
def normalize_factor(val1, val2, threshhold=0.2):
    diff = abs(val1 - val2)
    
    if diff <= threshhold:
        return 1.0
    
    weight = np.exp(-diff)
    return max(0, min(1, weight))


#Function to determine final weight of edges between each connections
def calculate_edge_weight(node1, node2, df, duck_id, available_factors, factors_weight=None):
    if factors_weight is None:
        factors_weight = {factor: 1.0/len(available_factors) for factor in available_factors}
    
    duck_data = df[df['tag-local-identifier'] == duck_id]
    
    node1_data = duck_data[
        (duck_data['location-long'] == node1[0]) & 
        (duck_data['location-lat'] == node1[1])
    ]
    node2_data = duck_data[
        (duck_data['location-long'] == node2[0]) & 
        (duck_data['location-lat'] == node2[1])
    ]
    
    #Returning a middle of the road weight if no values can be applied
    if node1_data.empty or node2_data.empty:
        return 0.5  
    
    #Normalize all factor values between 0 and 1
    factor_values = []
    for factor in available_factors:
        try:
            factor_values.append(
                normalize_factor(
                    node1_data[factor].values[0], 
                    node2_data[factor].values[0]
                )
            )
        except KeyError:
            #print(f"Warning: Factor {factor} not found in the DataFrame.")
            continue
    
    #Returning standard weight value if none can be found/applied
    if not factor_values:
        return 0.5
    
    #Adding all weights together for given edge
    edge_weight = sum(
        factors_weight.get(available_factors[i], 0) * value 
        for i, value in enumerate(factor_values)
    )
    
    #Min function used to ensure weight is no greater than 1
    return max(0, min(1, edge_weight))


#Function to apply calculated/finalized weights to resepective edges
def add_edge_weights(G, edge_count, df, ducks, available_factors, factors_weight=None):
    #Cycling through all available edges
    for edge, count in edge_count.items():
        edge_ducks = [duck for duck in ducks.values() if edge in list(zip(duck.coord[:-1], duck.coord[1:]))]
        
        #For all valid edges, calculating their weights
        if edge_ducks:
            weights = [
                calculate_edge_weight(
                    edge[0], edge[1], df, duck.duckID, available_factors, factors_weight
                ) for duck in edge_ducks
            ]
            #Determining edge's rough average weight
            avg_weight = np.mean(weights)
            
            #Adding completed edge to graph
            G.add_edge(edge[0], edge[1], weight=avg_weight)


#Function to predict next location of duck
def predict_next_location(G, current_location):
    #Finding all neighboring nodes
    neighbors = list(G.neighbors(current_location))
    
    #Error case if node is completely isolated
    if not neighbors:
        print("No neighboring locations to predict next stopover.")
        return None

    #Variable initializations for comparison below
    max_weight = -1
    next_location = None

    #Cycling through all neighboring nodes to find strongest potential edge
    for neighbor in neighbors:
        weight = G[current_location][neighbor].get('weight', 1) 
        if weight > max_weight:
            max_weight = weight
            next_location = neighbor

    #Returning edge with greatest likelyhood
    return next_location


#Function to create unique color for each duck
def generate_duck_colors(ducks):
    colors = plt.cm.rainbow(np.linspace(0, 1, len(ducks)))
    return {duck_id: colors[i] for i, duck_id in enumerate(sorted(ducks))}


#Function to add all ducks to graph
def graph_ducks(G, edges, duck_edge_map, duck_colors):
    #Figure initialization (empty)
    plt.figure(figsize=(10, 7))

    #Layout declaration, can adjust to test efficiency (some work better based off whether visualization is used or not)
    pos = nx.spring_layout(G, seed = 42, k = 0.5) 

    #Swapping format for 'big data' (can be expanded upon)
    if(len(G.nodes) > 10000):
        pos = nx.random_layout(G)
    
    #Adding nodes to graph
    nx.draw_networkx_nodes(G, pos, node_size=100, node_color='skyblue')

    #Adding edges between nodes
    for edge in edges:
        duck_ids = duck_edge_map.get(edge, set())  
        if duck_ids:
            colors = [duck_colors[duck_id] for duck_id in duck_ids if duck_id in duck_colors]
            avg_color = np.mean(colors, axis=0) if colors else "black"  
        else:
            avg_color = "black"

        nx.draw_networkx_edges(G, pos, edgelist=[edge], edge_color=[avg_color], width=2)

    #Printing graph to screen
    plt.title("Duck Migration Network")
    plt.show()


#Function to create raw edges between nodes (locations)
def create_edges(ducks):
    #Variable declarations
    edges = []
    edge_count = {}
    duck_edge_map = {}

    #Finding adjacent nodes by cycling through available nodes
    for duck in ducks.values():
        for i in range(len(duck.coord) - 1):
            #Finding duck's next location
            node1 = duck.coord[i]
            node2 = duck.coord[i + 1]

            #ensuring node isn't a repeated(erroneous) value
            if node1 != node2: 
                edge = (node1, node2)

                #Updating edge count
                edge_count[edge] = edge_count.get(edge, 0) + 1

                #Store multiple duck IDs for shared edges
                if edge in duck_edge_map:
                    duck_edge_map[edge].add(duck.duckID)
                else:
                    duck_edge_map[edge] = {duck.duckID}

    return list(edge_count.keys()), duck_edge_map, edge_count


if __name__ == "__main__":
    #Reading in data set
    df = pd.read_csv("ShortTermSetData(Aug-Sept).csv")

    #Determining total number of ducks in sample
    total, uniqueIDs = countDucks(df)

    #Creating scalable, random sample of ducks
    sampleDucks = selectDucks(total, uniqueIDs)

    #Declaration for duck storage
    ducks = {}

    #Creating profiles for each duck in sample
    for duck_id in sampleDucks:
        duck = Duck(duck_id)
        duck.importLoc(df)
        ducks[duck_id] = duck
    
    print("Fetching weather data for sample ducks...")
    # Fetch weather for each duck's locations (limiting to most recent 3 locations to avoid API rate limits)
    all_weather_data = []
    for duck_id, duck in ducks.items():
        print(f"Processing Duck {duck_id}...")
        duck_weather = duck.fetch_weather_data() #use this to limit amount of times the location is used
        all_weather_data.extend(duck_weather)
        # Create a DataFrame of the duck's weather data
        weather_df = duck.get_weather_dataframe()
        print(f"Weather data for Duck {duck_id}:")
        if not weather_df.empty:
            print(weather_df[['latitude', 'longitude', 'temperature', 'pressure', 'wind_speed']]) #add heading if necessary
        print()
    
    # Process all collected weather data into a single DataFrame
    combined_weather_df = process_weather_data(all_weather_data)
    if not combined_weather_df.empty:
        print("\nCombined Weather Data Summary:")
        print(f"Total records: {len(combined_weather_df)}")
        print(combined_weather_df.groupby('duck_id').size().reset_index(name='location_count'))
    
    #Graph initialization (empty)
    G = nx.Graph()

    #Getting and setting nodes to the map
    nodes = create_nodes(ducks)
    G.add_nodes_from(nodes)

    #Creating raw edges
    edges, duck_edge_map, edge_count = create_edges(ducks)
    
    #Adjustable weight for each covariance
    factor_weights = {
        'pressure': 0.3,
        'wind_speed': 0.2,
        'humidity': 0.2,
        'temperature': 0.3
    }
    
    #Adding weights to graph
    add_edge_weights(G, edge_count, df, ducks, factor_weights)

    #Assigning each duck a unique color
    duck_colors = generate_duck_colors(ducks)

    #Optional: Visualize graph
    #graph_ducks(G, edges, duck_edge_map, duck_colors)

    #Prediction testing
    print("\nPrediction Testing:")
    for i, duck_id in enumerate(list(ducks.keys())[:3]):  # Test the first 3 ducks
        test_duck = ducks[duck_id]
        current_location = test_duck.coord[-1]
        print(f"Duck {duck_id} Current Location: {current_location}")
        next_location = predict_next_location(G, current_location)
        print(f"Duck {duck_id} Predicted Next Location: {next_location}")
        
        if next_location:
            # Get weather at predicted next location
            next_weather = get_openweather_forecast(next_location[1], next_location[0])
            if next_weather:
                print(f"Predicted location weather: {next_weather['temperature']}°C, {next_weather['weather_description']}")
        print()