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
from math import sqrt
from sklearn.cluster import DBSCAN

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

#Class Declaration
class Duck():

    #Variable initializations
    def __init__(self, duckID):
        self.duckID = duckID
        self.species = ''
        self.longs = []
        self.lats = []
        self.coord = []
        self.timestamps = []
    
    def importLoc(self, df):

        #Saving duck ID's
        duck_data = df[df['tag-local-identifier'] == self.duckID]

        #Saving species
        self.species = duck_data['individual-taxon-canonical-name']

        #Saving timestamps
        self.timestamps = duck_data['timestamp'].tolist()

        #Saving longitudes
        self.long = duck_data['location-long'].tolist()

        #Saving latitudes
        self.lat = duck_data['location-lat'].tolist()

        #Combining coordinates
        self.coord = list(zip(self.long, self.lat))


#Function to print duck data for testing purposes
def print_duck_data(ducks_dict, label="Duck Data"):

    print(f"\n{label}:")
    print("-"*25)

    for duck_id, duck in ducks_dict.items():
        print(f"Duck ID: {duck_id}")
        print(f"  Timestamps: {duck.timestamps[:3]}{'...' if len(duck.timestamps) > 3 else ''}")
        
        #Print the first 3 coordinates
        print("  Coordinates (long, lat):")
        for i, (lng, lat) in enumerate(duck.coord[:3]):
            print(f"    {i+1}: ({lng}, {lat})")
        
        if len(duck.coord) > 3:
            print("    ...")
            
        #Print the total number of coordinates
        print(f"  Total locations: {len(duck.coord)}")
        print("-" * 30)


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
    sampleIDs = random.sample(duckList, portion)

    return sampleIDs


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


#Calculate Haversine distance between two geographic coordinates
def haversine_distance(coord1, coord2):

    #Converting decimal degrees to radians
    lon1, lat1 = coord1
    lon2, lat2 = coord2
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
    #Haversine formula implementation
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  #Radius of earth in kilometers
    
    return c * r


#Create proximity-based graph preserving all original locations
def create_proximity_graph(ducks, proximity_threshold=0.5, sequential_weight=5.0):
    """
    Overall process - creates a graph that preserves all original locations and adds both:
    1. Sequential edges based on duck movements (heavily weighted)
    2. Proximity edges between nearby nodes (lighter weights)
    Arg specifications: 
        proximity_threshold: Distance threshold in km to consider nodes as connected
        sequential_weight: Base weight for sequential edges (actual duck movements)
    """
    # Create graph
    G = nx.Graph()
    
    #Set all nodes while preserving all, full locations
    nodes = create_nodes(ducks)
    G.add_nodes_from(nodes)
    print(f"Added {len(nodes)} unique location nodes to graph")
    
    #Adding all sequential edges (actual duck movements)
    print("Adding sequential edges based on duck movements...")
    sequential_edges = []
    edge_count = {}
    duck_edge_map = {}
    
    for duck in ducks.values():
        for i in range(len(duck.coord) - 1):
            node1 = duck.coord[i]
            node2 = duck.coord[i + 1]
            
            #Skipping self-loops
            if node1 == node2:
                continue
                
            edge = (node1, node2)
            sequential_edges.append(edge)
            
            #Counting how many times each edge appears
            edge_count[edge] = edge_count.get(edge, 0) + 1
            
            #Storing which ducks used each edge
            if edge in duck_edge_map:
                duck_edge_map[edge].add(duck.duckID)
            else:
                duck_edge_map[edge] = {duck.duckID}
    
    #Adding sequential edges with weights based on frequency
    for edge, count in edge_count.items():
        weight = sequential_weight * count  #Note: Higher weight for frequently used paths
        G.add_edge(edge[0], edge[1], weight=weight, edge_type='sequential', 
                   count=count, ducks=list(duck_edge_map[edge]))
    
    print(f"Added {len(edge_count)} sequential edges")
    
    #Adding proximity edges between nearby nodes that aren't already connected
    print("Adding proximity edges between nearby locations...")
    proximity_edges_added = 0
    
    #Checking nodes that are likely to be close (to maintain efficiency)
    #Grouping nodes into geographic bins
    bin_size = proximity_threshold * 2  #Note: km in longitude/latitude
    node_bins = {}
    
    for node in nodes:
        #Creating a bin key based on rough geographic location
        bin_x = int(node[0] / bin_size)
        bin_y = int(node[1] / bin_size)
        bin_key = (bin_x, bin_y)
        
        if bin_key in node_bins:
            node_bins[bin_key].append(node)
        else:
            node_bins[bin_key] = [node]
    
    #Checking proximity only for nodes in the same or adjacent bins
    for bin_key, bin_nodes in node_bins.items():
        bin_x, bin_y = bin_key
        
        #Getting nodes from current and adjacent bins
        nearby_nodes = bin_nodes.copy()
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                adj_key = (bin_x + dx, bin_y + dy)
                if adj_key in node_bins and adj_key != bin_key:
                    nearby_nodes.extend(node_bins[adj_key])
        
        #Checking proximity for nodes in this extended set
        for i, node1 in enumerate(bin_nodes):
            for node2 in nearby_nodes[i+1:]:  #Note: Starting from i+1 to avoid duplicate checks
                #Skipping if already connected by sequential edge
                if G.has_edge(node1, node2):
                    continue
                
                #Calculating distance
                distance = haversine_distance(node1, node2)
                
                #Connecting if within threshold
                if distance <= proximity_threshold:
                    #Weighting inversely proportional to distance (closer = stronger connection)
                    #Note: always less than sequential edges
                    weight = (1.0 - (distance / proximity_threshold)) * (sequential_weight * 0.5)
                    G.add_edge(node1, node2, weight=weight, edge_type='proximity', 
                              distance=distance)
                    proximity_edges_added += 1
    
    print(f"Added {proximity_edges_added} proximity edges")
    print(f"Total graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    return G, duck_edge_map


#Function to predict next location of duck
def predict_next_location(G, current_location, prefer_sequential=True):
    # Finding all neighboring nodes
    neighbors = list(G.neighbors(current_location))
    
    #Checking error case if node is completely isolated
    if not neighbors:
        print("No neighboring locations to predict next stopover.")
        return None

    #Variable initializations for comparison below
    max_weight = -1
    next_location = None

    #Cycling through all neighboring nodes to find strongest potential edge
    for neighbor in neighbors:
        weight = G[current_location][neighbor].get('weight', 1)
        
        #If prefer_sequential is True, prioritize sequential edges
        if prefer_sequential:
            edge_type = G[current_location][neighbor].get('edge_type', '')
            if edge_type == 'sequential':
                # Give sequential edges higher preference
                weight = weight * 2  # or some other boosting factor
                
        if weight > max_weight:
            max_weight = weight
            next_location = neighbor

    #Returning edge with greatest likelihood
    return next_location


def get_weather(lat, lon):

    url = f"https://api.weather.gov/points/{lat},{lon}"

    headers = {
        "User-Agent": "WaterfowlMigration",
        "Accept": "application/json"
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:

        print("URL success")
        data = response.json()
        forecast_url = data['properties']['forecast']

        forecast_response = requests.get(forecast_url, headers=headers)

        if forecast_response.status_code == 200:
            print("Forecast success")
            return forecast_response.json()
        else:
            print("Forecast fail")
            return None

    else:
        print("URL fail")
        return None

def process_weather(weather_raw):

    periods = weather_raw['properties']['periods']

    relevant_data = {
        "timestamp": [period['startTime'] for period in periods],
        "temperature": [period['temperature'] for period in periods],
        "temperature_unit": [period['temperatureUnit'] for period in periods],
        "wind_speed": [period['windSpeed'] for period in periods],
        "wind_direction": [period['windDirection'] for period in periods],
        "precipitation": [period.get('probabilityOfPrecipitation', {}).get('value', None) for period in periods],
        "short_forecast": [period['shortForecast'] for period in periods]
    }

    weather_filtered = pd.DataFrame(relevant_data)
    return weather_filtered


#Visualizing the duck migration network
def visualize_migration_network(G, ducks, output_file="migration_network.png", highlight_duck_id=None):
    """
    Visualize the duck migration network with nodes colored by frequency
    and edges colored by type (sequential vs proximity)
    
    Args:
        G: NetworkX graph of the migration network
        ducks: Dictionary of Duck objects
        output_file: Where to save the visualization
        highlight_duck_id: Optional ID of a specific duck to highlight
    """
    plt.figure(figsize=(12, 10))
    
    #Calculating node sizes based on how many ducks visited each location
    node_visits = {node: 0 for node in G.nodes()}
    for duck in ducks.values():
        for coord in duck.coord:
            if coord in node_visits:
                node_visits[coord] += 1
    
    #Node colors based on frequency (heat map)
    node_colors = [np.log1p(node_visits[node]) for node in G.nodes()]
    
    #Edge colors based on type (sequential vs proximity)
    edge_colors = []
    for u, v, data in G.edges(data=True):
        if data.get('edge_type') == 'sequential':
            edge_colors.append('blue')
        else:
            edge_colors.append('gray')
    
    #Creating position map based on geographic coordinates
    pos = {node: (node[0], node[1]) for node in G.nodes()}
    
    #Drawing the network
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=[max(20, 5*visits) for visits in node_visits.values()],
                          cmap=plt.cm.YlOrRd, alpha=0.7)
    
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=0.5, alpha=0.6)
    
    #If highlighting a specific duck, draw its path
    if highlight_duck_id and highlight_duck_id in ducks:
        duck = ducks[highlight_duck_id]
        duck_path = []
        for i in range(len(duck.coord) - 1):
            duck_path.append((duck.coord[i], duck.coord[i+1]))
        
        nx.draw_networkx_edges(G, pos, edgelist=duck_path, 
                              edge_color='red', width=2.0)
    
    plt.title("Duck Migration Network")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Network visualization saved to {output_file}")


import json

def save_predictions_as_json(ducks, G, output_file="duck_migration_forecast.json"):
    """
    Predict next location for each duck and save the result in JSON format.
    Output format: 
    [
        {
            "duck_id": "123456",
            "base_timestamp": "2024-09-09 04:19:05",
            "forecast_timestamp": "2024-09-10 04:19:05",
            "start_lat": 34.1234,
            "start_lon": -89.5678,
            "forecast_lat": 35.4321,
            "forecast_lon": -88.1234
        },
        ...
    ]
    """
    results = []
    forecast_timestamp = (pd.Timestamp.now() + pd.Timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")

    for duck_id, duck in ducks.items():
        if not duck.coord:
            continue

        start_location = duck.coord[-1]
        base_timestamp = pd.to_datetime(duck.timestamps[-1]).strftime("%Y-%m-%d %H:%M:%S")
        next_location = predict_next_location(G, start_location)

        if next_location:
            entry = {
                "duck_id": duck_id,
                "base_timestamp": base_timestamp,
                "forecast_timestamp": forecast_timestamp,
                "start_lat": start_location[1],
                "start_lon": start_location[0],
                "forecast_lat": next_location[1],
                "forecast_lon": next_location[0]
            }
            results.append(entry)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"✅ Saved forecast results to {output_file}")


if __name__ == "__main__":

    #Reading in data set
    df = pd.read_csv("ShortTermSetData(Aug-Sept).csv")

    #Determining total number of ducks in sample
    total, uniqueIDs = countDucks(df)

    #Creating scalable, random sample of ducks
    sampleDucks = selectDucks(total, uniqueIDs)
    print(f"Selected duck IDs: {sampleDucks}")

    #Declaration for duck storage
    ducks = {}

    #Creating profiles for each duck in sample
    for duck_id in sampleDucks:
        duck = Duck(duck_id)
        duck.importLoc(df)
        ducks[duck_id] = duck

    #Printing original duck data (for testing purposes)
    print_duck_data(ducks, "Original Duck Data")

    #Creating a proximity-based graph with preserved locations instead of rounding
    print("\nCreating proximity-based migration network...")
    G, duck_edge_map = create_proximity_graph(ducks, proximity_threshold=0.5, sequential_weight=5.0)
    
    #Print some statistics about the graph
    print("\nGraph Statistics:")
    print(f"Total unique nodes (locations): {G.number_of_nodes()}")
    print(f"Total edges: {G.number_of_edges()}")
    
    sequential_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('edge_type') == 'sequential']
    proximity_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('edge_type') == 'proximity']
    print(f"Sequential edges (actual migrations): {len(sequential_edges)}")
    print(f"Proximity edges (nearby locations): {len(proximity_edges)}")
    
    #Visualize the network
    visualize_migration_network(G, ducks)
    
    #Test prediction on sample duck
    test_duck = ducks[sampleDucks[0]]
    current_location = test_duck.coord[-1]
    print("\nPrediction Test:")
    print(f"Duck ID: {test_duck.duckID}")
    print(f"Current Location: {current_location}")
    
    next_location_seq = predict_next_location(G, current_location)
    print(f"Predicted next location (sequential priority): {next_location_seq}")
    
    next_location_all = predict_next_location(G, current_location)
    print(f"Predicted next location (all edges): {next_location_all}")
    
    #Optional: Generate visualization highlighting this specific duck's path
    visualize_migration_network(G, ducks, output_file=f"duck_{test_duck.duckID}_path.png", 
                              highlight_duck_id=test_duck.duckID)


    #roundedDucks = loc_round(ducks, roundPoint=3)
    #print_duck_data(roundedDucks, "Rounded Duck Data")
    #compare_reduction(ducks, roundedDucks)

    #Graph initialization (empty)
    #G = nx.Graph()

    #Getting and setting nodes to the map
    #nodes = create_nodes(roundedDucks)
    #G.add_nodes_from(nodes)

    #Creating raw edges
    #edges,duck_edge_map, edge_count = create_edges(roundedDucks)
    
    #Adjustable weight for each covariance
    factor_weights = {
        'barometric_pressure': 0.3,
        'wind_speed': 0.2,
        'snow': 0.2,
        'temperature': 0.3
    }  
    
    save_predictions_as_json(ducks, G)

    #Adding weights to graph
    #add_edge_weights(G, edges) #edge_count, df, roundedDucks, factor_weights
    #print("Added edge weights")

    #Prediction testing, validation still required
    #test_duck = roundedDucks[sampleDucks[0]]
    #current_location = test_duck.coord[-1]
    #print("Current Location: ", test_duck.duckID, " , ", current_location)
    #next_location = predict_next_location(G, current_location)
    #print("Predicted next: ", next_location)
