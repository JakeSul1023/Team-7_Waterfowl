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

#Stored API Token
NOAA_TOKEN = "pKsYMZQINiCtWEbiUdvJHImQDqYlZmhu"

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

        #Saving 
        self.species = duck_data['individual-taxon-canonical-name']

        #Saving timestamps
        self.timestamps = duck_data['timestamp'].tolist()

        #Saving longitudes
        self.long = duck_data['location-long'].tolist()

        #Saving latitudes
        self.lat = duck_data['location-lat'].tolist()

        #Combining coordinates
        self.coord = list(zip(self.long, self.lat))


#Function to print duck data fro testing purposes
def print_duck_data(ducks_dict, label="Duck Data"):
    print(f"\n{label}:")
    print("-"*25)

    for duck_id, duck in ducks_dict.items():
        print(f"Duck ID: {duck_id}")
        print(f"  Timestamps: {duck.timestamps[:3]}{'...' if len(duck.timestamps) > 3 else ''}")
        
        # Print the first 3 coordinates
        print("  Coordinates (long, lat):")
        for i, (lng, lat) in enumerate(duck.coord[:3]):
            print(f"    {i+1}: ({lng}, {lat})")
        
        if len(duck.coord) > 3:
            print("    ...")
            
        # Print the total number of coordinates
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


#Function to round off latitude/longitutde coordinates
def loc_round(rawDucks, roundPoint):
    
    roundedDucks = {}

    #Processing each duck
    for duck_id, duck in rawDucks.items():

        #Creating a new duck object based off the Duck class
        rounded_duck = Duck(duck_id)
        
        #Copying species
        rounded_duck.species = duck.species

        #Copying timestamps
        rounded_duck.timestamps = duck.timestamps.copy() if hasattr(duck, 'timestamps') else []
        
        #Rounding coordinates
        rounded_long = [round(lng, roundPoint) for lng in duck.long]
        rounded_lat = [round(lat, roundPoint) for lat in duck.lat]

        #Updating duck with rounded coordinates
        rounded_duck.long = rounded_long
        rounded_duck.lat = rounded_lat
        rounded_duck.coord = list(zip(rounded_long, rounded_lat))
        roundedDucks[duck_id] = rounded_duck
    
        return roundedDucks


#Function to determine final weight of edges between each connections
def calculate_edge_weight(node1, node2, df, duck_id): #available_factors, factors_weight=None
    
    #if factors_weight is None:
    #    factors_weight = {factor: 1.0/len(available_factors) for factor in available_factors}
    
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
    #for factor in available_factors:
        #try:
        #    factor_values.append(
        #        normalize_factor(
        #            node1_data[factor].values[0], 
        #            node2_data[factor].values[0]
        #        )
        #    )
        #except KeyError:
        #    print(f"Warning: Factor {factor} not found in the DataFrame.")
        #    continue
    
    #Returning standard weight value if none can be found/applied
    if not factor_values:
        return 0.5
    
    #Adding all weights together for given edge
    #edge_weight = sum(
    #    factors_weight.get(available_factors[i], 0) * value 
    #    for i, value in enumerate(factor_values)
    #)
    
    #Min function used to ensure weight is no greater than 1
    return max(0, 0.5) #min(1, edge_weight)


#Function to apply calculated/finalized weights to resepective edges
def add_edge_weights(G, edges):
 
     edge_count = {}
 
     for edge in edges:
         if edge not in edge_count:
             edge_count[edge] = 1
         else:
             edge_count[edge] += 1
 
     for edge, count in edge_count.items():
         if G.has_edge(*edge):
             G[edge[0]][edge[1]]['weight'] += count  
         else:
             G.add_edge(edge[0], edge[1], weight=count)

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
#Not necessary for current model as visualization is optional, will be removed
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

    #Adding edges between nodes (need to remove unneccesary color functionality)
    for edge in edges:
        duck_ids = duck_edge_map.get(edge, set())  
        if duck_ids:
            colors = [duck_colors[duck_id] for duck_id in duck_ids if duck_id in duck_colors]
            avg_color = np.mean(colors, axis=0) if colors else "black"  
        else:
            avg_color = "black"

        nx.draw_networkx_edges(G, pos, edgelist=[edge], edge_color=[avg_color], width=2)

    #Printing graph to screen, not necessary for backend modeling
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

def compare_reduction(originalDucks, roundedDucks):

    original_nodes = set()
    for duck in originalDucks.values():
        original_nodes.update(duck.coord)

    rounded_nodes = set()
    for duck in roundedDucks.values():
        rounded_nodes.update(duck.coord)

    original_count = len(original_nodes)
    rounded_count = len(rounded_nodes)

    if original_count > 0:
        percent_reduction = ((original_count - rounded_count) / original_count) * 100
    else:
        percent_reduction = 0

    print(f"Origianl Nodes: {original_count}")
    print(f"Rounded Nodes: {rounded_count}")
    print(f"Reduction: {percent_reduction:.2f}%")
    
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

    roundedDucks = loc_round(ducks, roundPoint=3)
    print_duck_data(roundedDucks, "Rounded Duck Data")
    compare_reduction(ducks, roundedDucks)

    #Graph initialization (empty)
    G = nx.Graph()

    #Getting and setting nodes to the map
    nodes = create_nodes(roundedDucks)
    G.add_nodes_from(nodes)

    #Creating raw edges
    edges,duck_edge_map, edge_count = create_edges(roundedDucks)
    
    #Adjustable weight for each covariance
    factor_weights = {
        'barometric_pressure': 0.3,
        'wind_speed': 0.2,
        'snow': 0.2,
        'temperature': 0.3
    }  
    
    #Adding weights to graph
    add_edge_weights(G, edges) #edge_count, df, roundedDucks, factor_weights
    print("Added edge weights")
    #Assigning each duck a unique color (remove in Iteration 5)
    duck_colors = generate_duck_colors(roundedDucks)

    #graph_ducks(G, edges, duck_edge_map, duck_colors)

    #Prediction testing, validation still required
    test_duck = roundedDucks[sampleDucks[0]]
    current_location = test_duck.coord[-1]
    print("Current Location: ", test_duck.duckID, " , ", current_location)
    next_location = predict_next_location(G, current_location)
    print("Predicted next: ", next_location)

'''
    print(" ")
    print("*"*10)
    print("Weather Testing")
    #Test points
    lat = 40.723487854003906
    lon = -91.1420669555664
    weather_r = get_weather(lat, lon)
    weather_f = process_weather(weather_r)
    print(f"Weather data: {weather_f}")   
'''
