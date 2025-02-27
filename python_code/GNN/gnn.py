#Author: Revel Etheridge
#Date: 02-25-2025
#Model 4: Graph Neural Network (GNN) for Network-Based Migration Prediction 
#Goal: Model migration as a graph problem using historical stopover sites. 
#Why? Ducks tend to follow structured migration paths, which can be modeled as a graph of stopovers rather than just sequential time-series data. 

#Approach: 
#Nodes = Historical stopover locations. 
#Edges = Migration connections between locations (weighted by frequency). 
#GNN predicts the most probable next node (stopover location). 
#Use Case: Useful for network-based decision-making, such as conservation planning. 
#Assigned Team Members: Revel Etheridge
#Build the stopover graph from historical data. 

import pandas as pd
import matplotlib.pyplot as plt
import random
import networkx as nx
import numpy as np

#Class Declaration
class Duck():

    def __init__(self, duckID):
        self.duckID = duckID
        self.longs = []
        self.lats = []
        self.coord = []
        self.timestamps = []
    
    def importLoc(self, df):

        duck_data = df[df['tag-local-identifier'] == self.duckID]
       
        #Saving longitudes
        self.long = duck_data['location-long'].tolist()

        #Saving latitudes
        self.lat = duck_data['location-lat'].tolist()

        #Combining coordinates
        self.coord = list(zip(self.long, self.lat))

def countDucks(df):

    #Pulling entire column of duck id's (including repeats)
    pooledIDs = df['tag-local-identifier'].tolist()
    
    #Placeholder for list of unique duck IDs
    uniqueIDs = list(set(pooledIDs))

    return len(uniqueIDs), uniqueIDs

def selectDucks(totalDucks, duckList):

    print("Total Number of unique duck IDs imported: ", totalDucks)
    portion = int(input("How many ducks would you like to model with? "))
    sampleList = random.sample(duckList, portion)

    return sampleList

def create_nodes(ducks):
    nodes = set()  
    for duck in ducks.values():
        nodes.update(duck.coord) 
    return list(nodes)  

def add_edge_weights(G, edge_count):
    for edge, count in edge_count.items():
        G.add_edge(edge[0], edge[1], weight=count)

def predict_next_location(G, current_location):

    neighbors = list(G.neighbors(current_location))
    
    if not neighbors:
        print("No neighboring locations to predict next stopover.")
        return None

    max_weight = -1
    next_location = None

    for neighbor in neighbors:
        weight = G[current_location][neighbor].get('weight', 1) 
        if weight > max_weight:
            max_weight = weight
            next_location = neighbor

    return next_location

def generate_duck_colors(ducks):
    colors = plt.cm.rainbow(np.linspace(0, 1, len(ducks)))
    
    return {duck_id: colors[i] for i, duck_id in enumerate(sorted(ducks))}

def graph_ducks(G, edges, duck_edge_map, duck_colors):
    
    plt.figure(figsize=(10, 7))

    pos = nx.spring_layout(G, seed = 42, k = 0.5) 

    if(len(G.nodes) > 10000):
        pos = nx.random_layout(G)
    
    nx.draw_networkx_nodes(G, pos, node_size=100, node_color='skyblue')

    for edge in edges:
        duck_ids = duck_edge_map.get(edge, set())  
        if duck_ids:
            colors = [duck_colors[duck_id] for duck_id in duck_ids if duck_id in duck_colors]
            avg_color = np.mean(colors, axis=0) if colors else "black"  
        else:
            avg_color = "black"

        nx.draw_networkx_edges(G, pos, edgelist=[edge], edge_color=[avg_color], width=2)

    plt.title("Duck Migration Network")
    plt.show()

def create_edges(ducks):
    edges = []
    edge_count = {}
    duck_edge_map = {}

    for duck in ducks.values():
        for i in range(len(duck.coord) - 1):
            node1 = duck.coord[i]
            node2 = duck.coord[i + 1]

            if node1 != node2: 
                edge = (node1, node2)

                edge_count[edge] = edge_count.get(edge, 0) + 1

                # Store multiple duck IDs for shared edges
                if edge in duck_edge_map:
                    duck_edge_map[edge].add(duck.duckID)
                else:
                    duck_edge_map[edge] = {duck.duckID}

    return list(edge_count.keys()), duck_edge_map, edge_count

if __name__ == "__main__":

    with open('ShortTermSetData(Aug-Sept).csv', mode='r')as file:
        df = pd.read_csv("ShortTermSetData(Aug-Sept).csv")

    #Determining total number of ducks in sample
    total, uniqueIDs = countDucks(df)

    #Creating scalable, random sample of ducks
    sampleDucks = selectDucks(total, uniqueIDs)

    ducks = {}

    #Creating profiles for each duck in sample
    for duck_id in sampleDucks:
        duck = Duck(duck_id)
        duck.importLoc(df)
        ducks[duck_id] = duck

    #Graph initialization (empty)
    G = nx.Graph()

    nodes = create_nodes(ducks)
    G.add_nodes_from(nodes)

    edges,duck_edge_map, edge_count = create_edges(ducks)
    G.add_edges_from(edges)
    add_edge_weights(G, edge_count)

    duck_colors = generate_duck_colors(ducks)

    graph_ducks(G, edges, duck_edge_map, duck_colors)

    test_duck = ducks[sampleDucks[0]]
    current_location = test_duck.coord[-1]
    print("Current Location: ", test_duck.duckID, " , ", current_location)
    next_location = predict_next_location(G, current_location)
    print("Predicted next: ", next_location)