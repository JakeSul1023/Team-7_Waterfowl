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

    #Placeholder for total number of unique duck IDs
    total = 0 

    #Pulling entire column of duck id's (including repeats)
    pooledIDs = df['tag-local-identifier'].tolist()
    
    #Placeholder for list of unique duck IDs
    uniqueIDs = []

    #Count unique ID's
    for item in pooledIDs:
        #Check if unique ID is already logged
        if item not in uniqueIDs:
            #update total number of ducks
            total += 1
            #update list of unique IDs
            uniqueIDs.append(item)

    return total, uniqueIDs

def selectDucks(totalDucks, duckList):

    print("Total Number of unique duck IDs imported: ", totalDucks)
    userInput = input("How many ducks would you like to model with? ")
    portion = int(userInput)
    sampleList = random.sample(duckList, portion)

    return sampleList

def create_edges(ducks):

    edges = []
    for duck in ducks.values():
        for i in range(len(duck.coord) - 1):
            node1 = duck.coord[i]
            node2 = duck.coord[i + 1]
            if node1 != node2:
                edges.append((node1, node2))
    return edges

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

    all_coords = set()  

    for duck in ducks.values():
        all_coords.update(duck.coord) 

    #G.add_nodes_from(all_coords) 

    edges = create_edges(ducks)
    G.add_edges_from(edges)

    G.add_edges_from(edges)

    # Get the weight of each edge, defaulting to 1 if no weight is set
    weights = [G[u][v].get('weight', 1) for u, v in G.edges()]

    pos = nx.kamada_kawai_layout(G)  

    # Visualize the graph with weights as edge widths
    nx.draw(G, pos, with_labels=False, node_size=100, node_color='skyblue', font_weight='bold', width=weights)

    # Show the plot
    plt.show()

    test_duck = ducks[sampleDucks[0]]
    current_location = test_duck.coord[-1]
    print("Current Location: ", test_duck.duckID, " , ", current_location)
    next_location = predict_next_location(G, current_location)
    print("Predicted next: ", next_location)





