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
    userInput = input("How many ducks would you like to model with? ")
    portion = int(userInput)
    sampleList = random.sample(duckList, portion)

    return sampleList

def create_nodes(ducks):
    nodes = set()  
    for duck in ducks.values():
        nodes.update(duck.coord) 
    return list(nodes)  

def create_edges(ducks):

    edges = []
    for duck in ducks.values():
        for i in range(len(duck.coord) - 1):
            node1 = duck.coord[i]
            node2 