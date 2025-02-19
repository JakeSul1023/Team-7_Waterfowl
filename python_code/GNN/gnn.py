#Author: Revel Etheridge
#Date: 02-18-2025
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

#Location importing via csv
def importLoc():
    with open('ShortTermSetData(Aug-Sept).csv', mode='r')as file:
        df = pd.read_csv(file)
        print(df.head())

    #Isolating location data

    #Saving longitudes
    long = df['location-long'].tolist()

    #Testing column ioslation
    print("Longitudes: ")
    for line in long[:5]:
        print(line)

if __name__ == "__main__":

    #Read in CSV
    importLoc()






























'''
    try:
        #importing networkx module and veryfing accessibility
        import networkx as nx
        print("Networkx import success")

        #Graph initialization (empty)
        G = nx.Graph()

        #Adding first set of nodes
        G.add_nodes_from([2,3])

        #Creating edges to relate nodes
        edges = [(2,1),(2,2),(3,2),(4,3),(6,4),(7,5),(14,5)]
        G.add_edges_from(edges)

        #Printing graph
        nx.draw(G, with_labels=True, font_weight='bold')
        import matplotlib.pyplot as plt
        plt.show()

    except ImportError as e:
        print("Error -> ", e)
'''




