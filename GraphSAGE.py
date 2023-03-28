import stellargraph as sg
import pandas as pd
import pickle as pkl
import numpy as np
from stellargraph import StellarGraph
import networkx as nx

def HinSAGE_embedding(HG, claim_data_features, labels, simple_model_YN = True):
    # We will first extract the different nodes and edges 
    # in order to assign them the necesseary featrues
    claim_nodes = pd.DataFrame(index=  HG.nodes("claim"))
    claim_nodes.index.name = "ID"
    car_nodes = pd.DataFrame(index= HG.nodes("car"))
    car_nodes.index.name = "ID"
    policy_nodes = pd.DataFrame(index= HG.nodes("policy"))
    policy_nodes.index.name = "ID"
    broker_nodes = pd.DataFrame(index= HG.nodes("broker"))
    broker_nodes.index.name = "ID"
    
    nodes = {"claim": claim_nodes, "broker": broker_nodes, "car": car_nodes, "policy": policy_nodes}
    edges = HG.edges()
    
    # Initialise the features of the different node types
    # Only for the claims do we have additional information
    # The other features are set to 1, since HinSAGE requires all nodes to have features to work
    broker_nodes["Feature"] = 1
    car_nodes["Feature"] = 1
    policy_nodes["Feature"] = 1
    claim_features = claim_data_features[claim_data_features["SI01_NO_SIN"].isin(claim_nodes.index)].reset_index(drop = True).set_index("SI01_NO_SIN")
    
    # The network is constructed in networkx in order to easily incorporate the features as well
    G_nx = nx.Graph()
    
    # For the nodes, iteration over the dictionary is needed
    for key, values in nodes.items(): 
        G_nx.add_nodes_from(list(values.index), ntype=key) 
        
    # Edges can just be added
    
    
    