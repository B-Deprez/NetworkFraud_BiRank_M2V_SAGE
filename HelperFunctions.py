import pickle as pkl
import pandas as pd
import scipy as sp
import numpy as np
import scipy.sparse
from stellargraph import StellarGraph
from datetime import timedelta
from sklearn.preprocessing import OrdinalEncoder
import networkx as nx

def load_network(dataset_1,fraud_node_tf=False):
    if dataset_1:
        claim_data = pkl.load(open( "data/claims_data", "rb" ))
        broker_nodes = pkl.load(open("data/broker_nodes_brunosept.pkl", "rb"))
        cars_nodes = pkl.load(open("data/cars_nodes_brunosept.pkl", "rb"))
        claims_nodes = pkl.load(open("data/claims_nodes_brunosept.pkl", "rb"))
        policy_nodes = pkl.load(open("data/policy_nodes_brunosept.pkl", "rb"))
        edges = pkl.load(open("data/edges_brunosept.pkl", "rb"))

        to_exclude = set(claims_nodes.index).difference(set(claim_data["SI01_NO_SIN"].values))
        to_include = edges.target[edges.target.isin(list(to_exclude))==False].index
        edges = edges.loc[to_include]

        claims_nodes = claims_nodes.loc[np.array(set(claim_data["SI01_NO_SIN"].values).intersection(set(claims_nodes.index)))]
    
        labels = pd.DataFrame(pkl.load(open("data/Y", "rb")))
        labels.rename(columns={"y1": "Fraud", "y2": "Labelled"}, inplace=True)
        labels = labels.loc[claims_nodes.index]
        labels.index.name = "SI01_NO_SIN"
    
        if fraud_node_tf:
            #Add the artificial node
            fraud_node = pd.DataFrame(index =["F"])
            fraud_node.index.rename("ID", inplace = True)
            #Edge from claim to  node must exist when fraud label = 1
            fraud_edges = labels[["Fraud"]].reset_index()
            fraud_edges = fraud_edges[fraud_edges["Fraud"] == 1]
            fraud_edges["target"] = fraud_edges["SI01_NO_SIN"]
            fraud_edges["source"] = "F"
            fraud_edges = fraud_edges[["source", "target"]]
            #Add new edges to existing ones
            edges_F = pd.concat([edges, fraud_edges]).reset_index(drop=True)
            #Build graph with artificial node
            HG = StellarGraph({"claim": claims_nodes, "car": cars_nodes, "policy": policy_nodes, "broker": broker_nodes,
                               "fraud": fraud_node}, edges_F)
        
        else:
            #No artificial node is added
            HG = StellarGraph({"claim": claims_nodes, "car": cars_nodes, "policy": policy_nodes, "broker": broker_nodes}, edges)

    
    else:
        claim_data = pkl.load(open( "data/claims_data", "rb" ))
        counterparty_data = pkl.load(open("data/counterparties", "rb"))
        labels = pkl.load(open("data/frauds", "rb")).sort_values("SI01_NO_SIN")

        claims_nodes = claim_data[["SI01_NO_SIN"]].drop_duplicates().set_index("SI01_NO_SIN").sort_values("SI01_NO_SIN")
        contract_nodes = claim_data[["SI01_NO_CNT"]].drop_duplicates().set_index("SI01_NO_CNT")
        broker_nodes = claim_data[["SI01_C_INTER"]].drop_duplicates().set_index("SI01_C_INTER")
        
        counterparty_nodes = counterparty_data[["C-TIE"]].drop_duplicates()
        counterparty_nodes = counterparty_nodes[[cp not in broker_nodes.index for cp in counterparty_nodes["C-TIE"]]]
        counterparty_nodes = counterparty_nodes.set_index("C-TIE")
        
        claim_contract = claim_data[["SI01_NO_SIN", "SI01_NO_CNT"]].reset_index(drop=True)
        claim_broker = claim_data[["SI01_NO_SIN", "SI01_C_INTER"]].reset_index(drop=True)
        claim_counter = counterparty_data[["NO-SIN", "C-TIE"]].reset_index(drop=True)
        
        claim_contract.columns = ["source", "target"]
        claim_broker.columns = ["source", "target"]
        claim_counter.columns = ["source", "target"]
        
        edges = pd.concat([claim_contract, claim_broker,claim_counter]).reset_index(drop = True)
        
        HG = StellarGraph({"claim" : claims_nodes, "contract" : contract_nodes, "broker" : broker_nodes, "counterparty" : counterparty_nodes}, edges)
        

    return(HG, labels, claim_data)

def to_bipartite(HG):
    HG_claims = HG.nodes("claim")
    HG_cars = HG.nodes("car")
    HG_sub = list(HG_claims) + list(HG_cars)
    adjmat_claim_car = HG.to_adjacency_matrix(HG_sub)[:len(HG_claims), len(HG_claims):]

    HG_policies = HG.nodes("policy")
    HG_sub = list(HG_cars) + list(HG_policies)
    adjmat_car_policy = HG.to_adjacency_matrix(HG_sub)[:len(HG_cars), len(HG_cars):]

    HG_brokers = HG.nodes("broker")
    HG_sub = list(HG_policies) + list(HG_brokers)
    adjmat_policy_broker = HG.to_adjacency_matrix(HG_sub)[:len(HG_policies), len(HG_policies):]

    C = adjmat_claim_car
    P = adjmat_car_policy
    B = adjmat_policy_broker
    CP = C @ P
    CB = CP @ B

    A_bipartite = sp.sparse.hstack(
        (
            C,
            CP,
            CB
        )
    ).tocsr()

    return(A_bipartite)

def feature_engineering(claims_data):
    reporting_delay =[min(timedelta(days = 90), delay) for delay in claims_data["SI01_D_DCL"]-claims_data["SI01_D_SURV_SIN"]]
    claims_data["Reporting_delay"] = reporting_delay

    claims_data["Day_Accident"] = claims_data["SI01_D_SURV_SIN"].dt.weekday
    claims_data["Month_Accident"] = claims_data["SI01_D_SURV_SIN"].dt.month

    claims_data["SI01_H_SIN"].replace(0, pd.NA, inplace = True)
    decimal_hours = claims_data["SI01_H_SIN"]//100+claims_data["SI01_H_SIN"]%100/60

    #Some wrong hours. This is set to 0 = <NA>
    raw_hours = [round(h,2) if (str(h) != '<NA>') & (str(h) != 'nan') else 12 for h in decimal_hours ]
    claims_data["Closest_Hour"] = [h if h<= 24 else 0 for h in raw_hours]
    
    selected_features = ["SI01_NO_SIN",
                     "SI01_C_CAU", 
                     "SI01_C_FAM_PROD",
                     "Reporting_delay",
                     "Day_Accident", 
                     "Month_Accident",
                     "Closest_Hour"]
    
    claims_data = claims_data[selected_features]
    
    #Encode all factor features using sequential encoding
    #Need to know the factor columns first
    all_columns =  set(claims_data.columns)
    numeric_columns = set(claims_data.describe().columns)
    factor_columns = all_columns.difference(numeric_columns).difference(set(["SI01_NO_SIN"]))
    #Do the encoding
    columns = [*factor_columns]
    first_column = columns[0]
    for column in columns:
        enc = OrdinalEncoder()
        X_encoder = enc.fit(claims_data[[column]])
        X_encoded = X_encoder.transform(claims_data[[column]])
        if column == first_column:
            X_full_encoded = X_encoded
        else:
            X_full_encoded = np.hstack((X_full_encoded, X_encoded))
        
    X_full_encoded = np.hstack((X_full_encoded, claims_data[["SI01_NO_SIN"]]))
    
    #Get the encoding ready to add to full dataset again
    X_full_encoded_df = pd.DataFrame(X_full_encoded)
    columns.append("SI01_NO_SIN")
    X_full_encoded_df.columns = columns
    
    columns_numeric = [*numeric_columns]
    columns_numeric.append("SI01_NO_SIN")
    df_full = claims_data[columns_numeric].merge(X_full_encoded_df, on = "SI01_NO_SIN")
    df_full['Reporting_delay']=df_full['Reporting_delay'].dt.days
    
    return(df_full)

def geodesic(G):
    simple_graph = nx.Graph(G)
    cycles_G = nx.cycle_basis(simple_graph)

    dict_cycle_lengths = {}
    dict_cycle_num = {}
    for cycle in cycles_G:
        for node in cycle:
            if node not in dict_cycle_lengths:
                dict_cycle_lengths[node] = []
                dict_cycle_num[node] = 0
            dict_cycle_lengths[node].append(len(cycle))
            dict_cycle_num[node] += 1

    dict_geodesic = dict((n, min(l)) for n, l in dict_cycle_lengths.items())
    df_geodesic = pd.DataFrame({'Item': [item for item in dict_geodesic],
                                'Geodesic distance': [dict_geodesic[item] for item in dict_geodesic],
                                'Number of cycles': [dict_cycle_num[item] for item in dict_cycle_num]})
    return(df_geodesic)

def simple_network_feature_engineering(HG, dataset_1):
    if dataset_1:
        nodes_nobrokers = list(HG.nodes("claim")) + list(HG.nodes("car")) + list(HG.nodes("policy"))
    else:
        nodes_nobrokers = list(HG.nodes("claim"))+list(HG.nodes("contract"))+list(HG.nodes("counterparty"))
    HG_nobrokers = HG.subgraph(nodes_nobrokers)
    HG_nx_nobrokers = HG_nobrokers.to_networkx()
    
    
    ## Features based on cycles
    # Select the cycle-features for the claims only 
    full_geo_G = geodesic(HG_nx_nobrokers)
    Geo_claims = full_geo_G[full_geo_G['Item'].isin(HG.nodes("claim"))]
    # We set the value for claims not part of a cycle to 0
    full_geo_claims = pd.DataFrame({"Item":HG.nodes("claim")}).merge(Geo_claims, on = "Item", how = "outer").fillna(0)
    
    ## Features based on centrality
    # Degree centrality
    HG_nx = HG.to_networkx()
    deg_cen = nx.degree_centrality(HG_nx)
    df_degcen = pd.DataFrame({'claim': [claim for claim in HG.nodes("claim")],
                              'degree': [deg_cen[claim] for claim in HG.nodes("claim")] })
    
    # Calculated using other sub-routine to save time
    centralities = pd.read_csv("Centralities\Centralities.csv", low_memory=False)
    
    claim_centralities = centralities[centralities["node_id"].isin(HG.nodes("claim"))].sort_values("node_id").fillna(0)
    claim_centralities = claim_centralities.merge(df_degcen, left_on = "node_id", right_on = "claim")[["node_id", "Closeness Centrality", "Betweenness Centrality", "degree"]]
    
    df_simple = full_geo_claims.merge(claim_centralities, left_on = "Item", right_on = "node_id")[["node_id", "Geodesic distance", "Number of cycles", "Closeness Centrality", "Betweenness Centrality", "degree"]]
    
    return(df_simple)



