import pickle as pkl
import pandas as pd
import scipy as sp
import numpy as np
import scipy.sparse
from stellargraph import StellarGraph
from datetime import timedelta
from sklearn.preprocessing import OrdinalEncoder

def load_network():
    claim_data = pkl.load(open("data/claims_data", "rb"))

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
    
    HG = StellarGraph({"claim": claims_nodes, "car": cars_nodes, "policy": policy_nodes, "broker": broker_nodes}, edges)

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













