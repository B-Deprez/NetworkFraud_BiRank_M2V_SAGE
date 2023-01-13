import pickle as pkl
import pandas as pd
import scipy as sp
import scipy.sparse
from stellargraph import StellarGraph
from datetime import timedelta

def load_network():
    broker_nodes = pkl.load(open("data/broker_nodes_brunosept.pkl", "rb"))
    cars_nodes = pkl.load(open("data/cars_nodes_brunosept.pkl", "rb"))
    claims_nodes = pkl.load(open("data/claims_nodes_brunosept.pkl", "rb"))
    policy_nodes = pkl.load(open("data/policy_nodes_brunosept.pkl", "rb"))
    edges = pkl.load(open("data/edges_brunosept.pkl", "rb"))

    labels = pd.DataFrame(pkl.load(open("data/Y", "rb")))
    labels.rename(columns={"y1": "Fraud", "y2": "Labelled"}, inplace=True)

    HG = StellarGraph({"claim": claims_nodes, "car": cars_nodes, "policy": policy_nodes, "broker": broker_nodes}, edges)

    return(HG, labels)

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
    
    return(claims_data[selected_features])