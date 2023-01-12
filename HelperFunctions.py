import pickle as pkl
import pandas as pd
import scipy as sp
import scipy.sparse
from stellargraph import StellarGraph

def load_network():
    broker_nodes = pkl.load(open("data/broker_nodes_brunosept.pkl", "rb"))
    cars_nodes = pkl.load(open("data/cars_nodes_brunosept.pkl", "rb"))
    claims_nodes = pkl.load(open("data/claims_nodes_brunosept.pkl", "rb"))
    policy_nodes = pkl.load(open("data/policy_nodes_brunosept.pkl", "rb"))
    edges = pkl.load(open("data/edges_brunosept.pkl", "rb"))

    labels = pd.DataFrame(pkl.load(open("data/Y", "rb")))
    labels.rename(columns={"y1": "Fraud", "y2": "Labelled"}, inplace=True)

    HG = StellarGraph({"claim": claims_nodes, "car": cars_nodes, "policy": policy_nodes, "broker": broker_nodes}, edges)

    return(HG)

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

