from BiRank import *
from metapath2vec import *
from HelperFunctions import to_bipartite
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt

def BiRank_subroutine(HG, labels):
    # Extract all nodes from the networks per type
    HG_claims = HG.nodes("claim")
    HG_cars = HG.nodes("car")
    HG_policies = HG.nodes("policy")
    HG_brokers = HG.nodes("broker")

    # Split the nodes into two groups
    claim_nodes = pd.DataFrame({"ID": HG_claims}).set_index("ID")

    HG_parties = np.concatenate((HG_cars, HG_policies, HG_brokers))
    party_nodes = pd.DataFrame({"ID": HG_parties}).set_index("ID")

    # Build the bipartite adjacency matrix
    ADJ = to_bipartite(HG)

    # Set-up the fraud scores
    fraud = {"FraudInd": labels["Fraud"].values}
    fraudMat = pd.DataFrame(fraud)

     # Do the train-test split for both the nodes and their fraud labels
    number_claims = ADJ.shape[0]
    train_set_size = int(np.round(0.6 * number_claims))
    test_set_size = number_claims - train_set_size

    fraud_train = {"FraudInd": labels["Fraud"].values[:train_set_size]}
    fraudMat_train = pd.DataFrame(fraud_train)
    test_set_fraud = {"FraudInd": [0] * test_set_size}
    fraudMat_test_set = pd.DataFrame(test_set_fraud)
    fraudMat_test = fraudMat_train.append(fraudMat_test_set)

    ADJ = ADJ.transpose().tocsr()

    print("Starting BiRank calculations")
    Claims_res, Parties_res, aMat, iterations, convergence = BiRank(ADJ, claim_nodes, party_nodes, fraudMat_test)

    y = labels[train_set_size:].Fraud.values
    pred = Claims_res.sort_values("ID")[train_set_size:].ScaledScore
    fpr_bi, tpr_bi, thresholds = metrics.roc_curve(y, pred)
    plt.plot(fpr_bi, tpr_bi)
    plt.plot([0, 1], [0, 1], color="grey", alpha=0.5)
    plt.title("AUC: " + str(np.round(metrics.auc(fpr_bi, tpr_bi), 3)))
    plt.savefig("figures/AUC_BiRank_simple.pdf")

def Metapath2Vec_subroutine():
    print("")