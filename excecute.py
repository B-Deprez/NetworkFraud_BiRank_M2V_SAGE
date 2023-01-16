from BiRank import *
from metapath2vec import *
from HelperFunctions import to_bipartite
import pandas as pd
from sklearn import metrics
import pickle as pkl
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier

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
    split_size = int(round(train_size/2,0))
    test_set_size = number_claims - train_set_size

    fraud_train = {"FraudInd": labels["Fraud"].values[:split_size]}
    fraudMat_train = pd.DataFrame(fraud_train)
    test_set_fraud = {"FraudInd": [0] * (test_set_size + (train_size - split_size))}
    fraudMat_test_set = pd.DataFrame(test_set_fraud)
    fraudMat_test = fraudMat_train.append(fraudMat_test_set)

    ADJ = ADJ.transpose().tocsr()

    print("Starting BiRank calculations")
    Claims_res, Parties_res, aMat, iterations, convergence = BiRank(ADJ, claim_nodes, party_nodes, fraudMat_test)

    y = labels[train_set_size:].Fraud.values
    pred_bi = Claims_res.sort_values("ID")[train_set_size:].ScaledScore
    fpr_bi, tpr_bi, thresholds = metrics.roc_curve(y, pred_bi)
    plt.plot(fpr_bi, tpr_bi)
    plt.plot([0, 1], [0, 1], color="grey", alpha=0.5)
    plt.title("AUC: " + str(np.round(metrics.auc(fpr_bi, tpr_bi), 3)))
    plt.savefig("figures/AUC_BiRank_simple.pdf")
    plt.close()
    
    res_bi = pd.concat(
        [
            claim_nodes.reset_index().rename(columns={"ID": "Claim_ID"}), 
            Claims_res.sort_values("ID")
            ], 
        axis = 1
        )[["Claim_ID", "StdScore"]]

    return(pred_bi, fpr_bi, tpr_bi, res_bi)
    
    
def Metapath2Vec_subroutine(HG, labels):
    dimensions = 20
    num_walks = 1
    walk_length = 13  # Go from claim to claim via broker twice
    context_window_size = 10

    metapaths = [
        ["claim", "car", "claim"],
        ["claim", "car", "policy", "car", "claim"],
        ["claim", "car", "policy", "broker", "policy", "car", "claim"]
    ]

    node_ids, node_embeddings, node_targets = Metapath2vec(HG,
                                                           metapaths,
                                                           dimensions=dimensions,
                                                           num_walks=num_walks,
                                                           walk_length=walk_length,
                                                           context_window_size=context_window_size)

    claims_nodes = pkl.load(open("data/claims_nodes_brunosept.pkl", "rb"))

    embedding_df = pd.DataFrame(node_embeddings)
    embedding_df.index = node_ids
    claim_embedding_df = embedding_df.loc[list(claims_nodes.index)]
    embedding_fraud = claim_embedding_df.merge(labels, left_index=True, right_index=True)
    embedding_fraud.sort_index(inplace=True)

    train_size = int(round(0.6 * len(embedding_fraud), 0))

    X_train = embedding_fraud.iloc[:train_size, :20]
    y_train = embedding_fraud.iloc[:train_size, 20]

    X_test = embedding_fraud.iloc[train_size:, :20]
    y_test = embedding_fraud.iloc[train_size:, 20]

    embedding_model = GradientBoostingClassifier(n_estimators=100,
                                                 subsample=0.8,
                                                 max_depth=2,
                                                 random_state=1997).fit(X_train, y_train)

    y_pred_meta = embedding_model.predict_proba(X_test)[:, 1]
    fpr_meta, tpr_meta, thresholds = metrics.roc_curve(y_test, y_pred_meta)
    plt.plot(fpr_meta, tpr_meta)
    plt.plot([0, 1], [0, 1], color="grey", alpha=0.5)
    plt.title("AUC: " + str(np.round(metrics.auc(fpr_meta, tpr_meta), 3)))
    plt.savefig("figures/AUC_Metapath2vec_simple.pdf")
    plt.close()

    return(y_pred_meta, fpr_meta, tpr_meta, embedding_fraud)

def training_gradient_boosting(df_full, selected_features, name):
    train_size = int(round(0.6 * len(df_full), 0))
    split_size = int(round(train_size/2,0))
    
    X_full = df_full[selected_features]
    
    y_full = df_full["Fraud_y"]
                
    X_train = X_full.iloc[split_size:train_size, :]
    y_train = y_full[split_size:train_size]

    X_test = X_full.iloc[train_size:, :]
    y_test = y_full[train_size:]
    
    embedding_model = GradientBoostingClassifier(n_estimators=100,
                                                 subsample=0.8,
                                                 max_depth=2,
                                                 random_state=1997).fit(X_train, y_train)

    y_pred = embedding_model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], color="grey", alpha=0.5)
    plt.title("AUC: " + str(np.round(metrics.auc(fpr, tpr), 3)))
    plt.savefig("figures/AUC_full_model_"+str(name)+".pdf")
    plt.close()
    

def fullModel_subroutine(df_basic_features, df_BiRank_embedding, df_Metapath2Vec_embedding, labels):
    print("Putting everything together.")
    df_full = df_basic_features.merge(
        df_BiRank_embedding, 
        left_on = "SI01_NO_SIN",
        right_on = "Claim_ID", 
        how = "inner"
        ).merge(
            df_Metapath2Vec_embedding.reset_index(),
            left_on = "Claim_ID", 
            right_on = "index", 
            how = "inner"
            ).merge(
                labels.reset_index(),
                left_on = "Claim_ID", 
                right_on = "SI01_NO_SIN",
                how = "inner"
                )

    #Basic Model
    selected_features = ["Month_Accident", "Closest_Hour", "Reporting_delay", "Day_Accident", "SI01_C_FAM_PROD","SI01_C_CAU"]
    training_gradient_boosting(df_full, selected_features, "simple")
    
    
    #BiRank
    selected_features = ["Month_Accident", "Closest_Hour", "Reporting_delay", "Day_Accident", "SI01_C_FAM_PROD","SI01_C_CAU",
                         "StdScore",
                         #"n1_q1", "n1_med", "n1_max"
                         ]
    training_gradient_boosting(df_full, selected_features, "BiRank")
    
    #Metapath2Vec
    selected_features = ["Month_Accident", "Closest_Hour", "Reporting_delay", "Day_Accident", "SI01_C_FAM_PROD","SI01_C_CAU",
                         0,                   1,                   2,
                         3,                   4,                   5,
                         6,                   7,                   8,
                         9,                  10,                  11,
                        12,                  13,                  14,
                        15,                  16,                  17,
                        18,                  19]
    training_gradient_boosting(df_full, selected_features, "Metapath2Vec")
    
    #Full Model
    selected_features = ["Month_Accident", "Closest_Hour", "Reporting_delay", "Day_Accident", "SI01_C_FAM_PROD","SI01_C_CAU",
                         "StdScore",
                         0,                   1,                   2,
                         3,                   4,                   5,
                         6,                   7,                   8,
                         9,                  10,                  11,
                        12,                  13,                  14,
                        15,                  16,                  17,
                        18,                  19]
    training_gradient_boosting(df_full, selected_features, "Total")
    
    
    
    
    
    
    