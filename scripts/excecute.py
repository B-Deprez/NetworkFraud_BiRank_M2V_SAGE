from src.BiRank import *
from src.metapath2vec import *
from src.GraphSAGE_impl import *
from src.HelperFunctions import to_bipartite
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
import src.Metrics as Metrics
import numpy as np

def BiRank_subroutine(HG, labels, dataset_1):
    print("Starting BiRank calculations")
    
    if dataset_1:
        # Extract all nodes from the networks per type
        HG_claims = HG.nodes("claim")
        HG_cars = HG.nodes("car")
        HG_policies = HG.nodes("policy")
        HG_brokers = HG.nodes("broker")
        
        HG_parties = np.concatenate((HG_cars, HG_policies, HG_brokers))  
        
        # Build the bipartite adjacency matrix
        ADJ = to_bipartite(HG)

        fraud_label_column = "Fraud"

    else: 
        HG_claims = HG.nodes("claim")
        
        HG_parties = np.concatenate((HG.nodes("contract"), HG.nodes("broker"), HG.nodes("counterparty")))
        
        nodes = list(HG.nodes("claim"))+list(HG.nodes("contract"))+list(HG.nodes("broker"))+list(HG.nodes("counterparty"))
        ADJ = HG.to_adjacency_matrix(nodes)[:len(HG.nodes("claim")), len(HG.nodes("claim")):]

        fraud_label_column = "Proven_fraud"
        
    claim_nodes = pd.DataFrame({"ID": HG_claims}).sort_values("ID").set_index("ID")
    party_nodes = pd.DataFrame({"ID": HG_parties}).set_index("ID")
    
    
    # Set-up the fraud scores
    fraud = {"FraudInd": labels[fraud_label_column].values}
    fraudMat = pd.DataFrame(fraud)

    # Do the train-test split for both the nodes and their fraud labels
    number_claims = ADJ.shape[0]
    
    ADJ = ADJ.transpose().tocsr()
    
    train_set_size = int(np.round(0.6 * number_claims))
    claims_train = claim_nodes[:train_set_size]
    ADJ_train= ADJ[:,:train_set_size]
    test_set_size = number_claims-train_set_size
    split_size = int(round(train_set_size/2,0))

    #First train split
    fraud_train = {"FraudInd": labels[fraud_label_column].values[:split_size]}
    fraudMat_train = pd.DataFrame(fraud_train)
    validate_set_fraud = {"FraudInd": [0]*(train_set_size - split_size)}
    fraudMat_test_set = pd.DataFrame(validate_set_fraud)
    fraudMat_test = fraudMat_train.append(fraudMat_test_set)
    
    Claims_res_1, Parties_res_1, aMat_1, iterations_1, convergence_1 = BiRank(ADJ_train, claims_train, party_nodes, fraudMat_test)

    #Final scores for real test set
    fraud_score_train_validate = Claims_res_1.sort_values("ID")["Score"].values[:train_set_size]
    
    fraud_train_res = {"FraudInd": fraud_score_train_validate}
    test_set_fraud = {"FraudInd": [0]*test_set_size}
    fraudMat_train_res = pd.DataFrame(fraud_train_res)
    fraudMat_test_set = pd.DataFrame(test_set_fraud)
    fraudMat_test = fraudMat_train_res.append(fraudMat_test_set)

    Claims_res, Parties_res, aMat, iterations, convergence = BiRank(ADJ, claim_nodes, party_nodes, fraudMat_test)

    y = labels[train_set_size:][fraud_label_column].values
    pred_bi = Claims_res.sort_values("ID")[train_set_size:].ScaledScore
    fpr_bi, tpr_bi, thresholds = metrics.roc_curve(y, pred_bi)
    plt.plot(fpr_bi, tpr_bi)
    plt.plot([0, 1], [0, 1], color="grey", alpha=0.5)
    plt.title("AUC-ROC: " + str(np.round(metrics.auc(fpr_bi, tpr_bi), 3)))
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.savefig("figures/AUC_BiRank_simple.pdf")
    plt.close()
    
    
    Claims_res_test = Claims_res.sort_values("ID")[train_set_size:]
    frames = [Claims_res_1, Claims_res_test]
    Claims_res = pd.concat(frames)
    
    res_bi = pd.concat(
        [
            claim_nodes.reset_index().rename(columns={"ID": "Claim_ID"}), 
            Claims_res.sort_values("ID")
            ], 
        axis = 1
        )[["Claim_ID", "StdScore"]]

    return(pred_bi, fpr_bi, tpr_bi, res_bi)
    
    
def Metapath2Vec_subroutine(HG, labels, dataset_1, fraud_node_tf):
    dimensions = 64
    num_walks = 2
    walk_length = 7
    context_window_size = 5

    if dataset_1:
        metapaths = [
            ["claim", "car", "claim"],
            ["claim", "car", "policy", "car", "claim"],
            ["claim", "car", "policy", "broker", "policy", "car", "claim"]
            ]
    else:
        metapaths = [
            ["claim", "counterparty", "claim"],
            ["claim", "contract","claim"],
            ["claim", "broker", "claim"]
            ]
    
    if fraud_node_tf:
        metapaths.append(["claim", "fraud", "claim"])

    node_ids, node_embeddings, node_targets = Metapath2vec(HG,
                                                           metapaths,
                                                           dimensions=dimensions,
                                                           num_walks=num_walks,
                                                           walk_length=walk_length,
                                                           context_window_size=context_window_size)

    embedding_df = pd.DataFrame(node_embeddings)
    embedding_df.index = node_ids
    claim_embedding_df = embedding_df.loc[list(HG.nodes("claim"))]
    embedding_fraud = claim_embedding_df.merge(labels, left_index=True, right_index=True)
    embedding_fraud.sort_index(inplace=True)
    embedding_fraud.columns = ["Meta_"+str(i) for i in range(dimensions)] + list(labels.columns)

    train_size = int(round(0.6 * len(embedding_fraud), 0))

    X_train = embedding_fraud.iloc[:train_size, :dimensions]
    y_train = embedding_fraud.iloc[:train_size, dimensions]

    X_test = embedding_fraud.iloc[train_size:, :dimensions]
    y_test = embedding_fraud.iloc[train_size:, dimensions]
    
    print("Building the model...")

    embedding_model = GradientBoostingClassifier(n_estimators=100,
                                                 max_depth=2,
                                                 random_state=1997).fit(X_train, y_train)

    y_pred_meta = embedding_model.predict_proba(X_test)[:, 1]
    fpr_meta, tpr_meta, thresholds = metrics.roc_curve(y_test, y_pred_meta)
    plt.plot(fpr_meta, tpr_meta)
    plt.plot([0, 1], [0, 1], color="grey", alpha=0.5)
    plt.title("AUC-ROC: " + str(np.round(metrics.auc(fpr_meta, tpr_meta), 3)))
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.savefig("figures/AUC_Metapath2vec_simple.pdf")
    plt.close()

    return(y_pred_meta, fpr_meta, tpr_meta, embedding_fraud)

def HinSAGE_subroutine(HG, claim_data_features, labels):
    dimensions = [64, 64]
    batch_size = 50 
    epochs = 50
    
    train_size = int(np.round(0.5 * len(labels)))
    val_size = int(np.round(0.6 * len(labels))) - train_size #to have the same train test split as the ohters (otherwise mistakes possible via rounding)
    
    full_emb = HinSAGE_embedding(
        HG, 
        claim_data_features, 
        labels, 
        dimensions=dimensions, 
        batch_size = batch_size, 
        epochs = epochs,
        train_size = train_size,
        val_size = val_size
        )

    embedding_sage = full_emb.iloc[:, :dimensions[-1]]
    embedding_sage.columns = ["Sage_"+str(i) for i in range(64)]
    predictions_sage = full_emb.iloc[:, -1]
    
    y_pred_sage = predictions_sage[(train_size+val_size):] #everything after the train and validation are the predictions from the test set, hence the y_pred
    y_test = labels.sort_index()["Proven_fraud"][(train_size+val_size):]
    
    fpr_sage, tpr_sage, thresholds = metrics.roc_curve(y_test, y_pred_sage)
    plt.plot(fpr_sage, tpr_sage)
    plt.plot([0, 1], [0, 1], color="grey", alpha=0.5)
    plt.title("AUC-ROC: " + str(np.round(metrics.auc(fpr_sage, tpr_sage), 3)))
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.savefig("figures/AUC_GraphSAGE_simple.pdf")
    plt.close()

    return(y_pred_sage, fpr_sage, tpr_sage, embedding_sage)

def training_gradient_boosting(df_full, selected_features, name):
    train_size = int(round(0.6 * len(df_full), 0))
    split_size = int(round(train_size/2,0))
    
    X_full = df_full[selected_features]
    
    try:
        y_full = df_full["Fraud_y"]
    
    except:
        y_full = df_full["Proven_fraud_y"]
                
    X_train = X_full.iloc[split_size:train_size, :]
    y_train = y_full[split_size:train_size]

    X_test = X_full.iloc[train_size:, :]
    y_test = y_full[train_size:]
    
    embedding_model = GradientBoostingClassifier(n_estimators=100,
                                                 max_depth=2,
                                                 random_state=1997).fit(X_train, y_train)

    y_pred = embedding_model.predict_proba(X_test)[:, 1]
    
    return(y_test, y_pred)

def AUC_plot(y_test, y_pred, close_plot, name, plotname):
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    AUC = np.round(metrics.auc(fpr, tpr), 3)
    plt.plot(fpr, tpr, alpha = 0.7, label = name + ": " + str(AUC))
    
    # plt.savefig("figures/AUC_full_model_"+str(name)+".pdf")
    
    if close_plot: 
        plt.title("AUC-ROC")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.plot([0, 1], [0, 1], color="grey", alpha=0.5)
        plt.legend()
        plt.savefig("figures/AUC_"+str(plotname)+".pdf")
        plt.close()
        
def AP_plot(y_test, y_pred, close_plot, name, plotname):
    precision_base, recall_base, thresholds = metrics.precision_recall_curve(y_test,y_pred)
    AP = np.round(metrics.average_precision_score(y_test, y_pred),3)
    plt.plot(recall_base,precision_base, label = name +": "+ str(AP), alpha =0.7)
    
    if close_plot:
        plt.title("AUC-PRC")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.savefig("figures/AP_"+str(plotname)+".pdf")
        plt.close()
    
def lift_plot(y_test, y_pred, close_plot, name, plotname):
    steps = np.arange(start = 0.05, stop = 1, step = 0.001)
    lft_base = Metrics.lift_curve_values(y_test, y_pred, steps)
    plt.plot(steps, lft_base, alpha = 0.7, label = name)
    
    if close_plot:
        plt.title("Lift Curve")
        plt.xlabel("p")
        plt.ylabel("Lift")
        plt.legend()
        plt.savefig("figures/Lift_"+str(plotname)+".pdf")
        plt.close()

def comp_plot(y_test, y_pred_1, y_pred_2, name):
    Y_pred = pd.DataFrame() 
    Y_pred['real'] = y_test
    Y_pred['1'] = y_pred_1
    Y_pred['2'] = y_pred_2

    percentages = [1, 5, 10, 20, 50]
    complementarity_12 = []
    complementarity_21 = []

    for i in percentages:
        p = i/100
        comp_12, comp_21 = Metrics.complementarity_measure(Y_pred, p)
        complementarity_12.append(comp_12)
        complementarity_21.append(comp_21)

    N = 5
    ind = np.arange(N)  # the x locations for the groups
    width = 0.27       # the width of the bars
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    rects1 = ax.bar(ind-width/2, complementarity_12, width, color ="#52BDEC")
    rects2 = ax.bar(ind+width/2, complementarity_21, width, color = "#00407A")
    
    ax.set_ylabel('Complementarity')
    ax.set_xticks(ind)
    ax.set_xticklabels( ('1%', '5%', '10%', '20%', '50%') )
    ax.legend( (rects1[0], rects2[0]), ('Model 1 not 2', 'Model 2 not 1') )
    
    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            ax.text(rect.get_x()+rect.get_width()/2., h, '%d'%int(h*100),
                    ha='center', va='bottom')
            
    autolabel(rects1)
    autolabel(rects2)

    plt.savefig("figures/Complementary_"+str(name)+".pdf")
    plt.close()

def fullModel_subroutine(df_basic_features, df_simple_network, df_BiRank_embedding, df_Metapath2Vec_embedding, df_GraphSage_embedding, labels):
    print("Putting everything together.")
    df_full = df_basic_features.merge(
        df_simple_network,
        left_on = "SI01_NO_SIN",
        right_on = "node_id", 
        how = "inner"
        ).merge(
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
                    df_GraphSage_embedding.reset_index(),
                    on = "SI01_NO_SIN",
                    how = "inner"
                    ).merge(
                        labels.reset_index(),
                        left_on = "Claim_ID", 
                        right_on = "SI01_NO_SIN",
                        how = "inner"
                        ).sort_values("Claim_ID")

    #Basic Model
    selected_features = ["Month_Accident", "Closest_Hour", "Reporting_delay", "Day_Accident", "SI01_C_FAM_PROD","SI01_C_CAU"]
    y_test_simple, y_pred_simple = training_gradient_boosting(df_full, selected_features, "simple")
    
    #Simple network feature
    selected_features = ["Month_Accident", "Closest_Hour", "Reporting_delay", "Day_Accident", "SI01_C_FAM_PROD","SI01_C_CAU",
                         "Geodesic distance", "Number of cycles", "Betweenness Centrality", "degree"
                         ]
    
    y_test_simple_network, y_pred_simple_network = training_gradient_boosting(df_full, selected_features, "simple")
    
    #BiRank
    selected_features = ["Month_Accident", "Closest_Hour", "Reporting_delay", "Day_Accident", "SI01_C_FAM_PROD","SI01_C_CAU",
                         "StdScore",
                         #"n1_q1", "n1_med", "n1_max"
                         ]
    y_test_BiRank, y_pred_BiRank = training_gradient_boosting(df_full, selected_features, "BiRank")
    
    #Metapath2Vec
    selected_features = ["Month_Accident", "Closest_Hour", "Reporting_delay", "Day_Accident", "SI01_C_FAM_PROD","SI01_C_CAU"] + \
        ["Meta_"+str(i) for i in range(64)]
    y_test_Meta, y_pred_Meta = training_gradient_boosting(df_full, selected_features, "Metapath2Vec")
    
    #HinSAGE
    selected_features = ["Month_Accident", "Closest_Hour", "Reporting_delay", "Day_Accident", "SI01_C_FAM_PROD","SI01_C_CAU"] + \
        ["Sage_"+str(i) for i in range(64)]
    y_test_sage, y_pred_sage = training_gradient_boosting(df_full, selected_features, "Metapath2Vec")
    
    #Full Model
    #selected_features = ["Month_Accident", "Closest_Hour", "Reporting_delay", "Day_Accident", "SI01_C_FAM_PROD","SI01_C_CAU",
    #                     "StdScore",
    #                     "Geodesic distance", "Number of cycles", "Betweenness Centrality", "degree"] + \
    #    ["Meta_"+str(i) for i in range(64)] + \
    #        ["Sage_"+str(i) for i in range(64)]
    #y_test_full, y_pred_full = training_gradient_boosting(df_full, selected_features, "Total")
    
    #Plot the AUC together
    AUC_plot(y_test_simple, y_pred_simple, close_plot=False, name="Simple Model", plotname="full")
    AUC_plot(y_test_simple_network, y_pred_simple_network, close_plot=False, name="Simple Network Features", plotname="full")
    AUC_plot(y_test_BiRank, y_pred_BiRank, close_plot=False, name="BiRank", plotname="full")
    AUC_plot(y_test_Meta, y_pred_Meta, close_plot=False, name="Metapath2Vec", plotname="full")
    AUC_plot(y_test_sage, y_pred_sage, close_plot=True, name="GraphSAGE", plotname="full")
    #AUC_plot(y_test_full, y_pred_full, close_plot=True, name="Full Model", plotname="full")
    
    #Plot the AP together
    AP_plot(y_test_simple, y_pred_simple, close_plot=False, name="Simple Model", plotname="full")
    AP_plot(y_test_simple_network, y_pred_simple_network, close_plot=False, name="Simple Network Features", plotname="full")
    AP_plot(y_test_BiRank, y_pred_BiRank, close_plot=False, name="BiRank", plotname="full")
    AP_plot(y_test_Meta, y_pred_Meta, close_plot=False, name="Metapath2Vec", plotname="full")
    AP_plot(y_test_sage, y_pred_sage, close_plot=True, name="GraphSAGE", plotname="full")
    #AP_plot(y_test_full, y_pred_full, close_plot=True, name="Full Model", plotname="full")

    #Plot the lift curves
    lift_plot(y_test_simple, y_pred_simple, close_plot=False, name="Simple Model", plotname="full")
    lift_plot(y_test_simple_network, y_pred_simple_network, close_plot=False, name="Simple Network Features", plotname="full")
    lift_plot(y_test_BiRank, y_pred_BiRank, close_plot=False, name="BiRank", plotname="full")
    lift_plot(y_test_Meta, y_pred_Meta, close_plot=False, name="Metapath2Vec", plotname="full")
    lift_plot(y_test_sage, y_pred_sage, close_plot=True, name="GraphSAGE", plotname="full")
    #lift_plot(y_test_full, y_pred_full, close_plot=True, name="Full Model", plotname="full")
    
    #Plot the complementarity
    comp_plot(y_test_simple, y_pred_simple, y_pred_simple_network, name="Simple Network Features")
    comp_plot(y_test_simple, y_pred_simple, y_pred_BiRank, name = "BiRank")
    comp_plot(y_test_simple, y_pred_simple, y_pred_Meta, name = "Meta")
    comp_plot(y_test_simple, y_pred_simple, y_pred_sage, name = "Sage")
    #comp_plot(y_test_simple, y_pred_simple, y_pred_full, name = "Full")
    
    
    