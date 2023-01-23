from HelperFunctions import load_network, feature_engineering, simple_network_feature_engineering
import excecute

def run_model(dataset_1, fraud_node_tf):
    # Start by initialising the data
    HG, labels, claim_data = load_network(dataset_1, fraud_node_tf=False)
    #HG_F, labels, claim_data = load_network(fraud_node_tf=fraud_node_tf)
    
    # Calculate all resutls for BiRank and generate figures
    pred_bi, fpr_bi, tpr_bi, res_bi = excecute.BiRank_subroutine(HG, labels, dataset_1)
    
    # Calculate all resutls for BiRank and generate figures
    pred_meta, fpr_meta, tpr_meta, res_meta = excecute.Metapath2Vec_subroutine(HG, labels,dataset_1, fraud_node_tf=fraud_node_tf)
    
    # Feature engineering on claim specific data + selection of features
    claim_data_features = feature_engineering(claim_data)
    
    # Feature engineering for simple network features
    simple_network_features = simple_network_feature_engineering(HG, dataset_1)

    # Do the metrics calculations for full model
    excecute.fullModel_subroutine(claim_data_features, simple_network_features ,res_bi, res_meta, labels)
    

if __name__ == '__main__':
    print("Runnig the model...")
    run_model(dataset_1=False, fraud_node_tf = False) #run all models
    print("All done!")



