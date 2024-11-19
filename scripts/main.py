from src.HelperFunctions import load_network, feature_engineering, simple_network_feature_engineering
import scripts.excecute as excecute

def run_model(dataset_1, fraud_node_tf):
    # Start by initialising the data
    print("Load data")
    HG, labels, claim_data = load_network(dataset_1, fraud_node_tf=False)
    #HG_F, labels, claim_data = load_network(fraud_node_tf=fraud_node_tf)
    
    print("#### BiRank ####")
    # Calculate all resutls for BiRank and generate figures
    pred_bi, fpr_bi, tpr_bi, res_bi = excecute.BiRank_subroutine(HG, labels, dataset_1)
    
    print("#### Metapath2vec ####")
    # Calculate all resutls for metapath and generate figures
    pred_meta, fpr_meta, tpr_meta, res_meta = excecute.Metapath2Vec_subroutine(HG, labels,dataset_1, fraud_node_tf=fraud_node_tf)
    
    print("#### Feature Engineering ####")
    # Feature engineering on claim specific data + selection of features
    claim_data_features = feature_engineering(claim_data)
    
    print("#### GraphSAGE ####")
    # Calculate all results for HinSAGE and generate figures
    y_pred_sage, fpr_sage, tpr_sage, res_sage = excecute.HinSAGE_subroutine(HG, claim_data_features, labels)    
    
    print("#### Simple network features ####")
    # Feature engineering for simple network features
    simple_network_features = simple_network_feature_engineering(HG, dataset_1)

    print("#### Full model ####")
    # Do the metrics calculations for full model
    excecute.fullModel_subroutine(claim_data_features, simple_network_features ,res_bi, res_meta, res_sage, labels)
    

if __name__ == '__main__':
    print("Runnig the model...")
    run_model(dataset_1=False, fraud_node_tf = False) #run all models
    print("All done!")



