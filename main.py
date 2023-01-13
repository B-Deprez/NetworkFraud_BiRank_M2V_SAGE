from HelperFunctions import load_network
import excecute

def run_model():
    # Start by initialising the data
    HG, labels = load_network()

    # Calculate all resutls for BiRank and generate figures
    pred_bi, fpr_bi, tpr_bi, res_bi = excecute.BiRank_subroutine(HG, labels)
    # Calculate all resutls for BiRank and generate figures
    pred_meta, fpr_meta, tpr_meta, res_meta = excecute.Metapath2Vec_subroutine(HG, labels)
    
    
    

if __name__ == '__main__':
    print("Runnig the model...")
    run_model() #run all models
    print("All done!")



