import pandas as pd
import numpy as np

def lift_curve_values(y_val, y_pred, steps):
    vals_lift = [] #The lift values to be plotted

    df_lift = pd.DataFrame()
    df_lift['Real'] = y_val
    df_lift['Pred'] = y_pred
    df_lift.sort_values('Pred',
                        ascending=False,
                        inplace=True)

    global_ratio = df_lift['Real'].sum() / len(df_lift['Real'])

    for step in steps:
        data_len = int(np.ceil(step*len(df_lift)))
        data_lift = df_lift.iloc[:data_len, :]
        val_lift = data_lift['Real'].sum()/data_len
        vals_lift.append(val_lift/global_ratio)

    return(vals_lift)

def complementarity_measure(Y_predictions, p):
    Y_predictions.columns = ["Real", "Model_1", "Model_2"]

    data_len = int(np.ceil(p * len(Y_predictions)))

    Y_pred_sort1 = Y_predictions.sort_values('Model_1', ascending=False).iloc[:data_len, :]
    Y_pred_sort2 = Y_predictions.sort_values('Model_2', ascending=False).iloc[:data_len, :]

    Y_fraud_model_1 = set(Y_pred_sort1[Y_pred_sort1['Real'] == 1].index)
    Y_fraud_model_2 = set(Y_pred_sort2[Y_pred_sort2['Real'] == 1].index)

    model_1_not_2 = len(Y_fraud_model_1.difference(Y_fraud_model_2))/len(Y_fraud_model_1)
    model_2_not_1 = len(Y_fraud_model_2.difference(Y_fraud_model_1)) / len(Y_fraud_model_2)

    return(model_1_not_2, model_2_not_1)