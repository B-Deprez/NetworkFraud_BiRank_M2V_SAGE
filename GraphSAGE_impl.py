import stellargraph as sg
import pandas as pd
import pickle as pkl
import numpy as np
from stellargraph import StellarGraph
import networkx as nx

from stellargraph.layer import HinSAGE    
from stellargraph.mapper import HinSAGENodeGenerator, NodeSequence
from keras import layers
from tensorflow.keras import layers, optimizers, Model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import EarlyStopping

import sklearn

def HinSAGE_embedding(HG, claim_data_features, labels, dimensions= [64,64], batch_size = 50, epochs = 50, train_size=0, val_size=0):
    # We will first extract the different nodes and edges 
    # in order to assign them the necesseary featrues
    claim_nodes = pd.DataFrame(index=  HG.nodes("claim"))
    claim_nodes.index.name = "ID"
    contract_nodes = pd.DataFrame(index= HG.nodes("contract"))
    contract_nodes.index.name = "ID"
    counterparty_nodes = pd.DataFrame(index= HG.nodes("counterparty"))
    counterparty_nodes.index.name = "ID"
    broker_nodes = pd.DataFrame(index= HG.nodes("broker"))
    broker_nodes.index.name = "ID"
    
    nodes = {
        "claim": claim_nodes, 
        "broker": broker_nodes, 
        "contract": contract_nodes, 
        "counterparty": counterparty_nodes
        }
    edges = HG.edges()
    
    # Initialise the features of the different node types
    # Only for the claims do we have additional information
    # The other features are set to 1, since HinSAGE requires all nodes to have features to work
    broker_nodes["Feature"] = 1
    contract_nodes["Feature"] = 1
    counterparty_nodes["Feature"] = 1
    claim_features = claim_data_features[claim_data_features["SI01_NO_SIN"].isin(claim_nodes.index)].reset_index(drop = True).set_index("SI01_NO_SIN")
    
    node_features = {
        "claim": claim_features, 
        "broker": broker_nodes[["Feature"]], 
        "contract": contract_nodes[["Feature"]], 
        "counterparty": counterparty_nodes[["Feature"]]
        }
    
    # The network is constructed in networkx in order to easily incorporate the features as well
    G_nx = nx.Graph()
    
    # For the nodes, iteration over the dictionary is needed
    for key, values in nodes.items(): 
        G_nx.add_nodes_from(list(values.index), ntype=key) 
        
    # Edges can just be added
    G_nx.add_edges_from(edges)
    
    # Construct the stellargraph object
    G_sg = sg.StellarGraph.from_networkx(G_nx, node_type_attr="ntype", node_features=node_features)

    # We want the index to be sorted in order to have the time dimension right
    labels.sort_index(inplace = True)
    
    train_subjects = labels.iloc[:train_size]
    val_subjects = labels.iloc[train_size:(train_size+val_size)]
    test_subjects = labels.iloc[(train_size+val_size):]
    
    # Set-up of the HinSAGE model
    num_samples = [2,32]
    embedding_node_type = "claim"

    es_callback = EarlyStopping(
        monitor="val_auc", 
        patience=5, 
        restore_best_weights=True
        )
    
    generator = HinSAGENodeGenerator(
        G_sg, 
        batch_size, 
        num_samples, 
        head_node_type = embedding_node_type
        )
    
    train_gen = generator.flow(
        train_subjects.index, 
        train_subjects["Proven_fraud"]
        )
    
    val_gen = generator.flow(
        val_subjects.index, 
        val_subjects["Proven_fraud"]
        )

    model = HinSAGE(
        layer_sizes = dimensions, 
        generator = generator, 
        dropout=0)
    
    x_inp, x_out = model.build() 
    
    prediction = layers.Dense(
        units=1, 
        activation="sigmoid", 
        dtype='float32')(x_out)
    
    model = Model(
        inputs = x_inp, 
        outputs=prediction
        )
    
    model.compile(
        optimizer = optimizers.Adam(lr=1e-3),
        loss=binary_crossentropy,
        metrics=["AUC"],
        )
    
    weights = sklearn.utils.class_weight.compute_class_weight(
        'balanced', 
        classes = np.unique(
            labels["Proven_fraud"]
            ),
        y = labels["Proven_fraud"].values
        )
    
    weights_dic = {0: weights[0], 1: weights[1]}
    
    model.fit(
        train_gen, 
        epochs = epochs, 
        validation_data = val_gen,
        shuffle=False, # this should be False, since shuffling data means shuffling the whole graph
        verbose = 2,
        callbacks=[es_callback],
        class_weight=weights_dic,
        )
    
    test_gen = generator.flow(test_subjects.index, test_subjects["Proven_fraud"])

    test_metrics = model.evaluate(test_gen)
    print("\nTest Set Metrics:")
    for name, val in zip(model.metrics_names, test_metrics):
        print("\t{}: {:0.4f}".format(name, val))
        
    # Make both the predictions and the embeddings according to HinSAGE
    full_gen = generator.flow(labels.index, labels["Proven_fraud"])
    
    full_prediction = model.predict(full_gen).squeeze()
    
    trained_model = Model(inputs=x_inp, outputs=x_out)
    embeddings = trained_model.predict(full_gen)

    full_emb = pd.DataFrame(embeddings, index = labels.index)
    full_emb["Prediction"] = full_prediction
    
    return full_emb
    