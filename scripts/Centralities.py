from src.HelperFunctions import load_network, feature_engineering
import networkx as nx
import networkit
import pandas as pd


HG, labels, claim_data = load_network(fraud_node_tf=False)
nodes_nobrokers = list(HG.nodes("claim")) + list(HG.nodes("car")) + list(HG.nodes("policy"))
HG_nobrokers = HG.subgraph(nodes_nobrokers)
HG_nx_nobrokers = HG_nobrokers.to_networkx()

nx_graph_1 = HG.to_networkx()
nx_graph_2 = nx.Graph(nx_graph_1)
G_nit = networkit.nxadapter.nx2nk(nx_graph_2)

zipped_nodes = zip(nx_graph_2.nodes(), range(nx_graph_2.number_of_nodes()))
node_keys = pd.DataFrame(zipped_nodes)

cl_cen = networkit.centrality.ApproxCloseness(G_nit, 10000).run().ranking()
cl_cen_df = pd.DataFrame(cl_cen)

cl_cen_nodes = node_keys.merge(cl_cen_df, left_on = 1, right_on = 0)

cl_cen_nodes = cl_cen_nodes[["0_x", "1_y"]]
cl_cen_nodes.columns= ["node_id", "Closeness Centrality"]

btw_cen = networkit.centrality.EstimateBetweenness(G_nit, 10000).run().ranking()

btw_cen_df = pd.DataFrame(btw_cen)
btw_cen_nodes = node_keys.merge(btw_cen_df, left_on = 1, right_on = 0)
btw_cen_nodes = btw_cen_nodes[["0_x", "1_y"]]
btw_cen_nodes.columns= ["node_id", "Betweenness Centrality"]

centralities = cl_cen_nodes.merge(btw_cen_nodes, on = "node_id")
centralities.to_csv("Centralities.csv", index = False)