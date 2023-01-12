from stellargraph.data import UniformRandomMetaPathWalk

# Create the random walker
rw = UniformRandomMetaPathWalk(HG)
# specify the metapath schemas as a list of lists of node types (should start and end with "claim" !).
metapaths = [
    ["claim", "car", "claim"],
    ["claim", "car","policy","car","claim"]
]
walks = rw.run(
    nodes=list(HG.nodes()),  # root nodes
    length=5,  # maximum length of a random walk
    n=1,  # number of random walks per root node
    metapaths=metapaths,  # the metapaths
)

from gensim.models import Word2Vec

model = Word2Vec(walks, window=5, min_count=0, sg=1, workers=16)
model.wv.vectors.shape

print("Number of random walks: {}".format(len(walks)))

## visualisation (optional)
# Retrieve node embeddings and corresponding subjects
node_ids = model.wv.index_to_key  # list of node IDs
node_embeddings = (
    model.wv.vectors
)  # numpy.ndarray of size number of nodes times embeddings dimensionality
node_targets = [HG.node_type(node_id) for node_id in node_ids]
from sklearn.manifold import TSNE
import matplotlib
import numpy as np
#matplotlib.use("TkAgg")  # Do this before importing pyplot!
import matplotlib.pyplot as plt

transform = TSNE  # PCA

trans = transform(n_components=2)
node_embeddings_2d = trans.fit_transform(node_embeddings)
# draw the points
label_map = {l: i for i, l in enumerate(np.unique(node_targets))}
node_colours = [label_map[target] for target in node_targets]

plt.figure(figsize=(20, 16))
plt.axes().set(aspect="equal")
plt.scatter(x=node_embeddings_2d[:, 0], y=node_embeddings_2d[:, 1], c=node_colours, alpha=0.3)
plt.title("{} visualization of node embeddings".format(transform.__name__))
plt.show()