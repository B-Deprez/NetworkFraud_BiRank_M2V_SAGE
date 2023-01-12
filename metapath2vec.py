from stellargraph.data import UniformRandomMetaPathWalk
from gensim.models import Word2Vec
import multiprocessing

def Metapath2vec(G, metapaths, dimensions = 128, num_walks = 1, walk_length = 100, context_window_size = 10):
    rw = UniformRandomMetaPathWalk(G)
    walks = rw.run(
        G.nodes(), n=num_walks, length=walk_length, metapaths=metapaths
    )
    print("Number of random walks: {}".format(len(walks)))

    workers = multiprocessing.cpu_count()
    model = Word2Vec(
        walks,
        window=context_window_size,
        min_count=0,
        sg=1,
        workers=workers,
        vector_size=dimensions
    )

    node_ids = model.wv.index_to_key  # list of node IDs
    node_embeddings = (
        model.wv.vectors
    )  # numpy.ndarray of size number of nodes times embeddings dimensionality
    node_targets = [G.node_type(node_id) for node_id in node_ids]

    return node_ids, node_embeddings, node_targets
