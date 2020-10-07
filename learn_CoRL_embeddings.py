import os
import networkx as nx
from gensim.models import Word2Vec
from algorithms.node2vec import node2vec

'''
To learn the CoRL-based embeddings of users and items by different meta-paths,
we can flexibly apply the meta-path-based random walk mechanism of Metapath2Vec model or Node2Vec (DFS/BFS) 
with set of filtered target-node's types.
To simply the implementation of this source code, we demonstrated the approach of using Node2Vec 
to learn the CoRL-based embeddings of users and items
'''

# dataset = 'amazon-baby-100k'
dataset = 'amazon-baby-small'

metapath_dir = './outputs/{}/meta-paths'.format(dataset)
output_embeddings_dir = './outputs/{}/embeddings'.format(dataset)

emb_dim = 128
walk_len = 5
win_size = 3
num_walk = 10
seed = 1

metapaths = ['ici', 'uiu', 'uiciu']


def read_graph(input_graph_file_path):
    print('Reading graph file: {}'.format(input_graph_file_path))
    G = nx.Graph()
    with open(input_graph_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if len(line) > 0:
                u, v, w = line.split('\t')
                G.add_edge(int(u), int(v))

    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1
    return G


for metapath in metapaths:
    input_file = os.path.join(metapath_dir, metapath + '.txt')
    output_file = os.path.join(output_embeddings_dir, metapath + '.embedding')
    p = 1
    q = 1
    nx_G = read_graph(input_file)
    G = node2vec.Graph(nx_G, False, p, q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(walk_len, walk_len)
    walks = [[str(step) for step in walk] for walk in walks]

    print("Training...")
    model = Word2Vec(walks,
                     size=emb_dim,
                     window=win_size,
                     min_count=2,
                     sg=1,
                     hs=1,
                     iter=10,
                     workers=8)
    model.wv.save_word2vec_format(output_file)
