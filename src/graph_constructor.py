import numpy as np
import torch
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph
from torch_geometric.utils import add_self_loops

def construct_knn_graph(features, k=15):
    distances = pairwise_distances(features)
    sigma = np.mean(np.sort(distances, axis=1)[:, 1:k+1])
    weights = np.exp(-distances**2 / (2 * sigma**2))
    
    adjacency_matrix = kneighbors_graph(features, k, mode="connectivity", include_self=False)
    edge_index_np = np.array(adjacency_matrix.nonzero())
    edge_weights_np = weights[adjacency_matrix.nonzero()]
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)
    edge_weights = torch.tensor(edge_weights_np, dtype=torch.float32)
    
    edge_index, edge_weights = add_self_loops(edge_index, edge_weights, num_nodes=len(features))
    return edge_index, edge_weights