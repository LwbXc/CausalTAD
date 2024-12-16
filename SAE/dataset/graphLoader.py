import pdb
import pickle
import torch

class GraphLoader:

    def __init__(self, node_file, node_adj_file):

        self.nodes = pickle.load(open(node_file, 'rb'))
        self.node2id = {n:i for i,n in enumerate(self.nodes.keys())}
        self.load_adj(node_adj_file)
    
    def load_adj(self, node_adj_file):
        tmp = pickle.load(open(node_adj_file, 'rb'))

        self.node_adj = dict()
        for node, id in self.node2id.items():
            self.node_adj.setdefault(id, [])
            for neigh in tmp[node].keys():
                self.node_adj[id].append(self.node2id[neigh])

    def sample_subgraph(self, node_list):
        device = node_list.device
        points = list(set(node_list.view(-1).cpu().tolist()))
        sample_neighs = []
        for point in points:
            if point in self.node_adj.keys():
                neighs = self.node_adj[point]
            else:
                neighs = []
            sample_neighs.append(set(neighs))
            
        column_indices = [n for sample_neigh in sample_neighs for n in sample_neigh]
        row_indices = [points[i] for i in range(len(points)) for j in range(len(sample_neighs[i]))]
        sub_graph_edges = torch.LongTensor([row_indices, column_indices]).to(device)
        return sub_graph_edges