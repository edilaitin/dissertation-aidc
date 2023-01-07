import dgl
from dgl.data import DGLDataset
import torch
import numpy as np

class DGLGraph(DGLDataset):
    def __init__(self, graph_init):
        self.graph_init = graph_init
        super().__init__(name='dgl_graph')

    def process(self):
        src = []
        dest = []
        for edge in self.graph_init.edges:
            src.append(edge.node1.id)
            dest.append(edge.node2.id)

        self.graph = dgl.graph((src, dest), num_nodes=len(self.graph_init.nodes))
        features = [x.features for x in self.graph_init.nodes]
        labels = [1 for x in self.graph_init.nodes]
        edge_features = [x.features for x in self.graph_init.edges]

        self.graph.ndata['feat'] = torch.from_numpy(np.array(features))
        self.graph.ndata['label'] = torch.from_numpy(np.array(labels))
        self.graph.edata['constraints'] = torch.tensor(edge_features)

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1