import dgl
from dgl.data import DGLDataset
import torch
import numpy as np

class DGLGraph(DGLDataset):
    def __init__(self, graph_init):
        self.graph_init = graph_init
        super().__init__(name='dgl_graph')

    def process(self):
        components_src = []
        components_dest = []
        for edge in self.graph_init.edges:
            components_src.append(edge.node1.id - 1)
            components_dest.append(edge.node2.id - 1)

        size_components = len([n for n in self.graph_init.nodes if n.get_type() == "component"])
        size_vms = len([n for n in self.graph_init.nodes if n.get_type() == "vm"])

        link_src = []
        link_dest = []
        unlink_src = []
        unlink_dest = []
        for link in self.graph_init.links:
            if (link.features == 1):
                link_src.append(link.node1.id - 1)
                link_dest.append(link.node2.id - self.graph_init.vm_index_start)
            else:
                unlink_src.append(link.node1.id - 1)
                unlink_dest.append(link.node2.id - self.graph_init.vm_index_start)

        deployed_src = [j for j in range(size_components) for i in range(size_vms)]
        deployed_dest = [j for i in range(size_components) for j in range(size_vms)]
        print(link_src)
        print(link_dest)
        self.graph = dgl.heterograph({
            ('component', 'conflict', 'component'): (torch.tensor(components_src), torch.tensor(components_dest)), # component linkedCC
            ('component', 'linked', 'vm'): (torch.tensor(link_src), torch.tensor(link_dest)), # component linkedCM
            ('component', 'unlinked', 'vm'): (torch.tensor(unlink_src), torch.tensor(unlink_dest))  # component unlinkedCM
        })
        features_components = [x.features for x in self.graph_init.nodes if x.get_type() == "component"]
        features_vms = [x.features for x in self.graph_init.nodes if x.get_type() == "vm"]

        labels = [1 for x in self.graph_init.nodes]
        edge_features_conflicts = [x.features for x in self.graph_init.edges]

        self.graph.nodes['component'].data['feat'] = torch.from_numpy(np.array(features_components))
        self.graph.nodes['vm'].data['feat'] = torch.from_numpy(np.array(features_vms))

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1