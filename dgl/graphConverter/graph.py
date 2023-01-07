from ohe import edge_constraints_encoding
import collections.abc


class Node:
    def __init__(self, node_id, features):
        self.id = node_id
        self.features = features

    def __str__(self):
        return f'\n  id: {self.id}, features: {self.features}'

    def __repr__(self):
        return self.__str__()

    def get_label(self):
        return self.features

    def equal(self, node):
        return self.id == node.id


class Edge:
    def __init__(self, node1, node2, features):
        self.node1 = node1
        self.node2 = node2
        self.features = features

    def __str__(self):
        return f'\n  Edge: {self.node1.id} - {self.node2.id}, features: {self.features}'

    def __repr__(self):
        return self.__str__()

    def as_array(self):
        return [self.node1.id, self.node2.id]

    def equal(self, edge):
        return (self.node1.equal(edge.node1) and self.node2.equal(edge.node2)) or \
               (self.node1.equal(edge.node2) and self.node2.equal(edge.node1))

    def add_features(self, features):
        for index, f in enumerate(features):
            if f == 1:
                self.features[index] = 1


class Graph:
    def __init__(self, filename, component_nodes, vm_nodes, restrictions):
        self.filename = filename
        self.nodes = component_nodes + vm_nodes
        self.edges = []
        index_offset = len(component_nodes) + 1
        self.vm_index_start = index_offset

        def add_edge_features(eg):
            existing_edge = next((e for e in self.edges if e.equal(eg)), None)
            if existing_edge is not None:
                existing_edge.add_features(eg.features)
            else:
                self.edges.append(eg),

        for res in restrictions:
            node1 = None
            node2 = None
            if res["type"] in ["UpperBound", "LowerBound", "EqualBound"] and len(res["compsIdList"]) == 2:
                node1 = next((x for x in component_nodes if x.id == res["compsIdList"][0]))
                node2 = next((x for x in component_nodes if x.id == res["compsIdList"][1]))
            if res["type"] == "RequireProvide":
                node1 = next((x for x in component_nodes if x.id == res["alphaCompId"]))
                node2 = next((x for x in component_nodes if x.id == res["betaCompId"]))
            if res["type"] in ["Conflicts", "Collocation", "ExclusiveDeployment"]:
                node1 = next((x for x in component_nodes if x.id == res["alphaCompId"]))
                node2 = []
                for nid in res["compsIdList"]:
                    node2.append(next((x for x in component_nodes if x.id == nid)))

            if node1 is not None and node2 is not None:
                if isinstance(node2, collections.abc.Sequence):
                    for node2s in node2:
                        edge_features = edge_constraints_encoding(res["type"])
                        edge = Edge(node1, node2s, edge_features)
                        add_edge_features(edge)
                else:
                    edge_features = edge_constraints_encoding(res["type"])
                    edge = Edge(node1, node2, edge_features)
                    add_edge_features(edge)

    def __str__(self):
        return f'Nodes: {self.nodes}, \nEdges: {self.edges}'
