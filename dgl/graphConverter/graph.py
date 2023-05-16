from ohe import edge_constraints_encoding
import collections.abc
from collections import defaultdict

class Node:
    def __init__(self, node_id, features, type):
        self.id = node_id
        self.features = features
        self.type = type

    def __str__(self):
        return f'\n  id: {self.id}, type: {self.type}, features: {self.features}'

    def __repr__(self):
        return self.__str__()

    def get_label(self):
        return self.features

    def get_type(self):
        return self.type

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
    def __init__(self, filename, component_nodes, vm_nodes, restrictions, assign_matr, output, surrogate_result):
        self.filename = filename
        self.nodes = component_nodes + vm_nodes
        self.edges = []
        self.links = []
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
            if res["type"] == "OneToManyDependency":
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
        for node in component_nodes:
            edge_features = edge_constraints_encoding("Loop")
            edge = Edge(node, node, edge_features)
            add_edge_features(edge)
        links_exists = defaultdict(dict)
        for comp_idx, comp_links in enumerate(assign_matr):
            type_indexes = [0] * int(len(vm_nodes) / surrogate_result)
            for vm_idx, vm_linked in enumerate(comp_links):
                node1 = next((x for x in component_nodes if x.id == comp_idx + 1))
                type_vm = output['types_of_VMs'][vm_idx] - 1
                vm_type_idx = surrogate_result * type_vm + type_indexes[type_vm]
                type_indexes[type_vm] = type_indexes[type_vm] + 1
                if vm_linked:
                    node2 = next((x for x in vm_nodes if x.id == self.vm_index_start + vm_type_idx))
                    link = Edge(node1, node2, vm_linked)
                    links_exists[node1.id][node2.id] = True
                    self.links.append(link)

        for comp in component_nodes:
            for vm in vm_nodes:
                if not links_exists.get(comp.id).get(vm.id):
                    link = Edge(comp, vm, 0)
                    self.links.append(link)

    def __str__(self):
        return f'Nodes: {self.nodes}, \nEdges: {self.edges}, \nLinks: {self.links}'
