import json
import os
from copy import deepcopy
from random import choices


class Node:
    def __init__(self, node_id, label):
        self.id = node_id
        self.label = label

    def __str__(self):
        return f'id: {self.id}, label: {self.label}'

    def get_label(self):
        return self.label

    def equal(self, node):
        return self.id == node.id


class Edge:
    def __init__(self, node1, node2):
        self.node1 = node1
        self.node2 = node2

    def __str__(self):
        return f'Edge: {self.node1.id} - {self.node2.id}'

    def as_array(self):
        return [self.node1.id, self.node2.id]

    def equal(self, edge):
        return self.node1.equal(edge.node1) and self.node2.equal(edge.node2)


class Graph:
    def __init__(self, filename, component_nodes, vm_nodes, adjacency_matrix):
        self.filename = filename
        self.nodes = component_nodes + vm_nodes
        self.edges = []
        index_offset = len(component_nodes) + 1
        self.vm_index_start = index_offset
        for i, _ in enumerate(adjacency_matrix):
            for j, _ in enumerate(adjacency_matrix[i]):
                if adjacency_matrix[i][j] == 1:
                    node1 = next((x for x in component_nodes if x.id == i + 1))
                    node2 = next((x for x in vm_nodes if x.id == j + index_offset))
                    self.edges.append(Edge(node1, node2))

    def edit_graph(self):
        ged = 0
        # ADD NODES
        nr_nodes_added_pop = [0, 1, 2]
        weight_nodes = [0.5, 0.35, 0.15]
        add_nodes = choices(nr_nodes_added_pop, weight_nodes)[0]

        last_node = self.nodes[len(self.nodes) - 1]
        for i in range(0, add_nodes):
            ged = ged + 1
            label = self.nodes[choices(range(self.vm_index_start, len(self.nodes)))[0]].get_label()
            self.nodes.append(Node(last_node.id + i + 1, label))
        # ADD EDGES
        nr_edges_added_pop = range(10)
        add_edges = choices(nr_edges_added_pop)[0]
        for i in range(0, add_edges):
            from_node = self.nodes[choices(range(self.vm_index_start))[0]]
            to_node = self.nodes[choices(range(self.vm_index_start, len(self.nodes)))[0]]
            new_edge = Edge(from_node, to_node)
            is_new = True
            for edge in self.edges:
                if edge.equal(new_edge):
                    is_new = False
            if is_new:
                ged = ged + 1
                self.edges.append(new_edge)

        return ged

    def generate_data(self, mode):
        graph2 = deepcopy(self)
        ged = graph2.edit_graph()
        graph_dict = {
            'graph1': [x.as_array() for x in self.edges],
            'graph2': [x.as_array() for x in graph2.edges],
            'labels1': [x.get_label() for x in self.nodes],
            'labels2': [x.get_label() for x in graph2.nodes],
            'ged': ged
        }
        json_object = json.dumps(graph_dict)
        path = f'dataset/{mode}/{self.filename}'
        folder_path = path.rsplit('/', 1)[0]
        path_exist = os.path.exists(folder_path)
        if not path_exist:
            os.makedirs(folder_path)
        with open(path, 'w+') as outfile:
            outfile.write(json_object)

