import os
import json

from graph import Node, Graph
from dgl_graph import DGLGraph


def read_jsons(path_to_dir):
    all_json_data = []
    for file_name in os.listdir(path_to_dir):
        file_path = f'{path_to_dir}/{file_name}'
        with open(file_path, 'r') as json_file:
            json_data = json.load(json_file)
            json_data['filename'] = file_path
            all_json_data.append(json_data)
    return all_json_data


def without_keys(d, keys):
    return {k: v for k, v in d.items() if k not in keys}


def get_node_features(component, restrictions):
    cpu = component["Compute"]["CPU"]
    memory = component["Compute"]["Memory"]
    storage = component["Storage"]["StorageSize"]
    full_deploy = 0
    upper_b = 0
    lower_b = 0
    eq_b = 0

    node_id = component['id']
    for res in restrictions:
        if res["type"] == "FullDeployment" and res["alphaCompId"] == node_id:
            full_deploy = 1
        if res["type"] == "UpperBound" and len(res["compsIdList"]) == 1 and node_id in res["compsIdList"]:
            upper_b = res["bound"]
        if res["type"] == "LowerBound" and len(res["compsIdList"]) == 1 and node_id in res["compsIdList"]:
            lower_b = res["bound"]
        if res["type"] == "EqualBound" and len(res["compsIdList"]) == 1 and node_id in res["compsIdList"]:
            eq_b = res["bound"]

    return [cpu, memory, storage, full_deploy, upper_b, lower_b, eq_b]


def get_component_nodes(json_data, restrictions):
    component_nodes = []
    for component in json_data['components']:
        features = get_node_features(component, restrictions)
        component_node = Node(component['id'], features)
        component_nodes.append(component_node)
    return component_nodes


def get_vm_nodes(json_data, starting_index):
    vm_nodes = []
    vm_features = [0, 0, 0, 0, 0, 0, 0]
    for idx, vm_type in enumerate(json_data['output'][3]['types_of_VMs']):
        vm_nodes.append(Node(starting_index + idx, vm_features))
    return vm_nodes


def get_graph_data(json_data, file_name):
    restrictions = json_data["restrictions"]
    component_nodes = get_component_nodes(json_data, restrictions)
    vm_nodes = get_vm_nodes(json_data, len(component_nodes) + 1)
    return Graph(file_name, component_nodes, vm_nodes, restrictions)


if __name__ == '__main__':
    data = read_jsons('secureWebContainerJSONS')
    graphs = []
    for json_graph_data in data:
        filename = json_graph_data['filename']
        graphs.append(get_graph_data(json_graph_data, filename))

    for graph in graphs:
        print(graph)
        dataset = DGLGraph(graph)
        dgl_graph = dataset[0]
        print(dgl_graph)
