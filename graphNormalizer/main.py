import os
import json

from graph import Node, Graph


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


def get_component_nodes(json_data):
    component_nodes = []
    for component in json_data['components']:
        component_node = Node(component['id'], json.dumps(without_keys(component, ['id', 'name'])))
        component_nodes.append(component_node)
    return component_nodes


def get_vm_nodes(json_data, starting_index):
    vm_nodes = []
    for idx, vm_type in enumerate(json_data['output'][3]['types_of_VMs']):
        if vm_type == 0:
            vm_nodes.append(Node(starting_index + idx, '0'))
        else:
            vm_price = json_data['output'][4]['prices_of_VMs'][idx]
            vm_specs = next(
                (list(x.values())[0] for x in json_data['output'][5]['VMs specs'] if
                 list(x.values())[0]['price'] == vm_price), 'no spec found'
            )
            vm_nodes.append(Node(starting_index + idx, json.dumps(vm_specs)))
    return vm_nodes


def get_graph_data(json_data, file_name):
    component_nodes = get_component_nodes(json_data)
    vm_nodes = get_vm_nodes(json_data, len(component_nodes) + 1)
    adjacency_matrix = json_data['output'][6]['assign_matr']
    return Graph(file_name, component_nodes, vm_nodes, adjacency_matrix)


if __name__ == '__main__':
    data = read_jsons('secureWebContainerJSONS')
    graphs = []
    for json_graph_data in data:
        filename = json_graph_data['filename']
        graphs.append(get_graph_data(json_graph_data, filename))

    for graph in graphs:
        graph.generate_data(mode='train')
        graph.generate_data(mode='test')
