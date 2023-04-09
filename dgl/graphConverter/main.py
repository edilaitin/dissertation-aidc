import os
import json
import dgl
import torch

import numpy as np
from graph import Node, Graph
from dgl_graph import DGLGraph,print_dataset
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
import matplotlib.pyplot as plt


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


def get_node_features(component, restrictions, max_cpu, max_mem, max_storage):
    # normalize values
    cpu = component["Compute"]["CPU"] / max_cpu
    memory = component["Compute"]["Memory"] / max_mem
    storage = component["Storage"]["StorageSize"] / max_storage
    full_deploy = 0
    upper_b = 0
    lower_b = 0
    eq_b = 0

    node_id = component['id']
    for res in restrictions:
        if res["type"] == "FullDeployment" and res["alphaCompId"] == node_id:
            full_deploy = 1
        if res["type"] == "UpperBound" and len(res["compsIdList"]) == 1 and node_id in res["compsIdList"]:
            upper_b = 1
        if res["type"] == "LowerBound" and len(res["compsIdList"]) == 1 and node_id in res["compsIdList"]:
            lower_b = 1
        if res["type"] == "EqualBound" and len(res["compsIdList"]) == 1 and node_id in res["compsIdList"]:
            eq_b = 1

    return [cpu, memory, storage, full_deploy, upper_b, lower_b, eq_b]


def get_component_nodes(json_data, restrictions, max_cpu, max_mem, max_storage):
    component_nodes = []
    for component in json_data['components']:
        features = get_node_features(component, restrictions, max_cpu, max_mem, max_storage)
        component_node = Node(component['id'], features, "component")
        component_nodes.append(component_node)
    return component_nodes


def get_vm_nodes(json_data, starting_index, max_cpu, max_mem, max_storage, max_price, surrogate_result):
    vm_nodes = []
    idx = 0
    for vm_type in json_data['output']['offers'].keys():
        vm_specs = json_data['output']['offers'][vm_type]
        vm_features = [
            vm_specs["cpu"] / max_cpu,
            vm_specs["memory"] / max_mem,
            vm_specs["storage"] / max_storage,
            vm_specs["price"] / max_price
        ]
        for i in range(surrogate_result):
            vm_nodes.append(Node(starting_index + idx, vm_features, "vm"))
            idx = idx + 1
    return vm_nodes


def get_graph_data(json_data, file_name):
    restrictions = json_data["restrictions"]
    assign = json_data["output"]["assign_matr"]
    # Determine max of each cpu/memory/storage/price for normalization in [0, 1]
    max_cpu, max_mem, max_storage, max_price = 0, 0, 0, 0
    for component in json_data['components']:
        cpu = component["Compute"]["CPU"]
        memory = component["Compute"]["Memory"]
        storage = component["Storage"]["StorageSize"]
        if cpu > max_cpu: max_cpu = cpu
        if memory > max_mem: max_mem = memory
        if storage > max_storage: max_storage = storage
    for vm_type in json_data['output']['types_of_VMs']:
        vm_specs = [vm for vm in json_data['output']['VMs specs'] if list(vm.values())[0]['id'] == vm_type][0]
        cpu = list(vm_specs.values())[0]["cpu"]
        memory = list(vm_specs.values())[0]["memory"]
        storage = list(vm_specs.values())[0]["storage"]
        price = list(vm_specs.values())[0]["price"]
        if cpu > max_cpu: max_cpu = cpu
        if memory > max_mem: max_mem = memory
        if storage > max_storage: max_storage = storage
        if price > max_price: max_price = price

    surrogate_result = 6
    component_nodes = get_component_nodes(json_data, restrictions, max_cpu, max_mem, max_storage)
    vm_nodes = get_vm_nodes(json_data, len(component_nodes) + 1, max_cpu, max_mem, max_storage, max_price, surrogate_result)
    return Graph(file_name, component_nodes, vm_nodes, restrictions, assign, json_data["output"], surrogate_result)


class HeteroMLPPredictor(nn.Module):
    def __init__(self, in_dims, n_classes):
        super().__init__()
        self.W = nn.Linear(in_dims * 2, n_classes)

    def apply_edges(self, edges):
        x = torch.cat([edges.src['h'], edges.dst['h']], 1)
        y = self.W(x)
        return {'score': y}

    def forward(self, graph, h):
        # h contains the node representations for each edge type computed from
        # the GNN for heterogeneous graphs defined in the node classification
        # section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h  # assigns 'h' of all node types in one shot
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']


class Model(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, rel_names):
        super().__init__()
        self.sage = RGCN(in_features, hidden_features, out_features, rel_names)
        self.pred = HeteroMLPPredictor(out_features, len(rel_names))

    def forward(self, g, x, dec_graph):
        h = self.sage(g, x)
        return self.pred(dec_graph, h)


class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()
        # INCREASE LAYERS
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        return h

def to_assignment_matrix(graph, dec_graph, tensor, components_nr):
    vms_nr = int(len(tensor) / components_nr)
    assign_matrix = [[0 for _ in range(vms_nr)] for _ in range(components_nr)]

    for dec_ind in range(len(tensor)):
        type = dec_graph.edata[dgl.ETYPE][dec_ind].item()
        orig_index = dec_graph.edata[dgl.EID][dec_ind].item()
        type_edge = graph.etypes[type]
        component = graph.edges(form='uv', order='srcdst', etype=type_edge)[0][orig_index].item()
        vm = graph.edges(form='uv', order='srcdst', etype=type_edge)[1][orig_index].item()
        value = tensor[dec_ind].item()
        if value == 2:
            assign_matrix[component][vm] = 0
        else:
            assign_matrix[component][vm] = 1
    return assign_matrix


if __name__ == '__main__':
    data = read_jsons('secureWebContainers_DO')
    graphs = []
    for json_graph_data in data:
        filename = json_graph_data['filename']
        graphs.append(get_graph_data(json_graph_data, filename))

    dgl_graphs = []
    for graph in graphs[:10000]:
        # print('\n\nGraph Nodes AND Edges')
        # print(graph)
        dataset = DGLGraph(graph)
        dgl_graph = dataset[0]
        # print_dataset(dgl_graph)
        dgl_graphs.append(dgl_graph)

    arr = np.array(dgl_graphs)
    # Calculate the sizes of the three parts
    n = len(arr)
    size1 = int(0.6 * n)
    size2 = int(0.2 * n)

    # Split the array into three parts
    train = arr[:size1].tolist()
    validation = arr[size1:size1 + size2].tolist()
    test = arr[size1 + size2:].tolist()

    model = Model(7, 10, 5, ['conflict', 'linked', 'unlinked'])
    opt = torch.optim.Adam(model.parameters())
    loss_list = []
    loss_list_valid = []

    acc_training_list = []
    acc_validation_list = []

    epochs = 50
    for epoch in range(epochs):
        ###########################################################################################################################################################
        ######################################################################## TRAINING #########################################################################
        ###########################################################################################################################################################
        # set the model to train mode
        model.train()
        # create empty lists to store the predictions and true labels
        y_pred = []
        y_true = []

        total_logits = None
        total_labels = None
        for train_graph in train:
            dec_graph = train_graph['component', :, 'vm']

            edge_label = dec_graph.edata[dgl.ETYPE]
            comp_feats = train_graph.nodes['component'].data['feat']
            vm_feats =  train_graph.nodes['vm'].data['feat']
            node_features = {'component': comp_feats, 'vm': vm_feats}

            logits = model(train_graph, node_features, dec_graph)
            if total_logits == None:
                total_logits = logits
            else:
                total_logits = torch.cat((total_logits, logits))
            if total_labels == None:
                total_labels = edge_label
            else:
                total_labels = torch.cat((total_labels, edge_label))
            y_pred.append(logits.argmax(dim=-1))
            y_true.append(edge_label)

        # concatenate the predictions and true labels into tensors
        y_pred = torch.cat(y_pred)
        y_true = torch.cat(y_true)

        # compute the accuracy of the model on the training set
        accuracy = (y_pred == y_true).float().mean().item()
        acc_training_list.append(accuracy)
        print("Training accuracy:", accuracy)

        ###########################################################################################################################################################
        ####################################################################### VALIDATION ########################################################################
        ###########################################################################################################################################################

        # create empty lists to store the predictions and true labels
        y_pred = []
        y_true = []

        # set the model to evaluation mode
        # model.eval()
        avg_loss = []

        # loop over the validation graphs and compute the predictions and true labels
        for validation_graph in validation:
            dec_graph = validation_graph['component', :, 'vm']

            edge_label = dec_graph.edata[dgl.ETYPE]
            comp_feats = validation_graph.nodes['component'].data['feat']
            vm_feats = validation_graph.nodes['vm'].data['feat']
            node_features = {'component': comp_feats, 'vm': vm_feats}
            with torch.no_grad():
                logits = model(validation_graph, node_features, dec_graph)
                loss = F.cross_entropy(logits, edge_label)
                avg_loss.append(loss.item())

            y_pred.append(logits.argmax(dim=-1))
            y_true.append(edge_label)

        loss_avg = sum(avg_loss)/len(avg_loss)
        loss_list_valid.append(loss_avg)

        # concatenate the predictions and true labels into tensors
        y_pred = torch.cat(y_pred)
        y_true = torch.cat(y_true)

        # compute the accuracy of the model on the validation set
        accuracy = (y_pred == y_true).float().mean().item()
        acc_validation_list.append(accuracy)
        print("Validation accuracy:", accuracy)

        loss = F.cross_entropy(total_logits, total_labels)
        loss_list.append(loss.item())
        opt.zero_grad()
        loss.backward()
        opt.step()

    print(loss_list)
    print(loss_list_valid)

    plt.plot(range(epochs), loss_list, label='Loss Train')
    plt.plot(range(epochs), loss_list_valid, label='Loss Valid')
    # plt.plot(range(epochs), acc_list, label='Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

    # plt.plot(range(epochs), loss_list, label='Loss')
    plt.plot(range(epochs), acc_training_list, label='Training Accuracy')
    plt.plot(range(epochs), acc_validation_list, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    ###########################################################################################################################################################
    ######################################################################### TESTING #########################################################################
    ###########################################################################################################################################################
    # create empty lists to store the predictions and true labels
    y_pred = []
    y_true = []

    # set the model to evaluation mode
    model.eval()

    # loop over the test graphs and compute the predictions and true labels
    for test_graph in test:
        dec_graph = test_graph['component', :, 'vm']
        print(dec_graph)

        edge_label = dec_graph.edata[dgl.ETYPE]
        comp_feats = test_graph.nodes['component'].data['feat']
        vm_feats = test_graph.nodes['vm'].data['feat']
        node_features = {'component': comp_feats, 'vm': vm_feats}
        with torch.no_grad():
            logits = model(test_graph, node_features, dec_graph)
        pred = logits.argmax(dim=-1)
        y_pred.append(pred)
        print(f"Prediction {to_assignment_matrix(test_graph, dec_graph, pred, 5)}")
        y_true.append(edge_label)
        print(f"Actual {to_assignment_matrix(test_graph, dec_graph, edge_label, 5)}")

    # concatenate the predictions and true labels into tensors
    y_pred = torch.cat(y_pred)
    y_true = torch.cat(y_true)

    # compute the accuracy of the model on the validation set
    accuracy = (y_pred == y_true).float().mean().item()
    acc_validation_list.append(accuracy)
    print("Testing accuracy:", accuracy)


