import os
import json
import dgl
import torch

import numpy as np
from graph import Node, Graph
from dgl_graph import DGLGraph,print_dataset
from ohe import edge_constraints_encoding
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
import matplotlib.pyplot as plt
import dgl.function as fn


def read_jsons(path_to_dir):
    all_json_data = []
    index = 0
    for file_name in os.listdir(path_to_dir):
        index = index + 1
        print(f"DURING DIR READ {index}")
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

class GNNLayer(nn.Module):
    def __init__(self, ndim_in, edims, ndim_out, activation):
        super(GNNLayer, self).__init__()
        self.W_msg = nn.Linear(ndim_in + edims, ndim_out)
        self.W_apply = nn.Linear(ndim_in + ndim_out, ndim_out)
        self.activation = activation
    def message_func(self, edges):
        return {'m': F.relu(self.W_msg(torch.cat([edges.src['h'], edges.data['h']], 1)))}
    def forward(self, g_dgl, nfeats, efeats):
        with g_dgl.local_scope():
            g = g_dgl
            g.ndata['h'] = nfeats
            g.edata['h'] = efeats
            g.update_all(self.message_func, fn.sum('m', 'h_neigh'))
            g.ndata['h'] = F.relu(self.W_apply(torch.cat([g.ndata['h'], g.ndata['h_neigh']], 1)))
            return g.ndata['h']

class GNNTower(nn.Module):
    def __init__(self, in_feats, edge_dim, hidden_feats, out_feats, num_layers = 5):
        super(GNNTower, self).__init__()
        activation = nn.ReLU()
        self.layers = nn.ModuleList()
        self.layers.append(GNNLayer(in_feats, edge_dim, 50, activation))
        self.layers.append(GNNLayer(50, edge_dim, 25, activation))
        self.layers.append(GNNLayer(25, edge_dim, out_feats, activation))
        dropout = 0.5
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, nfeats, efeats):
        for i, layer in enumerate(self.layers):
            if i != 0:
                nfeats = self.dropout(nfeats)
            nfeats = layer(g, nfeats, efeats)
        return nfeats


class MLPTower(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(MLPTower, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_feats, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, out_feats)
        )

    def forward(self, features):
        h = self.mlp(features)
        return h


class Model(nn.Module):
    def __init__(self, gnn_in_feats, gnn_edge_feats, gnn_hidden_feats, gnn_out_feats,
                 mlp_in_feats, mlp_hidden_feats, mlp_out_feats):
        super(Model, self).__init__()
        self.gnn_tower = GNNTower(gnn_in_feats, gnn_edge_feats, gnn_hidden_feats, gnn_out_feats)
        self.mlp_tower = MLPTower(mlp_in_feats, mlp_hidden_feats, mlp_out_feats)

    def forward(self, g, component_features, component_edges_features, vm_features, edges):
        component_embeddings = self.gnn_tower(g, component_features, component_edges_features)
        component_embeddings = component_embeddings.to('cuda')
        vm_embeddings = self.mlp_tower(vm_features)
        vm_embeddings = vm_embeddings.to('cuda')

        edge_tensors = [e.clone().detach().to('cuda') for e in edges]
        edge_embeddings = torch.cat([component_embeddings[edge_tensors[0]], vm_embeddings[edge_tensors[1]]], dim=-1)
        edge_embeddings = edge_embeddings.to('cuda')
        logits = torch.sigmoid(torch.sum(edge_embeddings, dim=-1))
        # print(logits)
        return logits


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
        if value == 0:
            assign_matrix[component][vm] = 1
        else:
            assign_matrix[component][vm] = 0
    return assign_matrix


if __name__ == '__main__':
    print("BEFORE DIR READ")
    data = read_jsons('old_secureWebContainers_DO')
    print("AFTER DIR READ")

    graphs = []
    index = 0
    for json_graph_data in data[:1000]:
        index = index + 1
        print(f"DURING Graphs construct {index}")
        filename = json_graph_data['filename']
        graphs.append(get_graph_data(json_graph_data, filename))

    dgl_graphs = []
    index = 0

    for graph in graphs[:1000]:
        index = index + 1
        print(f"DURING Graphs dgl convert {index}")
        # print('\n\nGraph Nodes AND Edges')
        # print(graph)
        dataset = DGLGraph(graph)
        dgl_graph = dataset[0]
        dgl_graph = dgl_graph.to('cuda')
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

    model = Model(7, 7, 10, 5, 4, 10, 2)
    model = model.to('cuda')
    opt = torch.optim.Adam(model.parameters())
    loss_list = []
    loss_list_valid = []

    acc_training_list = []
    acc_validation_list = []

    epochs = 1000
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
            comp_feats = train_graph.nodes['component'].data['feat']
            component_edges_features = train_graph['conflict'].edata['feat']
            component_edges_features = component_edges_features.to('cuda')
            dec_graph = train_graph['component', :, 'vm']
            dec_graph = dec_graph.to('cuda')
            comp_feats = comp_feats.to('cuda')
            vm_feats =  train_graph.nodes['vm'].data['feat']
            vm_feats = vm_feats.to('cuda')
            node_features = {'component': comp_feats, 'vm': vm_feats}
            edges = dec_graph['linked+unlinked'].edges()

            positive_edge_ids = dec_graph.filter_edges(lambda edges: edges.data['_TYPE'] == 1)
            negative_edge_ids = dec_graph.filter_edges(lambda edges: edges.data['_TYPE'] == 2)

            testing_edge_ids = positive_edge_ids.clone()

            # Randomly sample 30% of negative_edge_ids
            num_samples = int(0.3 * negative_edge_ids.size(0))
            random_indices = torch.randperm(negative_edge_ids.size(0))[:num_samples]
            sampled_negative = negative_edge_ids[random_indices]

            # Concatenate sampled_negative with positives
            testing_edge_ids = torch.cat((testing_edge_ids, sampled_negative), dim=0)

            # Access the source and destination nodes of the filtered edges
            filtered_edges = dec_graph.find_edges(testing_edge_ids)

            edge_label = dec_graph.edata[dgl.ETYPE][testing_edge_ids]
            edge_label = edge_label.to('cuda')

            graph = train_graph['conflict']
            # print(dec_graph)
            target = edge_label.clone().detach() - 1
            target = target.to('cuda')

            logits = model(graph, comp_feats, component_edges_features, vm_feats, filtered_edges)
            if total_logits == None:
                total_logits = logits
            else:
                total_logits = torch.cat((total_logits, logits))
            # print(logits)
            if total_labels == None:
                total_labels = target
            else:
                total_labels = torch.cat((total_labels, target))
            predicted_class = logits.round().long()
            # print(predicted_class)
            y_pred.append(predicted_class)
            y_true.append(target)

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
            component_edges_features = validation_graph['conflict'].edata['feat']
            component_edges_features = component_edges_features.to('cuda')
            vm_feats = validation_graph.nodes['vm'].data['feat']
            edges = dec_graph['linked+unlinked'].edges()
            graph = validation_graph['conflict']
            target = edge_label.clone().detach() - 1
            target = target.to('cuda')
            with torch.no_grad():
                logits = model(graph, comp_feats, component_edges_features, vm_feats, edges)
                criterion = nn.BCEWithLogitsLoss()
                loss = criterion(logits, target.float())
                avg_loss.append(loss.item())

            predicted_class = logits.round().long()
            y_pred.append(predicted_class)
            y_true.append(target)

        loss_avg = sum(avg_loss)/len(avg_loss)
        loss_list_valid.append(loss_avg)

        # concatenate the predictions and true labels into tensors
        y_pred = torch.cat(y_pred)
        y_true = torch.cat(y_true)

        # compute the accuracy of the model on the validation set
        accuracy = (y_pred == y_true).float().mean().item()
        acc_validation_list.append(accuracy)
        print("Validation accuracy:", accuracy)

        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(total_logits, total_labels.float())
        print(loss.item())
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
        edge_label = dec_graph.edata[dgl.ETYPE]
        comp_feats = test_graph.nodes['component'].data['feat']
        component_edges_features = test_graph['conflict'].edata['feat']
        component_edges_features = component_edges_features.to('cuda')
        vm_feats = test_graph.nodes['vm'].data['feat']
        edges = dec_graph['linked+unlinked'].edges()
        graph = test_graph['conflict']
        target = edge_label.clone().detach() - 1
        target = target.to('cuda')

        with torch.no_grad():
            logits = model(graph, comp_feats, component_edges_features, vm_feats, edges)

        pred = logits.round().long()
        y_pred.append(pred)
        print(f"Prediction {to_assignment_matrix(test_graph, dec_graph, pred, 5)}")
        y_true.append(target)
        print(f"Actual {to_assignment_matrix(test_graph, dec_graph, target, 5)}")

    # concatenate the predictions and true labels into tensors
    y_pred = torch.cat(y_pred)
    y_true = torch.cat(y_true)

    # compute the accuracy of the model on the validation set
    accuracy = (y_pred == y_true).float().mean().item()
    acc_validation_list.append(accuracy)
    print("Testing accuracy:", accuracy)


