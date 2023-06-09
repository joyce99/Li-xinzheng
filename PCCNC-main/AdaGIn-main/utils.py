import os
import torch
import numpy as np
import networkx as nx
import scipy.io as sio
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn import metrics
from models import *
from layers import aggregator_lookup
from sklearn.decomposition import PCA
import scipy
from scipy.sparse import csc_matrix, lil_matrix
import copy
import random



def top_k_preds(y_true, y_pred):
    top_k_list = np.array(np.sum(y_true, 1), np.int32)
    predictions = []
    for i in range(y_true.shape[0]):
        pred_i = np.zeros(y_true.shape[1])
        pred_i[np.argsort(y_pred[i, :])[-top_k_list[i]:]] = 1
        predictions.append(np.reshape(pred_i, (1, -1)))
    predictions = np.concatenate(predictions, axis=0)
    top_k_array = np.array(predictions, np.int64)

    return top_k_array


def cal_f1_score(y_true, y_pred):
    micro_f1 = metrics.f1_score(y_true, y_pred, average='micro')
    macro_f1 = metrics.f1_score(y_true, y_pred, average='macro')

    return micro_f1, macro_f1


def batch_generator(nodes, batch_size, shuffle=True):
    num = nodes.shape[0]
    chunk = num // batch_size
    while True:
        if chunk * batch_size + batch_size > num:
            chunk = 0   
            if shuffle:
                idx = np.random.permutation(num)
        b_nodes = nodes[idx[chunk*batch_size:(chunk+1)*batch_size]]
        chunk += 1

        yield b_nodes


def eval_iterate(nodes, batch_size, shuffle=False):
    idx = np.arange(nodes.shape[0])
    if shuffle:
        idx = np.random.permutation(idx)
    n_chunk = idx.shape[0] // batch_size + 1
    for chunk_id, chunk in enumerate(np.array_split(idx, n_chunk)):
        b_nodes = nodes[chunk]

        yield b_nodes


def do_iter(emb_model, cly_model, adj, feature, labels, idx, cal_f1=False, is_social_net=False):
    embs = emb_model(idx, adj, feature)
    preds = cly_model(embs)
    if is_social_net:
        labels_idx = torch.argmax(labels[idx], dim=1)
        cly_loss = F.cross_entropy(preds, labels_idx)   
    else:
        cly_loss = F.multilabel_soft_margin_loss(preds, labels[idx])
    if not cal_f1:
        return embs, cly_loss
    else:
        targets = labels[idx].cpu().numpy()
        preds = top_k_preds(targets, preds.detach().cpu().numpy())
        return embs, cly_loss, preds, targets

def do_iter_1(emb_model, cly_model, adj, feature, labels, idx, cal_f1=False, is_social_net=False):
    embs, pseudo_mlp_s, mlp_dis = emb_model(feature[idx])
    # embs = emb_model(idx, adj, feature)
    preds = cly_model(embs)
    if is_social_net:
        labels_idx = torch.argmax(labels[idx], dim=1)
        cly_loss = F.cross_entropy(preds, labels_idx)
    else:
        cly_loss = F.multilabel_soft_margin_loss(preds, labels[idx])
    if not cal_f1:
        return embs, cly_loss, mlp_dis
    else:
        targets = labels[idx].cpu().numpy()
        preds = top_k_preds(targets, preds.detach().cpu().numpy())
        return embs, cly_loss, preds, targets, mlp_dis

def evaluate_1(mlp_model, concat_model, emb_model, cly_model, adj, feature, labels, idx, batch_size, mode='val', is_social_net=False):
    assert mode in ['val', 'test']
    embs, preds, targets = [], [], []
    cly_loss = 0
    for b_nodes in eval_iterate(idx, batch_size):
        embs_per_batch, cly_loss_per_batch, preds_per_batch, targets_per_batch = do_iter(emb_model, cly_model, adj, feature, labels,
                                                                                         b_nodes, cal_f1=True, is_social_net=is_social_net)
        # features_MLP, _, _ = mlp_model(feature[b_nodes])
        # embs_per_batch = concat_model(torch.concat([embs_per_batch, features_MLP], 1))
        embs.append(embs_per_batch.detach().cpu().numpy())
        preds.append(preds_per_batch)
        targets.append(targets_per_batch)
        cly_loss += cly_loss_per_batch.item()

    cly_loss /= len(preds)
    embs_whole = np.vstack(embs)
    targets_whole = np.vstack(targets)
    micro_f1, macro_f1 = cal_f1_score(targets_whole, np.vstack(preds))

    return cly_loss, micro_f1, macro_f1, embs_whole, targets_whole


def evaluate(emb_model, cly_model, adj, feature, labels, idx, batch_size, mode='val', is_social_net=False):
    assert mode in ['val', 'test']
    embs, preds, targets = [], [], []
    cly_loss = 0
    for b_nodes in eval_iterate(idx, batch_size):
        embs_per_batch, cly_loss_per_batch, preds_per_batch, targets_per_batch = do_iter(emb_model, cly_model, adj, feature, labels,
                                                                                         b_nodes, cal_f1=True, is_social_net=is_social_net)
        embs.append(embs_per_batch.detach().cpu().numpy())
        preds.append(preds_per_batch)
        targets.append(targets_per_batch)
        cly_loss += cly_loss_per_batch.item()

    cly_loss /= len(preds)
    embs_whole = np.vstack(embs)
    targets_whole = np.vstack(targets)
    micro_f1, macro_f1 = cal_f1_score(targets_whole, np.vstack(preds))

    return cly_loss, micro_f1, macro_f1, embs_whole, targets_whole


def get_split(labels, seed):
    idx_tot = np.arange(labels.shape[0])
    np.random.seed(seed)
    np.random.shuffle(idx_tot)

    return idx_tot


def make_adjacency(G, max_degree, seed):
    all_nodes = np.sort(np.array(G.nodes()))
    n_nodes = len(all_nodes)
    adj = (np.zeros((n_nodes, max_degree)) + (n_nodes - 1)).astype(int)
    np.random.seed(seed)
    for node in all_nodes:
        neibs = np.array(G[node])
        if len(neibs) == 0:
            neibs = np.array(node).repeat(max_degree)
        elif len(neibs) < max_degree:
            neibs = np.random.choice(neibs, max_degree, replace=True)
        else:
            neibs = np.random.choice(neibs, max_degree, replace=False)
        adj[node, :] = neibs

    return adj


def normalize(mx):
    rowsum = np.array(mx.sum(1), dtype=np.float64)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)

    return mx


def pre_social_net(adj, features, labels):
    features = csc_matrix(features.astype(np.uint8))
    labels = labels.astype(np.int32)

    return adj, features, labels

def load_network(file):
    net = sio.loadmat(file)
    X, A, Y = net['attrb'], net['network'], net['group']
    if not isinstance(X, scipy.sparse.lil_matrix):
        X = lil_matrix(X)

    return A, X, Y
def load_data_1(file_path="./Datasets", dataset='acmv9.mat'):
    # load raw data
    data_path = file_path + '/' + dataset
    data_mat = sio.loadmat(data_path)
    adj = data_mat['network']
    features = data_mat['attrb']
    labels = data_mat['group']

    return labels

def load_data(file_path="./Datasets", dataset='acmv9.mat', device='cpu', seed=123, is_blog=False):
    # load raw data
    data_path = file_path + '/' + dataset
    data_mat = sio.loadmat(data_path)
    adj = data_mat['network']
    features = data_mat['attrb']
    labels = data_mat['group']
    if is_blog:
        adj, features, labels = pre_social_net(adj, features, labels)
    features = normalize(features)

    # aug_adj = aug_random_edge(adj, drop_percent=0.1)  # random drop edges

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj_dense = np.array(adj.todense())
    # aug_adj = aug_adj + aug_adj.T.multiply(aug_adj.T > aug_adj) - aug_adj.multiply(aug_adj.T > aug_adj)
    # aug_adj_dense = np.array(aug_adj.todense())

    edges = np.vstack(np.where(adj_dense)).T
    Graph = nx.from_edgelist(edges)
    adj = make_adjacency(Graph, 128, seed)

    # aug_edges = np.vstack(np.where(aug_adj_dense)).T
    # aug_Graph = nx.from_edgelist(aug_edges)
    # aug_adj = make_adjacency(aug_Graph, 128, seed)
    idx_tot = get_split(labels, seed)
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)
    adj = torch.from_numpy(adj)
    # aug_adj = torch.from_numpy(aug_adj)
    idx_tot = torch.LongTensor(idx_tot)

    return adj.to(device), features.to(device), labels.to(device), idx_tot.to(device)

def kmeans(model, x, device):
    # x = torch.where(torch.isnan(x), torch.full_like(x, 0), x)
    x = x.cpu().detach().numpy()
    model.fit(x)
    cluster = model.cluster_centers_
    cluster2label = model.predict(x)
    centroids = torch.Tensor(cluster).to(device)
    cluster2label = torch.Tensor(cluster2label).to(device)
    centroids = F.normalize(centroids, p=2, dim=1)
    return centroids, cluster2label


def stru_cl_loss(emb1, emb2, tao=0.1):
    emb1 = F.normalize(emb1)
    emb2 = F.normalize(emb2)
    pos_score_user = torch.mul(emb1, emb2).sum(dim=1)
    pos_score_user = torch.exp(pos_score_user / tao)
    ttl_score_user = torch.matmul(emb1, emb2.transpose(0, 1))
    ttl_score_user = torch.exp(ttl_score_user / tao).sum(dim=1)

    stru_loss = -torch.log(pos_score_user / ttl_score_user).sum()
    return stru_loss

def ProtoNCE_loss(node_embedding, centroids, centroids2label, proto_reg=0.01, tao=0.1):
    centroids2label = centroids2label.type(torch.long)
    norm_user_embeddings = F.normalize(node_embedding)
    # norm_user_embeddings = norm_user_embeddings.unsqueeze(1).repeat(1, centroids.size(0), 1)
    centroids2 = centroids[centroids2label]
    pos_score_user = torch.mul(norm_user_embeddings, centroids2).sum(dim=1)
    pos_score_user = torch.exp(pos_score_user / tao)
    ttl_score_user = torch.matmul(norm_user_embeddings, centroids.transpose(0, 1))
    ttl_score_user = torch.exp(ttl_score_user / tao).sum(dim=1)

    proto_nce_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()

    proto_nce_loss = proto_reg * proto_nce_loss_user
    return proto_nce_loss

def get_A_r(adj, r):
    adj_label = adj.to_dense()
    if r == 1:
        adj_label = adj_label
    elif r == 2:
        adj_label = adj_label@adj_label
    elif r == 3:
        adj_label = adj_label@adj_label@adj_label
    elif r == 4:
        adj_label = adj_label@adj_label@adj_label@adj_label
    return adj_label

def Ncontrast(x_dis, adj_label, tau = 1):
    """
    compute the Ncontrast loss
    """
    x_dis = torch.exp( tau * x_dis)
    x_dis_sum = torch.sum(x_dis, 1)
    x_dis_sum_pos = torch.sum(x_dis*adj_label, 1)
    loss = -torch.log(x_dis_sum_pos * (x_dis_sum**(-1))+1e-8).mean()
    return loss

def get_adj_label(adj, index, order=2):
    """
    get a batch of feature & adjacency matrix
    """
    adj_label = get_A_r(adj, order)
    adj_label_batch = adj_label[index,:][:,index]
    return adj_label_batch


def preprocess_citation(adj, features, normalization="FirstOrderGCN"):
    adj_normalizer = fetch_normalization(normalization)

    adj = adj_normalizer(adj)

    features = normalize(features)
    return adj, features

def aug_normalized_adjacency(adj):
   adj = adj + sp.eye(adj.shape[0])
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()



def fetch_normalization(type):
   switcher = {
       'AugNormAdj': aug_normalized_adjacency,  # A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
   }
   func = switcher.get(type, lambda: "Invalid normalization technique.")
   return func

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def aug_random_edge(input_adj, drop_percent=0.2):
    percent = drop_percent / 2
    row_idx, col_idx = input_adj.nonzero()

    index_list = []
    for i in range(len(row_idx)):
        index_list.append((row_idx[i], col_idx[i]))

    single_index_list = []
    for i in list(index_list):
        single_index_list.append(i)
        index_list.remove((i[1], i[0]))

    edge_num = int(len(row_idx) / 2)  # 9228 / 2
    add_drop_num = int(edge_num * percent / 2)
    aug_adj = copy.deepcopy(input_adj.todense().tolist())

    edge_idx = [i for i in range(edge_num)]
    drop_idx = random.sample(edge_idx, add_drop_num)

    for i in drop_idx:
        aug_adj[single_index_list[i][0]][single_index_list[i][1]] = 0
        aug_adj[single_index_list[i][1]][single_index_list[i][0]] = 0

    '''
    above finish drop edges
    '''
    node_num = input_adj.shape[0]
    l = [(i, j) for i in range(node_num) for j in range(i)]
    add_list = random.sample(l, add_drop_num)

    for i in add_list:
        aug_adj[i[0]][i[1]] = 1
        aug_adj[i[1]][i[0]] = 1

    aug_adj = np.matrix(aug_adj)
    aug_adj = sp.csc_matrix(aug_adj)
    return aug_adj
