import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages
import scipy.io as sio
from scipy.sparse import csc_matrix
from scipy.sparse import lil_matrix
import scipy
from sklearn.metrics import f1_score
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.preprocessing import normalize
from sklearn import metrics
import math
import copy
import random

from flip_gradient import flip_gradient

np.seterr(divide='ignore', invalid='ignore')


def shuffle_aligned_list(data):
    num = data[0].shape[0]
    shuffle_index = np.random.permutation(num)
    return shuffle_index, [d[shuffle_index] for d in data]


def csr_2_sparse_tensor_tuple(csr_matrix):
    if not isinstance(csr_matrix, scipy.sparse.lil_matrix):
        csr_matrix = lil_matrix(csr_matrix)
    coo_matrix = csr_matrix.tocoo()
    indices = np.transpose(np.vstack((coo_matrix.row, coo_matrix.col)))
    values = coo_matrix.data
    shape = csr_matrix.shape
    return indices, values, shape


def nor(A, X, k, gamma):
    A = csc_matrix.toarray(A)
    X = lil_matrix.toarray(X)
    I = np.eye(A.shape[0])
    A_ = A + I
    D_ = np.diag(np.sum(A_, axis=1))
    L_ = D_ - A_
    P = np.linalg.matrix_power((I - gamma * np.matmul(np.linalg.inv(D_), L_)), k)
    return P, np.matmul(P, X)


def construct_graph(A, p, method, topk):
    A = csc_matrix.toarray(A)
    # A_sim = metrics.pairwise_distances(A, A)
    if method == 'heat':
        dist = -0.5 * metrics.pairwise_distances(A) ** 2
        dist = np.exp(dist)
    elif method == 'cos':
        dist = metrics.pairwise.cosine_similarity(A, A)

    start=0
    for i in dist:
        AA = i
        kth = sorted(AA)[-int(topk)]
        AA[AA < kth] = 0
        dist[start] = AA
        start = start + 1
    dist[dist > 0] = 1
    row, col = np.diag_indices_from(dist)
    dist[row, col] = 0
    return csc_matrix(dist)


def A_Zero_list(A):
    A_array = csc_matrix.toarray(A)
    A_num = np.sum(A_array, 1)
    A_zero_list = []
    A_tail_list = []
    A_list = []
    zero_num = 0
    tail_num = 0
    head_num = 0

    for i in range(A_array.shape[0]):
        if A_num[i] == 0:
            zero_num = zero_num + 1
            A_zero_list.append(i)
        elif A_num[i] > 0 and A_num[i] < 9:
            tail_num += 1
            A_tail_list.append(i)
        else:
            head_num += 1
            A_list.append(i)

    return A_zero_list, A_tail_list, A_list, zero_num, tail_num, head_num

def ComputeSim(X, q, k, zero_list, tail_list):
    '''Construct adjacency matrix based on attribute similarity'''
    if q == True:
        X = lil_matrix.toarray(X)
    else:
        X = csc_matrix.toarray(X)
    X_sim = metrics.pairwise.cosine_similarity(X, X)
    start = 0
    k_num = 0
    for i in X_sim:
        AA = i
        if q == False:
            if start in zero_list:
                k_num = 0
            elif start in tail_list:
                k_num = math.ceil(k/2)
            else:
                k_num = k
        else:
            k_num = k
        kth = sorted(AA)[-int(k_num)]
        AA[AA <= kth] = 0
        X_sim[start] = AA
        start = start + 1
    X_sim[X_sim > 0] = 1
    row, col = np.diag_indices_from(X_sim)
    X_sim[row, col] = 0
    return X_sim

def ComputeCosSim(X, p, q, k):
    '''Construct adjacency matrix based on attribute similarity'''
    if q == True:
        X = lil_matrix.toarray(X)
        # X[X > 0] = 1
    else:
        X = csc_matrix.toarray(X)
    X_sim = metrics.pairwise.cosine_similarity(X, X)
    start = 0
    if p == 1:
        n = X.shape[0] * 0.01 * k
        n = math.ceil(n)
    elif p == 2:
        n = k
    elif p == 3:
        n = math.pow(X.shape[0], 1 / 2)
        n = math.ceil(n)
    else:
        n = math.pow(X.shape[0], 1 / 3)
        n = 2 * math.ceil(n)
    for i in X_sim:
        AA = i
        kth = sorted(AA)[-int(n)]
        AA[AA < kth] = 0
        X_sim[start] = AA
        start = start + 1
    # for ii in range(X_sim.shape[0]):
    #     for jj in range(X_sim.shape[1]):
    #         if X_sim[ii][jj] == X_sim[jj][ii] and X_sim[ii][jj] != 0 and X_sim[jj][ii] != 0:
    #             X_sim[ii][jj] = 1
    #             X_sim[jj][ii] = 1
    #         else:
    #             X_sim[ii][jj] = 0
    #             X_sim[jj][ii] = 0
    X_sim[X_sim > 0] = 1
    row, col = np.diag_indices_from(X_sim)
    X_sim[row, col] = 0
    return X_sim


def batch_generator(data, batch_size, shuffle=True):
    if shuffle:
        shuffle_index, data = shuffle_aligned_list(data)
    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= data[0].shape[0]:
            batch_count = 0
            if shuffle:
                shuffle_index, data = shuffle_aligned_list(data)
        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [d[start:end] for d in data], shuffle_index[start:end]


def fc_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu, input_type='dense', drop=0.0):
    with tf.name_scope(layer_name):
        weight = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=1. / tf.sqrt(input_dim / 2.)),
                             name='weight')
        bias = tf.Variable(tf.constant(0.1, shape=[output_dim]), name='bias')
        if input_type == 'sparse':
            activations = act(tf.sparse_tensor_dense_matmul(input_tensor, weight) + bias)
        else:
            activations = act(tf.matmul(input_tensor, weight) + bias)

        activations = tf.nn.dropout(activations, rate=drop)

        return activations


def load_network(file):
    net = sio.loadmat(file)
    X, A, Y = net['attrb'], net['network'], net['group']
    if not isinstance(X, scipy.sparse.lil_matrix):
        X = lil_matrix(X)

    return A, X, Y


def MyScaleSimMat(W):
    '''L1 row norm of a matrix'''
    rowsum = np.array(np.sum(W, axis=1), dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    W = r_mat_inv.dot(W)
    return W


def AggTranProbMat(G, step):
    '''aggregated K-step transition probality'''
    G = MyScaleSimMat(G)
    G = csc_matrix.toarray(G)
    A_k = G
    A = G
    for k in np.arange(2, step + 1):
        A_k = np.matmul(A_k, G)
        A = A + A_k / k

    return A


def random_surf(G, num_hops, alpha):
    num_nodes = G.shape[0]
    G = MyScaleSimMat(G)
    G = csc_matrix.toarray(G)
    p0 = np.eye(num_nodes, dtype='float32')
    p = p0
    A = np.zeros((num_nodes, num_nodes), dtype='float32')
    for i in range(num_hops):
        p = (alpha * np.dot(p, G)) + ((1 - alpha) * p0)
        A = A + p
    return A


def ComputePPMI(A):
    '''compute PPMI, given aggregated K-step transition probality matrix as input'''
    np.fill_diagonal(A, 0)
    A = MyScaleSimMat(A)
    (p, q) = np.shape(A)
    col = np.sum(A, axis=0)
    col[col == 0] = 1
    PPMI = np.log((float(p) * A) / col[None, :])
    IdxNan = np.isnan(PPMI)
    PPMI[IdxNan] = 0
    PPMI[PPMI < 0] = 0

    return PPMI


def batchPPMI(batch_size, shuffle_index_s, shuffle_index_t, PPMI_s, PPMI_t):
    '''return the PPMI matrix between nodes in each batch'''

    ##proximity matrix between source network nodes in each mini-batch
    a_s = np.zeros((int(batch_size / 2), int(batch_size / 2)))
    for ii in range(int(batch_size / 2)):
        for jj in range(int(batch_size / 2)):
            if ii != jj:
                a_s[ii, jj] = PPMI_s[shuffle_index_s[ii], shuffle_index_s[jj]]

    ##proximity matrix between target network nodes in each mini-batch
    a_t = np.zeros((int(batch_size / 2), int(batch_size / 2)))
    for ii in range(int(batch_size / 2)):
        for jj in range(int(batch_size / 2)):
            if ii != jj:
                a_t[ii, jj] = PPMI_t[shuffle_index_t[ii], shuffle_index_t[jj]]

    return csr_2_sparse_tensor_tuple(MyScaleSimMat(a_s)), csr_2_sparse_tensor_tuple(MyScaleSimMat(a_t))


def feature_compression(features, dim=200):
    """Preprcessing of features"""
    features = features.toarray()
    feat = lil_matrix(PCA(n_components=dim, random_state=0).fit_transform(features))
    return feat


def feature_compression_sec(features, dim=200):
    """Preprcessing of features"""
    # features = features.toarray()
    feat = PCA(n_components=dim, random_state=0).fit_transform(features)
    return feat


def merge(A, B):
    A = np.array(csc_matrix.toarray(A), dtype=np.int32)
    B = np.array(csc_matrix.toarray(B), dtype=np.int32)
    C = A | B
    return csc_matrix(C)


def f1_scores(y_pred, y_true):
    def predict(y_true, y_pred):
        top_k_list = np.array(np.sum(y_true, 1), np.int32)
        predictions = []
        for i in range(y_true.shape[0]):
            pred_i = np.zeros(y_true.shape[1])
            pred_i[np.argsort(y_pred[i, :])[-top_k_list[i]:]] = 1
            predictions.append(np.reshape(pred_i, (1, -1)))
        predictions = np.concatenate(predictions, axis=0)

        return np.array(predictions, np.int32)

    results = {}
    predictions = predict(y_true, y_pred)
    averages = ["micro", "macro"]
    for average in averages:
        results[average] = f1_score(y_true, predictions, average=average)

    return results["micro"], results["macro"]


def f1_scores_1(y_pred, y_true, list_zero, list_head, list_tail):
    def predict(y_true, y_pred):
        top_k_list = np.array(np.sum(y_true, 1), np.int32)
        predictions = []
        for i in range(y_true.shape[0]):
            pred_i = np.zeros(y_true.shape[1])
            pred_i[np.argsort(y_pred[i, :])[-top_k_list[i]:]] = 1
            predictions.append(np.reshape(pred_i, (1, -1)))
        predictions = np.concatenate(predictions, axis=0)

        return np.array(predictions, np.int32)

    results = {}
    results_zero = {}
    results_head = {}
    results_tail = {}

    predictions = predict(y_true, y_pred)
    predictions_zero = predictions[list_zero, :]
    predictions_head = predictions[list_head, :]
    predictions_tail = predictions[list_tail, :]

    averages = ["micro", "macro"]
    averages_zero = ["micro", "macro"]
    averages_head = ["micro", "macro"]
    averages_tail = ["micro", "macro"]

    y_true_zero = y_true[list_zero, :]
    y_true_head = y_true[list_head, :]
    y_true_tail = y_true[list_tail, :]

    for average in averages:
        results[average] = f1_score(y_true, predictions, average=average)
    for average_zero in averages_zero:
        results_zero[average_zero] = f1_score(y_true_zero, predictions_zero, average=average_zero)
    for average_head in averages_head:
        results_head[average_head] = f1_score(y_true_head, predictions_head, average=average_head)
    for average_tail in averages_tail:
        results_tail[average_tail] = f1_score(y_true_tail, predictions_tail, average=average_tail)
    return results["micro"], results["macro"], results_zero["micro"], results_zero["macro"], results_head["micro"], \
           results_head["macro"], results_tail["micro"], results_tail["macro"]


def f1_scores_wo_zero(y_pred, y_true, list_head, list_tail):
    def predict(y_true, y_pred):
        top_k_list = np.array(np.sum(y_true, 1), np.int32)
        predictions = []
        for i in range(y_true.shape[0]):
            pred_i = np.zeros(y_true.shape[1])
            pred_i[np.argsort(y_pred[i, :])[-top_k_list[i]:]] = 1
            predictions.append(np.reshape(pred_i, (1, -1)))
        predictions = np.concatenate(predictions, axis=0)

        return np.array(predictions, np.int32)

    results = {}
    # results_zero = {}
    results_head = {}
    results_tail = {}

    predictions = predict(y_true, y_pred)
    # predictions_zero = predictions[list_zero, :]
    predictions_head = predictions[list_head, :]
    predictions_tail = predictions[list_tail, :]

    averages = ["micro", "macro"]
    # averages_zero = ["micro", "macro"]
    averages_head = ["micro", "macro"]
    averages_tail = ["micro", "macro"]

    # y_true_zero = y_true[list_zero, :]
    y_true_head = y_true[list_head, :]
    y_true_tail = y_true[list_tail, :]

    for average in averages:
        results[average] = f1_score(y_true, predictions, average=average)
    # for average_zero in averages_zero:
    #     results_zero[average_zero] = f1_score(y_true_zero, predictions_zero, average=average_zero)
    for average_head in averages_head:
        results_head[average_head] = f1_score(y_true_head, predictions_head, average=average_head)
    for average_tail in averages_tail:
        results_tail[average_tail] = f1_score(y_true_tail, predictions_tail, average=average_tail)
    return results["micro"], results["macro"], results_head["micro"], results_head["macro"], results_tail["micro"], \
           results_tail["macro"]


def domain_Dis(emb, Ada_lambda, n_emb, d_label):
    h_grl = flip_gradient(emb, Ada_lambda)
    h_dann_1 = fc_layer(h_grl, n_emb, n_emb, layer_name='dann_fc_1')
    h_dann_2 = fc_layer(h_dann_1, n_emb, n_emb, layer_name='dann_fc_2')
    W_domain = tf.Variable(tf.truncated_normal([n_emb, 2], stddev=1. / tf.sqrt(n_emb / 2.)), name='dann_weight')
    b_domain = tf.Variable(tf.constant(0.1, shape=[2]), name='dann_bias')
    d_logit_global = tf.matmul(h_dann_2, W_domain) + b_domain
    domain_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=d_logit_global, labels=d_label))
    return domain_loss


def loss_weight(loss1, loss2, loss3, loss4):
    best_loss_1 = 1.0
    loss1 = tf.cond(loss1 > 0.5, lambda: 1 - loss1, lambda: loss1)
    best_loss_1 = tf.cond(best_loss_1 > loss1, lambda: loss1, lambda: best_loss_1)
    error_1 = 2 * (1 - 2 * best_loss_1)

    best_loss_2 = 1.0
    loss2 = tf.cond(loss2 > 0.5, lambda: 1 - loss2, lambda: loss2)
    best_loss_2 = tf.cond(best_loss_2 > loss2, lambda: loss2, lambda: best_loss_2)
    error_2 = 2 * (1 - 2 * best_loss_2)

    best_loss_3 = 1.0
    loss3 = tf.cond(loss3 > 0.5, lambda: 1 - loss3, lambda: loss3)
    best_loss_3 = tf.cond(best_loss_3 > loss3, lambda: loss3, lambda: best_loss_3)
    error_3 = 2 * (1 - 2 * best_loss_3)

    best_loss_4 = 1.0
    loss4 = tf.cond(loss4 > 0.5, lambda: 1 - loss4, lambda: loss4)
    best_loss_4 = tf.cond(best_loss_4 > loss4, lambda: loss4, lambda: best_loss_4)
    error_4 = 2 * (1 - 2 * best_loss_4)
    loss = error_1 + error_2 + error_3 + error_4
    return error_1 / loss, error_2 / loss, error_3 / loss, error_4 / loss

def aug_random_mask(input_feature, drop_percent=0.2):
    input_feature = input_feature.toarray()
    node_num = input_feature.shape[1]
    mask_num = int(node_num * drop_percent)
    node_idx = [i for i in range(node_num)]
    aug_feature = copy.deepcopy(input_feature)
    for i in range(input_feature.shape[0]):
        mask_idx = random.sample(node_idx, mask_num)
    # zeros = np.zeros_like(aug_feature[0][0])
        for j in mask_idx:
            aa = np.random.choice([1.0, 2.0, 3.0])
            aug_feature[i][j] = aa
    return lil_matrix(aug_feature)