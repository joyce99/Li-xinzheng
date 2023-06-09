# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import utils
from scipy.sparse import vstack
from evalModel import train_and_evaluate
import scipy.io
from scipy.sparse import lil_matrix, csc_matrix
from sklearn import metrics

tf.set_random_seed(0)
np.random.seed(0)

source = 'Blog1'
target = 'Blog2'
emb_filename = str(source) + '_' + str(target)
Kstep = 3
alpha = 0.6
lamb = 50  # 加边个数

####################
# Load source data
####################
A_s, X_s, Y_s = utils.load_network('./input/' + str(source) + '.mat')
num_nodes_S = X_s.shape[0]

####################
# Load target data
####################
A_t, X_t, Y_t = utils.load_network('./input/' + str(target) + '.mat')

num_nodes_T = X_t.shape[0]

'''get index value'''
A_s_list_zero, A_s_list_tail, A_s_list_head, A_s_zero_num, A_s_tail_num, A_s_head_num = utils.A_Zero_list(A_s)
A_t_list_zero, A_t_list_tail, A_t_list_head, A_t_zero_num, A_t_tail_num, A_t_head_num = utils.A_Zero_list(A_t)
print("A_s_zero_num: %d      A_t_zero_num: %d" % (A_s_zero_num, A_t_zero_num))
print("A_s_tail_num: %d      A_t_tail_num: %d" % (A_s_tail_num, A_t_tail_num))
print("A_s_head_num: %d      A_t_head_num: %d" % (A_s_head_num, A_t_head_num))
print("A_s_all_num: %d      A_t_all_num: %d" % (
A_s_zero_num + A_s_tail_num + A_s_head_num, A_t_zero_num + A_t_tail_num + A_t_head_num))

''' Structure Similarity Graph'''
A_X_s_sim = csc_matrix(utils.ComputeCosSim(A_s, p=2, q=False, k=lamb))
A_X_t_sim = csc_matrix(utils.ComputeCosSim(A_t, p=2, q=False, k=lamb))
''' Attribute Similarity Graph'''
X_s_sim = csc_matrix(utils.ComputeCosSim(X_s, p=2, q=True, k=lamb))
X_t_sim = csc_matrix(utils.ComputeCosSim(X_t, p=2, q=True, k=lamb))

features = vstack((X_s, X_t))
features = utils.feature_compression(features, dim=1000)
X_s = features[0:num_nodes_S, :]
X_t = features[-num_nodes_T:, :]

''' 
Merge original and similar graph
'''
X_s_sim = utils.merge(A_s, X_s_sim)
X_t_sim = utils.merge(A_t, X_t_sim)
A_X_s_sim = utils.merge(A_s, A_X_s_sim)
A_X_t_sim = utils.merge(A_t, A_X_t_sim)

WL_s_attr, X_n_s_sim = utils.nor(X_s_sim, X_s, Kstep, alpha)
WL_s_stru, X_n_s_sec_sim = utils.nor(A_X_s_sim, X_s, Kstep, alpha)
WL_t_attr, X_n_t_sim = utils.nor(X_t_sim, X_t, Kstep, alpha)
WL_t_stru, X_n_t_sec_sim = utils.nor(A_X_t_sim, X_t, Kstep, alpha)

##input data
input_data = dict()
input_data['source'] = source
input_data['target'] = target
input_data['PPMI_S_X'] = WL_s_attr
input_data['PPMI_T_X'] = WL_t_attr
input_data['PPMI_S_A'] = WL_s_stru
input_data['PPMI_T_A'] = WL_t_stru
input_data['attrb_sim_S'] = X_n_s_sim
input_data['attrb_sim_T'] = X_n_t_sim
input_data['attrb_sec_sim_S'] = X_n_s_sec_sim
input_data['attrb_sec_sim_T'] = X_n_t_sec_sim
input_data['label_S'] = Y_s
input_data['label_T'] = Y_t
input_data['A_s_list_head'] = A_s_list_head
input_data['A_t_list_head'] = A_t_list_head
input_data['A_s_list_zero'] = A_s_list_zero
input_data['A_t_list_zero'] = A_t_list_zero
input_data['A_s_list_tail'] = A_s_list_tail
input_data['A_t_list_tail'] = A_t_list_tail

###model config
config = dict()
config['clf_type'] = 'multi-class'
config['dropout'] = 0.6
config['num_epoch'] = 100  # maximum training iteration
config['batch_size'] = 100
config['n_hidden'] = [512, 128]  # dimensionality for each hidden layer of FE1 and FE2
config['n_emb'] = 128  # embedding dimension d
config['l2_w'] = 1e-3  # weight of L2-norm regularization
config['net_pro_w'] = 0.0001  # weight of pairwise constraint
config['emb_filename'] = emb_filename  # output file name to save node representations
config['lr_ini'] = 0.01  # initial learning rate  0.01
numRandom = 5
microAllRandom = []
macroAllRandom = []
zero_microAllRandom = []
zero_macroAllRandom = []
head_microAllRandom = []
head_macroAllRandom = []
tail_microAllRandom = []
tail_macroAllRandom = []

print('source and target networks:', str(source), str(target))
for random_state in range(numRandom):
    print("%d-th random initialization" % (random_state + 1))
    # micro_t, macro_t= train_and_evaluate(input_data, config, random_state)
    # micro_t, macro_t, zero_micro_t, zero_macro_t, head_micro_t, head_macro_t, tail_micro_t, tail_macro_t = train_and_evaluate(
    #     input_data,
    #     config,
    #     random_state)

    micro_t, macro_t, head_micro_t, head_macro_t, tail_micro_t, tail_macro_t = train_and_evaluate(
        input_data,
        config,
        random_state)

    microAllRandom.append(micro_t)
    macroAllRandom.append(macro_t)
    # zero_microAllRandom.append(zero_micro_t)
    # zero_macroAllRandom.append(zero_macro_t)
    head_microAllRandom.append(head_micro_t)
    head_macroAllRandom.append(head_macro_t)
    tail_microAllRandom.append(tail_micro_t)
    tail_macroAllRandom.append(tail_macro_t)

'''avg F1 scores over 5 random splits'''
micro = np.mean(microAllRandom)
macro = np.mean(macroAllRandom)
micro_sd = np.std(microAllRandom)
macro_sd = np.std(macroAllRandom)
#
# zero_micro = np.mean(zero_microAllRandom)
# zero_macro = np.mean(zero_macroAllRandom)
# zero_micro_sd = np.std(zero_microAllRandom)
# zero_macro_sd = np.std(zero_macroAllRandom)

head_micro = np.mean(head_microAllRandom)
head_macro = np.mean(head_macroAllRandom)
head_micro_sd = np.std(head_microAllRandom)
head_macro_sd = np.std(head_macroAllRandom)

tail_micro = np.mean(tail_microAllRandom)
tail_macro = np.mean(tail_macroAllRandom)
tail_micro_sd = np.std(tail_microAllRandom)
tail_macro_sd = np.std(tail_macroAllRandom)

print('source and target networks:', str(source), str(target))

print("The avergae micro and macro F1 scores over %d random initializations are:  %f +/- %f and %f +/- %f: " % (
    numRandom, micro, micro_sd, macro, macro_sd))
# print("The avergae zero micro and macro F1 scores over %d random initializations are:  %f +/- %f and %f +/- %f: " % (
# numRandom, zero_micro, zero_micro_sd, zero_macro, zero_macro_sd))
print("The avergae tail micro and macro F1 scores over %d random initializations are:  %f +/- %f and %f +/- %f: " % (
    numRandom, tail_micro, tail_micro_sd, tail_macro, tail_macro_sd))
print("The avergae head micro and macro F1 scores over %d random initializations are:  %f +/- %f and %f +/- %f: " % (
    numRandom, head_micro, head_micro_sd, head_macro, head_macro_sd))

print("A_s_zero_num: %d      A_t_zero_num: %d" % (A_s_zero_num, A_t_zero_num))
print("A_s_tail_num: %d      A_t_tail_num: %d" % (A_s_tail_num, A_t_tail_num))
print("A_s_head_num: %d      A_t_head_num: %d" % (A_s_head_num, A_t_head_num))
print("A_s_all_num: %d      A_t_all_num: %d" % (
A_s_zero_num + A_s_tail_num + A_s_head_num, A_t_zero_num + A_t_tail_num + A_t_head_num))
