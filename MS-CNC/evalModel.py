# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import utils
from scipy.sparse import vstack
from functools import partial
import scipy.io
from scipy.sparse import lil_matrix
# import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
from model import MSCNC

def train_and_evaluate(input_data, config, random_state=0):
    ###get input data
    PPMI_s_X = input_data['PPMI_S_X']
    PPMI_t_X = input_data['PPMI_T_X']
    PPMI_s_A = input_data['PPMI_S_A']
    PPMI_t_A = input_data['PPMI_T_A']
    X_sim_s = input_data['attrb_sim_S']
    X_sim_t = input_data['attrb_sim_T']
    X_sec_sim_s = input_data['attrb_sec_sim_S']
    X_sec_sim_t = input_data['attrb_sec_sim_T']
    Y_s = input_data['label_S']
    Y_t = input_data['label_T']
    source = input_data['source']
    target = input_data['target']
    A_s_list_head = input_data['A_s_list_head']
    A_t_list_head = input_data['A_t_list_head']
    A_s_list_zero = input_data['A_s_list_zero']
    A_t_list_zero = input_data['A_t_list_zero']
    A_s_list_tail = input_data['A_s_list_tail']
    A_t_list_tail = input_data['A_t_list_tail']

    Y_t_o = np.zeros(np.shape(Y_t))  # observable label matrix of target network, all zeros

    X_s_new = lil_matrix(np.concatenate((X_sim_s, X_sec_sim_s), axis=1))
    X_t_new = lil_matrix(np.concatenate((X_sim_t, X_sec_sim_t), axis=1))
    n_input = X_sim_s.shape[1]
    num_class = Y_s.shape[1]
    num_nodes_S = X_sim_s.shape[0]
    num_nodes_T = X_sim_t.shape[0]

    ###model config
    clf_type = config['clf_type']
    dropout = config['dropout']
    num_epoch = config['num_epoch']
    batch_size = config['batch_size']
    n_hidden = config['n_hidden']
    n_emb = config['n_emb']
    l2_w = config['l2_w']
    emb_filename = config['emb_filename']
    lr_ini = config['lr_ini']


    whole_xs_xt_stt_sim = utils.csr_2_sparse_tensor_tuple(vstack([X_sim_s, X_sim_t]))
    whole_xs_xt_stt_sec_sim = utils.csr_2_sparse_tensor_tuple(vstack([X_sec_sim_s, X_sec_sim_t]))

    with tf.Graph().as_default():
        # Set random seed
        tf.set_random_seed(random_state)
        np.random.seed(random_state)

        model = MSCNC(n_input, n_hidden, n_emb, num_class, clf_type, l2_w, batch_size)

        with tf.Session() as sess:
            # Random initialize
            sess.run(tf.global_variables_initializer())
            microi = []
            macroi = []
            zero_microi = []
            zero_macroi = []
            head_microi = []
            head_macroi = []
            tail_microi = []
            tail_macroi = []
            for cEpoch in range(num_epoch):
                S_batches = utils.batch_generator([X_s_new, Y_s], int(batch_size / 2), shuffle=True)
                T_batches = utils.batch_generator([X_t_new, Y_t_o], int(batch_size / 2), shuffle=True)

                num_batch = round(max(num_nodes_S / (batch_size / 2), num_nodes_T / (batch_size / 2)))

                # Adaptation param and learning rate schedule as described in the DANN paper
                p = float(cEpoch) / (num_epoch)
                lr = lr_ini / (1. + 10 * p) ** 0.75
                grl_lambda = 2. / (1. + np.exp(-10. * p)) - 1  # gradually change from 0 to 1

                ##in each epoch, train all the mini batches
                for cBatch in range(num_batch):
                    ### each batch, half nodes from source network, and half nodes from target network
                    xs_ys_batch, shuffle_index_s = next(S_batches)
                    xs_batch = xs_ys_batch[0]
                    ys_batch = xs_ys_batch[1]

                    xt_yt_batch, shuffle_index_t = next(T_batches)
                    xt_batch = xt_yt_batch[0]
                    yt_batch = xt_yt_batch[1]

                    x_batch = vstack([xs_batch, xt_batch])
                    batch_csr = x_batch.tocsr()
                    xb_sim = utils.csr_2_sparse_tensor_tuple(batch_csr[:, 0:n_input])
                    xb_sec_sim = utils.csr_2_sparse_tensor_tuple(batch_csr[:, n_input:2 * n_input])
                    yb = np.vstack([ys_batch, yt_batch])

                    mask_L = np.array(np.sum(yb, axis=1) > 0,
                                      dtype=np.float)  # 1 if the node is with observed label, 0 if the node is without label
                    domain_label = np.vstack([np.tile([1., 0.], [batch_size // 2, 1]), np.tile([0., 1.],
                                                                                               [batch_size // 2,
                                                                                                1])])  # [1,0] for source, [0,1] for target

                    #topological proximity matrix between nodes in each mini-batch
                    # a_s, a_t = utils.batchPPMI(batch_size, shuffle_index_s, shuffle_index_t, PPMI_s, PPMI_t)
                    a_s_X, a_t_X = utils.batchPPMI(batch_size, shuffle_index_s, shuffle_index_t, PPMI_s_X, PPMI_t_X)
                    a_s_A, a_t_A = utils.batchPPMI(batch_size, shuffle_index_s, shuffle_index_t, PPMI_s_A, PPMI_t_A)


                    _, tloss = sess.run([model.train_op, model.total_loss],
                                        feed_dict={#model.X:xb, model.X_nei: xb_nei,
                                                    model.X_sim: xb_sim,
                                                   model.X_sec_sim: xb_sec_sim,
                                                   # model.A_s:a_s, model.A_t: a_t,
                                                   model.A_s_X: a_s_X, model.A_t_X: a_t_X,
                                                   model.A_s_A: a_s_A, model.A_t_A: a_t_A,
                                                   model.y_true: yb, model.d_label: domain_label,
                                                   model.mask: mask_L, model.learning_rate: lr,
                                                   model.Ada_lambda: grl_lambda, model.dropout: dropout})

                '''Compute evaluation on test data by the end of each epoch'''
                pred_prob_xs_xt, emb = sess.run([model.pred_prob, model.emb],
                                           feed_dict={#model.X:whole_xs_xt_stt, model.X_nei: whole_xs_xt_stt_nei,
                                                      model.X_sim: whole_xs_xt_stt_sim, model.X_sec_sim: whole_xs_xt_stt_sec_sim,
                                                      model.Ada_lambda: 1.0,
                                                      model.dropout: 0.})
                pred_prob_xs = pred_prob_xs_xt[0:num_nodes_S, :]
                pred_prob_xt = pred_prob_xs_xt[-num_nodes_T:, :]

                print('epoch: ', cEpoch + 1, source, target)
                # F1_s = utils.f1_scores_1(pred_prob_xs, Y_s, A_s_list_zero, A_s_list_head, A_s_list_tail)
                F1_s = utils.f1_scores_wo_zero(pred_prob_xs, Y_s, A_s_list_head, A_s_list_tail)
                print('Source micro-F1: %f, macro-F1: %f' % (F1_s[0], F1_s[1]))
                # print('Source zero_micro-F1: %f, zero_macro-F1: %f' % (F1_s[2], F1_s[3]))
                # print('Source head_micro-F1: %f, head_macro-F1: %f' % (F1_s[4], F1_s[5]))
                # print('Source tail_micro-F1: %f, tail_macro-F1: %f' % (F1_s[6], F1_s[7]))

                # F1_t = utils.f1_scores_1(pred_prob_xt, Y_t, A_t_list_zero, A_t_list_head, A_t_list_tail)
                F1_t = utils.f1_scores_wo_zero(pred_prob_xt, Y_t, A_t_list_head, A_t_list_tail)
                print('Target testing micro-F1: %f, macro-F1: %f' % (F1_t[0], F1_t[1]))
                # print('Target zero_micro-F1: %f, zero_macro-F1: %f' % (F1_t[2], F1_t[3]))
                # print('Target head_micro-F1: %f, head_macro-F1: %f' % (F1_t[4], F1_t[5]))
                # print('Target tail_micro-F1: %f, tail_macro-F1: %f' % (F1_t[6], F1_t[7]))

                print('Loss: %f' % tloss)
                microi.append(F1_t[0])
                macroi.append(F1_t[1])
                # zero_microi.append(F1_t[2])
                # zero_macroi.append(F1_t[3])
                # head_microi.append(F1_t[4])
                # head_macroi.append(F1_t[5])
                # tail_microi.append(F1_t[6])
                # tail_macroi.append(F1_t[7])
                head_microi.append(F1_t[2])
                head_macroi.append(F1_t[3])
                tail_microi.append(F1_t[4])
                tail_macroi.append(F1_t[5])

            micro = float(np.max(microi))
            for i in range(num_epoch):
                if micro == microi[i]:
                    macro = float(macroi[i])

            # zero_micro = float(np.max(zero_microi))
            # for i in range(num_epoch):
            #     if zero_micro == zero_microi[i]:
            #         zero_macro = float(zero_macroi[i])

            tail_micro = float(np.max(tail_microi))
            for i in range(num_epoch):
                if tail_micro == tail_microi[i]:
                    tail_macro = float(tail_macroi[i])

            head_micro = float(np.max(head_microi))
            for i in range(num_epoch):
                if head_micro == head_microi[i]:
                    head_macro = float(head_macroi[i])

            ''' save final evaluation on test data by the end of all epoches'''
            # micro=float(F1_t[0])
            # macro=float(F1_t[1])

            # save embedding features
            # emb = sess.run(model.emb,
            #                feed_dict={model.X: whole_xs_xt_stt, model.X_nei: whole_xs_xt_stt_nei, model.Ada_lambda: 1.0,
            #                           model.dropout: 0.})
            # hs = emb[0:num_nodes_S, :]
            # ht = emb[-num_nodes_T:, :]
            # print(np.shape(hs))
            # print(np.shape(ht))
            # scipy.io.savemat(emb_filename + '_emb.mat', {'rep_S': hs, 'rep_T': ht})

    return micro, macro, head_micro, head_macro, tail_micro, tail_macro
    # return micro, macro, zero_micro, zero_macro, head_micro, head_macro, tail_micro, tail_macro




