import numpy as np
import tensorflow as tf
import utils
from flip_gradient import flip_gradient

class MSCNC(object):
    def __init__(self, n_input, n_hidden, n_emb, num_class, clf_type, l2_w, batch_size):
        self.X_sim = tf.sparse_placeholder(dtype=tf.float32)
        self.X_sec_sim = tf.sparse_placeholder(dtype=tf.float32)
        self.y_true = tf.placeholder(dtype=tf.float32)
        self.d_label = tf.placeholder(dtype=tf.float32)  # domain label, source network [1 0] or target network [0 1]
        self.Ada_lambda = tf.placeholder(dtype=tf.float32)  # grl_lambda Gradient reversal scaler
        self.dropout = tf.placeholder(tf.float32)
        self.A_s_X = tf.sparse_placeholder(dtype=tf.float32)  # network proximity matrix of source network
        self.A_t_X = tf.sparse_placeholder(dtype=tf.float32)  # network proximity matrix of target network
        self.A_s_A = tf.sparse_placeholder(dtype=tf.float32)  # network proximity matrix of source network
        self.A_t_A = tf.sparse_placeholder(dtype=tf.float32)  # network proximity matrix of target network
        self.mask = tf.placeholder(dtype=tf.float32)  # check a node is with observable label (1) or not (0)
        self.learning_rate = tf.placeholder(dtype=tf.float32)

        with tf.name_scope('Network_Embedding'):
            ##feature exactor
            h1_sim = utils.fc_layer(self.X_sim, n_input, n_hidden[0], layer_name='hidden1_sim', input_type='sparse',
                                    drop=self.dropout)
            h2_sim = utils.fc_layer(h1_sim, n_hidden[0], n_hidden[1], layer_name='hidden2_sim')

            h1_sec_sim = utils.fc_layer(self.X_sec_sim, n_input, n_hidden[0], layer_name='hidden1_sec_sim',
                                        input_type='sparse', drop=self.dropout)
            h2_sec_sim = utils.fc_layer(h1_sec_sim, n_hidden[0], n_hidden[1], layer_name='hidden2_sec_sim')

            '''all'''
            h_emb = tf.concat([h2_sim, h2_sec_sim], 1)
            emb_f_1 = utils.fc_layer(h_emb, n_hidden[-1] * 2, n_hidden[0], layer_name='concat')
            self.emb = utils.fc_layer(emb_f_1, n_hidden[0], n_hidden[1], layer_name='concat')

        with tf.name_scope('Node_Classifier'):
            ##node classification
            W_clf = tf.Variable(tf.truncated_normal([n_emb, num_class], stddev=1. / tf.sqrt(n_emb / 2.)),
                                name='clf_weight')
            b_clf = tf.Variable(tf.constant(0.1, shape=[num_class]), name='clf_bias')
            pred_logit = tf.matmul(self.emb, W_clf) + b_clf

            if clf_type == 'multi-class':
                ### multi-class, softmax output
                loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred_logit, labels=self.y_true)
                loss = loss * self.mask  # count loss only based on labeled nodes
                self.clf_loss = tf.reduce_sum(loss) / tf.reduce_sum(self.mask)
                self.pred_prob = tf.nn.softmax(pred_logit)

            elif clf_type == 'multi-label':
                ### multi-label, sigmod output
                loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=pred_logit, labels=self.y_true)
                loss = loss * self.mask[:, None]  # count loss only based on labeled nodes, each column mutiply by mask
                self.clf_loss = tf.reduce_sum(loss) / tf.reduce_sum(self.mask)
                self.pred_prob = tf.sigmoid(pred_logit)

        # two-layer adversarial
        with tf.name_scope('Domain_Discriminator'):
            self.domain_loss_sim = utils.domain_Dis(h2_sim, self.Ada_lambda, n_emb, self.d_label)
            self.domain_loss_sec_sim = utils.domain_Dis(h2_sec_sim, self.Ada_lambda, n_emb, self.d_label)
            self.multi_domain_loss = self.domain_loss_sec_sim + self.domain_loss_sim

            self.domain_loss_global = utils.domain_Dis(self.emb, self.Ada_lambda, n_emb, self.d_label)

        all_variables = tf.trainable_variables()
        self.l2_loss = l2_w * tf.add_n([tf.nn.l2_loss(v) for v in all_variables if 'bias' not in v.name])
        self.total_loss = self.clf_loss + self.l2_loss
        self.total_loss += 0.1 * self.multi_domain_loss
        self.total_loss += self.domain_loss_global

        self.train_op = tf.train.MomentumOptimizer(self.learning_rate, 0.9).minimize(self.total_loss)
