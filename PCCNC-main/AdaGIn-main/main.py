from parse_args import *
from utils import *
from dgi import *
import torch.nn.functional as F
import loss_func
from modules import network, contrastive_loss
import scipy


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    args.instance_temperature = 0.5
    args.cluster_temperature = 1.0
    print(device)
    print(args)
    print("loading data...")
    adj_s, feature_s, label_s, idx_tot_s = load_data(dataset=args.source_dataset + '.mat', device=device,
                                                                seed=args.seed, is_blog=args.is_blog)
    adj_t, feature_t, label_t, idx_tot_t = load_data(dataset=args.target_dataset + '.mat', device=device,
                                                                seed=args.seed, is_blog=args.is_blog)

    n_samples = args.n_samples.split(',')
    output_dims = args.output_dims.split(',')
    emb_filename = str(args.source_dataset) + '_' + str(args.target_dataset)
    emb_model = GraphSAGE(**{
        "aggregator_class": aggregator_lookup[args.aggregator_class],
        "input_dim": feature_s.shape[1],
        "layer_specs": [
            {
                "n_sample": int(n_samples[0]),
                "output_dim": int(output_dims[0]),
                "activation": F.relu,
            },
            {
                "n_sample": int(n_samples[1]),
                "output_dim": int(output_dims[1]),
                "activation": F.relu,
            },
            {
                "n_sample": int(n_samples[-1]),
                "output_dim": int(output_dims[-1]),
                "activation": F.relu,
            }
        ],
        "device": device
    }).to(device)

    criterion_instance = contrastive_loss.InstanceLoss(int(args.batch_size/2), args.instance_temperature, device).to(device)
    criterion_cluster = contrastive_loss.ClusterLoss(label_t.shape[1], args.cluster_temperature, device).to(device)
    cl_model = network.Network(2 * int(output_dims[-1]), 2 * int(output_dims[-1]), label_s.shape[1]).to(device)
    cly_model = Cly_net(2 * int(output_dims[-1]), label_s.shape[1], args.arch_cly).to(device)
    disc_model = Disc(2 * int(output_dims[-1]) * label_s.shape[1], args.arch_disc, 1).to(device)
    # define the optimizers
    total_params = list(emb_model.parameters()) + list(cly_model.parameters()) + list(disc_model.parameters()) + list(
        criterion_cluster.parameters()) + list(criterion_instance.parameters()) + list(cl_model.parameters())
    dgi_model = DGI(2 * int(output_dims[-1])).to(device)
    total_params += list(dgi_model.parameters())
    cly_optim = torch.optim.Adam(total_params, lr=args.lr_cly, weight_decay=args.weight_decay)
    lr_lambda = lambda epoch: (1 + 10 * float(epoch) / args.epochs) ** (-0.75)
    scheduler = torch.optim.lr_scheduler.LambdaLR(cly_optim, lr_lambda=lr_lambda)
    best_micro_f1, best_macro_f1, best_epoch, best_acc = 0, 0, 0, 0
    num_batch = round(max(feature_s.shape[0] / (args.batch_size / 2), feature_t.shape[0] / (args.batch_size / 2)))
    print('start learning')

    for epoch in range(args.epochs):
        s_batches = batch_generator(idx_tot_s, int(args.batch_size / 2))
        t_batches = batch_generator(idx_tot_t, int(args.batch_size / 2))
        emb_model.train()
        cly_model.train()
        disc_model.train()
        dgi_model.train()
        cl_model.train()
        criterion_cluster.train()
        criterion_instance.train()
        p = float(epoch) / args.epochs
        grl_lambda = min(2. / (1. + np.exp(-10. * p)) - 1, 0.2)
        for iter in range(num_batch):
            b_nodes_s = next(s_batches)
            b_nodes_t = next(t_batches)
            source_features, cly_loss_s = do_iter(emb_model, cly_model, adj_s, feature_s, label_s, idx=b_nodes_s,
                                                  is_social_net=args.is_social_net)
            target_features, _ = do_iter(emb_model, cly_model, adj_t, feature_t, label_t, idx=b_nodes_t,
                                         is_social_net=args.is_social_net)

            shuf_idx_s = np.arange(label_s.shape[0])
            np.random.shuffle(shuf_idx_s)
            shuf_feat_s = feature_s[shuf_idx_s, :]
            shuf_idx_t = np.arange(label_t.shape[0])
            np.random.shuffle(shuf_idx_t)
            shuf_feat_t = feature_t[shuf_idx_t, :]
            aug_source_feats_1 = emb_model(b_nodes_s, adj_s, shuf_feat_s)
            logits_s = dgi_model(aug_source_feats_1, source_features)
            aug_target_feats_1 = emb_model(b_nodes_t, adj_t, shuf_feat_t)
            logits_t = dgi_model(aug_target_feats_1, target_features)

            '''结构全局信息'''
            labels_dgi = torch.cat(
                [torch.zeros(int(args.batch_size / 2)), torch.ones(int(args.batch_size / 2))]).unsqueeze(0).to(device)
            dgi_loss = args.dgi_param * (
                    F.binary_cross_entropy_with_logits(logits_s, labels_dgi) + F.binary_cross_entropy_with_logits(
                logits_t, labels_dgi))

            '''learn node or cluster'''
            z_i_s, z_j_s, c_i_s, c_j_s = cl_model(source_features, aug_source_feats_1)
            z_i_t, z_j_t, c_i_t, c_j_t = cl_model(target_features, aug_target_feats_1)

            '''node-level cl loss'''
            loss_instance_s = criterion_instance(z_i_s, z_j_s)
            loss_instance_t = criterion_instance(z_i_t, z_j_t)

            '''cluster-level cl loss'''
            loss_cluster_s = criterion_cluster(c_i_s, c_j_s)
            loss_cluster_t = criterion_cluster(c_i_t, c_j_t)

            '''node-level cl loss'''
            loss_node = loss_instance_s + loss_instance_t
            loss_node += dgi_loss

            '''cluster-level cl loss'''
            loss_cluster_s_t = criterion_cluster(c_j_s, c_j_t) + criterion_cluster(c_i_s, c_i_t)

            loss_cluster = loss_cluster_s + loss_cluster_t
            loss_cluster += loss_cluster_s_t
            # loss_cluster = loss_cluster_s_t

            '''cl all loss'''
            cl_loss = loss_node + loss_cluster
            # cl_loss = loss_node
            # cl_loss = loss_cluster
            # cl_loss = loss_cluster_s_t

            features = torch.cat((source_features, target_features), 0)
            outputs = cly_model(features)
            softmax_output = nn.Softmax(dim=1)(outputs)
            # print('dgi_loss {:.4f}'.format(dgi_loss))

            domain_loss = args.cdan_param * loss_func.CDAN([features, softmax_output], disc_model, None, grl_lambda,
                                                           None, device=device)
            loss = cly_loss_s
            loss += domain_loss
            loss += cl_loss
            cly_optim.zero_grad()
            loss.backward()
            cly_optim.step()

        emb_model.eval()
        cly_model.eval()
        cly_loss_bat_s, micro_f1_s, macro_f1_s, embs_whole_s, targets_whole_s = evaluate(emb_model, cly_model,
                                                                                                adj_s, feature_s,
                                                                                                label_s,
                                                                                                idx_tot_s,
                                                                                                args.batch_size,
                                                                                                mode='test',
                                                                                                is_social_net=args.is_social_net)

        print("epoch {:03d}, source_data {}, target_data {}".format(epoch, args.source_dataset, args.target_dataset))
        print("source loss {:.4f} | source micro-F1 {:.4f} | source macro-F1 {:.4f}".
              format(cly_loss_bat_s, micro_f1_s, macro_f1_s))
        cly_loss_bat_t, micro_f1_t, macro_f1_t, embs_whole_t, targets_whole_t = evaluate(emb_model, cly_model,
                                                                                                adj_t, feature_t,
                                                                                                label_t,
                                                                                                idx_tot_t,
                                                                                                args.batch_size,
                                                                                                mode='test',
                                                                                                is_social_net=args.is_social_net)
        print("target loss {:.4f} | target micro-F1 {:.4f} | target macro-F1 {:.4f}".format(
            cly_loss_bat_t, micro_f1_t, macro_f1_t))
        # if(acc_t > best_acc):
        #     best_acc = acc_t
        #     best_epoch = epoch
        #     print('saving model...')

        if (micro_f1_t + macro_f1_t) > (best_micro_f1 + best_macro_f1):
            best_micro_f1 = micro_f1_t
            best_macro_f1 = macro_f1_t
            best_epoch = epoch
            print('saving model...')
            scipy.io.savemat('./' + emb_filename + '_emb_PCCNC_wo_d.mat',
                             {'rep_S': embs_whole_s, 'rep_T': embs_whole_t})
        scheduler.step()
    print("test metrics on target graph:")
    print('---------- random seed: {:03d} ----------'.format(args.seed))
    print("best_epoch {}, micro-F1 {:.4f} | macro-F1 {:.4f} | best acc {:.4f}".format(best_epoch, best_micro_f1,
                                                                                      best_macro_f1, best_acc))



if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cuda:' + str(args.device))
    main(args)
