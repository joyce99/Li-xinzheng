import torch
from torch import nn
from functools import partial
import torch.nn.functional as F
from torch.autograd import Function
import math


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grads):
        dx = ctx.lambda_ * grads.neg()

        return dx, None


def uniform_neib_sampler(adj, ids, n_samples, device='cpu'):
    tmp = adj[ids]
    perm = torch.randperm(tmp.shape[1]).to(device)
    tmp = tmp[:, perm]

    return tmp[:, :n_samples]


class GraphSAGE(nn.Module):
    def __init__(self, aggregator_class, input_dim, layer_specs, device='cpu'):
        super(GraphSAGE, self).__init__()
        self.sample_fns = [partial(uniform_neib_sampler, n_samples=s['n_sample'], device=device) for s in layer_specs]
        agg_layers = []
        for spec in layer_specs:
            agg = aggregator_class(input_dim=input_dim, output_dim=spec['output_dim'], activation=spec['activation'])
            agg_layers.append(agg)
            input_dim = agg.output_dim
        self.agg_layers = nn.Sequential(*agg_layers)

    def forward(self, ids, adj, feats):
        tmp_feats = feats[ids]
        all_feats = [tmp_feats]
        for _, sampler_fn in enumerate(self.sample_fns):
            ids = sampler_fn(adj=adj, ids=ids).contiguous().view(-1)
            ids = ids.type(torch.int64)
            tmp_feats = feats[ids]
            all_feats.append(tmp_feats)
        for agg_layer in self.agg_layers.children():
            all_feats = [agg_layer(all_feats[k], all_feats[k + 1]) for k in range(len(all_feats) - 1)]
        assert len(all_feats) == 1, "len(all_feats) != 1"
        out = all_feats[0]

        return out


class Cly_net(nn.Module):
    def __init__(self, ninput, noutput, layers, activation=nn.ReLU(), dropout=nn.Dropout()):
        super(Cly_net, self).__init__()
        self.ninput = ninput
        self.noutput = noutput
        layer_sizes = [ninput] + [int(x) for x in layers.split('-') if x != '']
        self.layers = []
        layer = nn.Linear(layer_sizes[-1], noutput)
        self.layers.append(layer)
        self.add_module("layer"+str(len(layer_sizes)), layer)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)

        return x


class Disc(nn.Module):
    def __init__(self, ninput, layers, noutput=1):
        super(Disc, self).__init__()
        self.ninput = ninput
        self.noutput = noutput
        layer_sizes = [ninput] + [int(x) for x in layers.split('-')]
        self.model = nn.Sequential(
            nn.Linear(layer_sizes[0], layer_sizes[1]),
            nn.ReLU(),
            nn.Linear(layer_sizes[1], layer_sizes[2]),
            nn.ReLU(),
            nn.Linear(layer_sizes[2], layer_sizes[-1]),
            nn.ReLU(),
            nn.Linear(layer_sizes[-1], noutput),
        )

    def forward(self, x, lambda_):
        x = GradientReversalFunction.apply(x, lambda_)
        x = self.model(x)

        return x


class Mlp(nn.Module):
    def __init__(self, input_dim, hid_dim, dropout):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(input_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.act_fn = torch.nn.functional.gelu
        self._init_weights()

        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(hid_dim, eps=1e-6)

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.layernorm(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def get_feature_dis(x):
    """
    x :           batch_size x nhid
    x_dis(i,j):   item means the similarity between x(i) and x(j).
    """
    x_dis = x @ x.T
    mask = torch.eye(x_dis.shape[0]).cuda()
    x_sum = torch.sum(x ** 2, 1).reshape(-1, 1)
    x_sum = torch.sqrt(x_sum).reshape(-1, 1)
    x_sum = x_sum @ x_sum.T
    x_dis = x_dis * (x_sum ** (-1))
    x_dis = (1 - mask) * x_dis
    return x_dis


class GMLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GMLP, self).__init__()
        self.nhid = nhid
        self.mlp = Mlp(nfeat, self.nhid, dropout)
        self.classifier = nn.Linear(self.nhid, nclass)

    def forward(self, x):
        x = self.mlp(x)
        # return x
        feature_cls = x
        Z = x


        x_dis = get_feature_dis(Z)

        class_feature = self.classifier(feature_cls)
        class_logits = F.log_softmax(class_feature, dim=1)


        return x, class_logits, x_dis





