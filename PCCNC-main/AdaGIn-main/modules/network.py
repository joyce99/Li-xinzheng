import torch.nn as nn
import torch
from torch.nn.functional import normalize


class Network(nn.Module):
    def __init__(self, intput_dim, feature_dim, class_num):
        super(Network, self).__init__()
        self.input_dim = intput_dim
        self.feature_dim = feature_dim
        self.cluster_num = class_num
        self.instance_projector = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim),
            nn.ReLU(),
            nn.Linear(self.input_dim, self.feature_dim),
        )
        self.cluster_projector = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim),
            nn.ReLU(),
            nn.Linear(self.input_dim, self.cluster_num),
            nn.Softmax(dim=1)
        )

    def forward(self, x_i, x_j):
        # h_i = self.resnet(x_i)
        # h_j = self.resnet(x_j)

        z_i = normalize(self.instance_projector(x_i), dim=1)
        z_j = normalize(self.instance_projector(x_j), dim=1)

        c_i = self.cluster_projector(x_i)
        c_j = self.cluster_projector(x_j)

        return z_i, z_j, c_i, c_j

    def forward_cluster(self, x):
        h = self.resnet(x)
        c = self.cluster_projector(h)
        c = torch.argmax(c, dim=1)
        return c
