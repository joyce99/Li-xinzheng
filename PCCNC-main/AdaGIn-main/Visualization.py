import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from utils import *
from parse_args import parse_args

args = parse_args()
device = torch.device('cuda:' + str(args.device))
source = 'citationv1'
target = 'acmv9'

label_s = load_data_1(dataset=source + '.mat')
label_t = load_data_1(dataset=target + '.mat')

# A_s, X_s, Y_s = load_network('./Datasets/' + str(source) + '.mat')
# A_t, X_t, Y_t = load_network('./Datasets/' + str(target) + '.mat')

net = sio.loadmat("citationv1_acmv9_emb_PCCNC_wo_d.mat")
source_only_emb = net['rep_S']
target_only_emb = net['rep_T']

Y_s = label_s
Y_t = label_t


a = np.array(np.where(Y_s == 1))
y_0 = np.array(np.where(a[1] == 0))
# y_0 = np.delete(y_0, [246, 248, 161])
y_1 = np.array(np.where(a[1] == 1))
y_2 = np.array(np.where(a[1] == 2))
y_3 = np.array(np.where(a[1] == 3))
y_4 = np.array(np.where(a[1] == 4))
# y_5 = np.array(np.where(a[1] == 5))
# y_6 = np.array(np.where(a[1] == 6))

a_t = np.array(np.where(Y_t == 1))
y_0t = np.array(np.where(a_t[1] == 0))
# y_0t = np.delete(y_0t, [273, 274, 275, 276])
y_1t = np.array(np.where(a_t[1] == 1))
y_2t = np.array(np.where(a_t[1] == 2))
y_3t = np.array(np.where(a_t[1] == 3))
y_4t = np.array(np.where(a_t[1] == 4))
# y_5t = np.array(np.where(a_t[1] == 5))
# y_6t = np.array(np.where(a_t[1] == 6))
# print(b)

total_feature = np.vstack((source_only_emb, target_only_emb))
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
total_only_tsne = tsne.fit_transform(total_feature)

source_only_tsne = total_only_tsne[:source_only_emb.shape[0], :]
target_only_tsne = total_only_tsne[source_only_emb.shape[0]:, :]

plt.scatter(source_only_tsne[y_0, 0], source_only_tsne[y_0, 1], color='', edgecolors="navy", marker="^", s=100, linewidths=1, label="s1")
plt.scatter(source_only_tsne[y_1, 0], source_only_tsne[y_1, 1], color='', edgecolors="yellow", marker="^", s=100, linewidths=1, label="s2")
plt.scatter(source_only_tsne[y_2, 0], source_only_tsne[y_2, 1], color='', edgecolors="darkorange", marker="^", s=100, linewidths=1, label="s3")
plt.scatter(source_only_tsne[y_3, 0], source_only_tsne[y_3, 1], color='', edgecolors="g", marker="^", s=100, linewidths=1, label="s4")
# plt.scatter(source_only_tsne[y_4, 0], source_only_tsne[y_4, 1], color='', edgecolors="blue", marker="^", s=100, linewidths=1, label="s5")
# plt.scatter(source_only_tsne[y_5, 0], source_only_tsne[y_5, 1], color='', edgecolors="red", marker="^", s=100, linewidths=1, label="s6")
# plt.scatter(source_only_tsne[y_6, 0], source_only_tsne[y_6, 1], color='', edgecolors="black", marker="^", s=50, linewidths=1, label="1")
# plt.legend(["source"])
plt.scatter(target_only_tsne[y_0t, 0], target_only_tsne[y_0t, 1], color="navy", marker="+", s=100, linewidths=1, label="t1")
plt.scatter(target_only_tsne[y_1t, 0], target_only_tsne[y_1t, 1], color="yellow", marker="+", s=100, linewidths=1, label="t2")
plt.scatter(target_only_tsne[y_2t, 0], target_only_tsne[y_2t, 1], color="darkorange", marker="+", s=100, linewidths=1, label="t3")
plt.scatter(target_only_tsne[y_3t, 0], target_only_tsne[y_3t, 1], color="g", marker="+", s=100, linewidths=1, label="t4")
plt.scatter(target_only_tsne[y_4t, 0], target_only_tsne[y_4t, 1], color="blue", marker="+", s=100, linewidths=1, label="t5")
# plt.scatter(target_only_tsne[y_5t, 0], target_only_tsne[y_5t, 1], color="red", marker="+", s=100, linewidths=1, label="t6")
# plt.scatter(target_only_tsne[y_6t, 0], target_only_tsne[y_6t, 1], color="black", marker="+", s=50, linewidths=1)

# plt.scatter(target_only_tsne[y_0, 0], target_only_tsne[y_0, 1], color="navy", marker=".", s=100, linewidths=1)
# plt.scatter(target_only_tsne[y_1, 0], target_only_tsne[y_1, 1], color="yellow", marker=".", s=100, linewidths=1)
# plt.scatter(target_only_tsne[y_0t, 0], target_only_tsne[y_0t, 1], color="navy", marker=".", s=100, linewidths=1)
# plt.scatter(target_only_tsne[y_1t, 0], target_only_tsne[y_1t, 1], color="yellow", marker=".", s=100, linewidths=1)


# plt.legend(ncol=2)
plt.xticks([])
plt.yticks([])
# plt.xlabel("(a) DANN", fontsize=20)
plt.tight_layout()


# plt.savefig(filename="a2wour.png")
# plt.savefig('./Blog1_Blog2_ACDNE.eps', format='eps')
plt.show()
