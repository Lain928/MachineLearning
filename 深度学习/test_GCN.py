import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch.nn.init as init
import networkx as nx
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


G = nx.karate_club_graph()
print(G.number_of_nodes()) # 34
print(G.number_of_edges()) # 78


def norm(adj):
    adj += np.eye(adj.shape[0]) # 为每个结点增加自环
    degree = np.array(adj.sum(1)) # 为每个结点计算度
    degree = np.diag(np.power(degree, -0.5))
    return degree.dot(adj).dot(degree) # dot numpy中的矩阵计算


class GraphConvolution(nn.Module):
    def __init__(self, input_size, output_size):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, adj, features):
        out = torch.mm(adj, features) # torch.mm(a, b)是矩阵a和b矩阵相乘
        out = self.linear(out)
        return out


class GCN(nn.Module):
    def __init__(self, input_dim=34, hidden_size=5):
        super(GCN, self).__init__()
        self.gcn1 = GraphConvolution(input_dim, hidden_size)
        self.gcn2 = GraphConvolution(hidden_size, 2)

    def forward(self, adj, features):
        out = F.relu(self.gcn1(adj, features))
        out = self.gcn2(adj, out)
        return out


LEARNING_RATE = 0.1
WEIGHT_DACAY = 5e-4
EPOCHS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

features = np.eye(34, dtype=np.float) # 特征信息矩阵， 34维单位矩阵
print(features)

y = np.zeros(G.number_of_nodes())
for i in range(G.number_of_nodes()):
    if G.nodes[i]['club'] == 'Mr. Hi':
        y[i] = 0
    else:
        y[i] = 1

# #获得空手道俱乐部数据
# G = nx.karate_club_graph() # 获取图信息 空手道信息
# A = nx.adjacency_matrix(G).todense() # 得到邻接矩阵
# #A需要正规化
# A_normed = norm(torch.FloatTensor(A),True)

adj = np.zeros((34, 34))  # 邻阶矩阵
for k, v in G.adj.items():
    for item in v.keys():
        adj[k][item] = 1
adj = norm(adj)


features = torch.tensor(features, dtype=torch.float).to(DEVICE)
y = torch.tensor(y, dtype=torch.long).to(DEVICE)
adj = torch.tensor(adj, dtype=torch.float).to(DEVICE) # 代表邻接矩阵

net = GCN()
optimizer = optim.Adam(net.parameters())

def train():
    for epoch in range(EPOCHS):
        out = net(adj, features)
        mask = [False if x != 0 and x != 33 else True for x in range(34)] # 只选择管理员和教练进行训练
        loss = F.nll_loss(out[mask], y[mask])
        # l = loss(out[mask], y[mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"epoch: {epoch}, loss: {loss.item()}")
train()

import matplotlib.pyplot as plt
def draw():
    fig, ax = plt.subplots()
    fig.set_tight_layout(False)
    #nx_G = G.to_networkx().to_undirected()
    pos=nx.kamada_kawai_layout(G)
    nx.draw(G, pos, with_labels=True, node_color=[[0.5, 0.5, 0.5]])
    plt.show()
draw()
# r = net(adj, features).cpu().detach().numpy()
# fig = plt.figure()
# for i in range(34):
#     plt.scatter(r[i][0], r[i][1], color="r" if y[i] == 0 else 'b')
# plt.show()

