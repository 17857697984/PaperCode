import torch.nn as nn
import torch.nn.functional as F
import torch


class static_graph_constructor (nn.Module):
    def __init__(self, nnodes, dim, device, alpha=3, theta1=0.5, theta2=0.5):
        super (static_graph_constructor, self).__init__ ()
        self.nnodes = nnodes
        self.emb1 = nn.Embedding (nnodes, dim)
        self.emb2 = nn.Embedding (nnodes, dim)

        self.device = device
        self.dim = dim
        # alpha为模型的超参数
        self.alpha = alpha
        # 设置theta1和theta2为可学习的参数
        self.theta1 = torch.nn.Parameter(torch.FloatTensor(dim,dim),requires_grad=True)
        self.theta2 = torch.nn.Parameter(torch.FloatTensor(dim,dim),requires_grad=True)

        # self.lin1 = nn.Linear (dim, dim)
        # self.lin2 = nn.Linear (dim, dim)

    def forward(self, id):
        nodevec1 = self.emb1 (id)
        nodevec2 = self.emb2 (id)
        # print(nodevec1.shape)

        m1 = torch.tanh (self.alpha * torch.mm(nodevec1 , self.theta1))
        m2 = torch.tanh (self.alpha * torch.mm (nodevec2 , self.theta2))
        # print(m1.shape)
        # print(m2.transpose(1,0).shape)
        adj = torch.relu (torch.tanh (self.alpha * torch.mm (m1, m2.transpose (1, 0))))
        # print(adj.shape)
        # print(adj)

        return adj


class dynamic_graph_constructor (nn.Module):
    def __init__(self, horizon,nodes,seq_len):
        super (dynamic_graph_constructor, self).__init__ ()
        self.horizon = horizon
        self.nodes = nodes
        self.num_layers = seq_len
        self.gru = nn.GRU(input_size=1, hidden_size=1, num_layers=self.num_layers, batch_first=True)

    def forward(self, inputs):  # input shape: batch_size, dim, node_num, seq_len
        seq_len = inputs.size(3)
        node_num = inputs.size(2)
        # print(seq_len,node_num)
        # h_gru = torch.zeros (self.num_layers, inputs.size (0), self.hidden_dim).to (inputs.device)
        h0_gru = []
        outputs = []
        adj_set = []
        # 第一次输入得到第1-168个gru的隐含状态（就是2：168+1）
        for i in range(self.nodes):
            input_i = inputs[:,:,i,:].squeeze(2).transpose (1, 2)
            # print(input_i.shape)
            output,h_gru = self.gru(input_i)
            # print(h_gru.shape,output.shape)
            h0_gru.append(h_gru)
            outputs.append(output)
            # adj_set.append(h_gru)

        # 去得到最终的gru隐状态
        for j in range(self.horizon-1):
            for i in range(self.nodes):
                # print(h0_gru[i])
                output,h_gru = self.gru(outputs[i],h0_gru[i])
                # print(h_gru)
                h0_gru[i] = h_gru
                outputs[i] = output
        # print (len (h0_gru), len (outputs))
            # adj_set.append(h_gru))
        # 逐时间步来建立每个时间步上的图
        for i in range(seq_len):
            temp_adj = torch.zeros(self.nodes, self.nodes).to(inputs.device)
            for j in range(self.nodes):
                for k in range(self.nodes):
                    # print (h0_gru[j].shape, h0_gru[j].shape)
                    h1 = h0_gru[j][i,:,:]
                    h2 = h0_gru[k][i,:,:]
                    # print(h1,h2)
                    # 根据h1和h2的余弦相似度来给矩阵对应位置赋值
                    temp_adj[j,k] = torch.cosine_similarity(h1, h2, dim=0)
            # print(temp_adj)
            adj_set.append(temp_adj)
        # print(len(adj_set))
        return adj_set,h0_gru
class graph_interation (nn.Module):
    def __init__(self,k):
        super (graph_interation, self).__init__ ()
        self.k = k
    def forward(self, adj_set,adj_static):
        len = len(adj_set)

        for i in range(len):
            adj_sum = adj_set[i] + adj_set[i].transpose (0, 1)
            adj_sum += torch.eye (adj_set[i].size (0))  # 添加单位矩阵
            mark_s = torch.where (adj_sum > 0, torch.ones_like (adj_sum), torch.zeros_like (adj_sum))  # 大于0设为1，否则设为0
    def topK(self):
        pass




