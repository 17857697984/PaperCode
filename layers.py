import torch.nn as nn
import torch.nn.functional as F
import torch


class static_graph_constructor (nn.Module):
    def __init__(self, nnodes, dim, device, batch_size=4, out_channels=1):
        super (static_graph_constructor, self).__init__ ()
        self.nnodes = nnodes
        self.batch_size = batch_size
        self.device = device
        self.dim = dim
        self.conv2d_layer = nn.Conv2d (self.dim, out_channels, kernel_size=(1, 3))

    def forward(self, inputs):
        # 将输入矩阵输入到卷积层中
        output = self.conv2d_layer (inputs)
        # print ("Input shape:", inputs.shape)
        #
        # print (output.shape)
        #
        # print (output.permute (0, 2, 1, 3).shape)
        # 计算每个特征向量的余弦相似度
        adj_matrix = torch.cosine_similarity (output, output.permute (0, 2, 1, 3), dim=-1)

        # print ("邻接矩阵:")
        adj_matrix = adj_matrix.unsqueeze (1)
        # print (adj_matrix)
        # print (adj_matrix.shape)
        return adj_matrix


class gru_h (nn.Module):
    def __init__(self, horizon, nodes, seq_len, dim=1, batch_size=4):
        super (gru_h, self).__init__ ()
        self.horizon = horizon
        self.nodes = nodes
        self.seq_len = seq_len
        self.dim = dim
        self.batch_size = batch_size
        # self.fc = nn.Linear (1, 1)
        self.gru = nn.GRU (input_size=1, hidden_size=1, num_layers=seq_len, batch_first=True)

    def forward(self, x, hidden=None):
        # input shape: batch_size, dim, node_num, seq_len
        # x 的形状为 (batch_size, num_nodes, seq_len, input_size)
        x = x.permute(0, 2, 3, 1)
        batch_size, num_nodes, seq_len, _ = x.size ()

        future_hidden_states = None

        # 对每个节点单独进行预测
        for i in range (num_nodes):
            # 将输入数据进行 GRU 运算
            x_i, hidden_i = self.gru (x [:, i, :, :], hidden)

            for j in range (self.horizon):
                x_i, hidden_i = self.gru (x_i, hidden_i)
            if future_hidden_states is None:
                future_hidden_states = hidden_i
            else:
                future_hidden_states = torch.cat ((future_hidden_states, hidden_i), dim=2)
            # print("合成")
            # print(future_hidden_states.shape)

        future_hidden_states = future_hidden_states.unsqueeze(1)
        # print (future_hidden_states.shape)
        future_hidden_states = future_hidden_states.permute(2,1,3,0)
        result = []
        for i in range(future_hidden_states.size(3)):
            result.append(future_hidden_states[:,:,:,i].unsqueeze(3))
            # print(result[i].shape)
        return result


class dynamic_graph_constructor (nn.Module):
    def __init__(self, horizon, nodes, seq_len):
        super (dynamic_graph_constructor, self).__init__ ()
        self.horizon = horizon
        self.nodes = nodes
        self.num_layers = seq_len
        self.gru = gru_h(horizon, nodes, seq_len)

    def forward(self, inputs):  # input shape: batch_size, dim, node_num, seq_len
        # 得到每个时间步长的隐含状态
        h = self.gru(inputs)
        result = []
        for i in range(len(h)):
            # 计算每个特征向量的余弦相似度
            adj_matrix = torch.cosine_similarity (h[i], h[i].permute (0, 2, 1, 3), dim=-1)

            # print ("邻接矩阵:")
            adj_matrix = adj_matrix.unsqueeze (1)
            result.append(adj_matrix)
            # print (adj_matrix.shape)
        return result,h

class graph_interation (nn.Module):
    def __init__(self, k):
        super (graph_interation, self).__init__ ()
        self.k = k

    def forward(self, adj_set, adj_static):
        seq_len = len (adj_set)
        adj_static_list = []
        for i in range (seq_len):
            adj_sum = adj_static + torch.transpose(adj_static, 2, 3)
            # print (adj_sum.shape)
            # 创建单位矩阵，并扩展成与邻接矩阵相同的形状
            unit_matrix = torch.eye (adj_static.size (2)).unsqueeze (0).unsqueeze (0)
            unit_matrix = unit_matrix.repeat (adj_static.size (0), adj_static.size (1), 1, 1)
            # print (unit_matrix.shape)
            adj_sum = adj_sum + unit_matrix  # 添加单位矩阵
            # print(unit_matrix.shape)
            # print("完成")
            # print(adj_sum.shape)
            # 得到了动态图的mask
            mask_d = torch.where (adj_sum > 0, torch.ones_like (adj_sum), torch.zeros_like (adj_sum))  # 大于0设为1，否则设为0

            # 获取每个邻接矩阵的前 self.k 个最小值和对应的索引
            _, topK_indices = torch.topk (adj_set [i], self.k, dim=3, largest=False)

            # 根据索引创建 mask
            mask_s = torch.zeros_like (adj_set [i])
            mask_s.scatter_ (3, topK_indices, 1)

            # 获取每行的最大值
            max_adj, _ = torch.max (adj_set [i], dim=1)

            # 使用 mask 过滤出小于等于最大值的元素
            mask_s = torch.where (adj_set [i] <= max_adj.unsqueeze (1), torch.ones_like (adj_set [i]),torch.zeros_like (adj_set [i]))

            # print (adj_static.shape, mask_s.shape)
            # 通过mask进行交互
            adj_static = torch.mul (adj_static, mask_s)
            # print(adj_static.shape)
            adj_static_list.append (adj_static)
            adj_set [i] = torch.mul (adj_set [i], mask_d)
        return adj_static_list, adj_set


class Static_Graph_Convolution (nn.Module):
    def __init__(self, beta, node):
        super (Static_Graph_Convolution, self).__init__ ()
        self.beta = beta
        # 权重矩阵
        self.W = nn.Parameter (torch.FloatTensor (node, node))
        self.node_num = node
    def forward(self, adj_static_list, h_s_gru):
        result = []
        for i in range(len(adj_static_list)):
            A = adj_static_list[i]
            h = h_s_gru[i]
            inv_D = self.calculate_degree_and_inverse_matrix (A)
            unit_matrix = torch.eye (A.size (2)).unsqueeze (0).unsqueeze (0)
            unit_matrix = unit_matrix.repeat (A.size (0), A.size (1), 1, 1)
            A = A + unit_matrix # 添加单位矩阵
            # print (A.shape, h.shape)
            A_st = torch.mul(A, inv_D)
            # print (A_st.shape)
            A_st_11 = torch.mul (A_st, h)
            # print (h.shape, A_st.shape, A_st_11.shape)
            t1 = self.beta * h
            # print (t1.shape)
            t2 = (1 - self.beta) * A_st
            # print (t2.shape)
            temp = self.beta * h + (1 - self.beta) * A_st
            # print (temp.shape)
            temph = torch.matmul (temp, self.W)
            # print (temph.shape)
            result.append(temph)
        return result

    # 计算度矩阵和其逆矩阵
    def calculate_degree_and_inverse_matrix(self,A):
        # 计算度矩阵 D，保持形状不变
        D = torch.sum((A > 0).float(), dim=-1, keepdim=True).expand_as(A).clone()
        # print(D.shape)
        # 添加一个单位矩阵以防止除零错误
        unit_matrix = torch.eye (A.size (2)).unsqueeze (0).unsqueeze (0).to(A.device)
        unit_matrix = unit_matrix.repeat (A.size (0), A.size (1), 1, 1)
        # print(D.shape,unit_matrix.shape)
        D += unit_matrix
        # 计算度矩阵的逆矩阵
        D_inv = torch.reciprocal (D)
        return D_inv


class Dynamic_Graph_Convolution (nn.Module):
    def __init__(self):
        super (Dynamic_Graph_Convolution, self).__init__ ()

    def forward(self, adj_set, h_gru):
        result = []
        for i in range(len(adj_set)):
            A = adj_set[i]
            h = h_gru[i]
            # print(A.shape,h.shape)
            # 计算每行的指数形式的和
            row_exp_A = torch.sum (torch.exp (A), dim=1).unsqueeze(1)
            # print(row_exp_A.shape)
            # 将矩阵中的每个元素取指数
            matrix_exp = torch.exp (A)
            # print(matrix_exp.shape)
            # 将矩阵中的每个元素除以相应行和的指数形式的和
            normalized_A = matrix_exp / row_exp_A
            # print (normalized_A.shape)
            # 将归一化后的矩阵和矩阵 h 的同一行相乘并求和
            result.append(torch.sum (normalized_A * h, dim=1).unsqueeze(1))

        return result
class Prediction_Module (nn.Module):
    def __init__(self, horizon, nodes, seq_len, hidden_size = 8,input_size = 1, dim=1):
        super (Prediction_Module, self).__init__ ()
        self.horizon = horizon
        self.nodes = nodes
        self.seq_len = seq_len
        self.dim = dim
        self.fc1 = nn.Linear (input_size, hidden_size)  # 内置层，自动初始化权重矩阵
        self.fc2 = nn.Linear (hidden_size, 1)
    def forward(self, h_gru, h_s_gru):
        d = h_gru [-1]
        s = h_s_gru [-1]
        # print (s.shape, d.shape)
        a = torch.cat ([d, s], dim=1)
        # print (a.shape)
        x = a.view (a.size (0),a.size(2), -1)  # 将张量展平
        # print (x.shape)
        x = F.relu (self.fc1 (x))
        y = self.fc2 (x)
        # print (y.shape)
        # print(y)
        return y