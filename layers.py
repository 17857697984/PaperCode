import torch.nn as nn
import torch.nn.functional as F
import torch


class static_graph_constructor (nn.Module):
    def __init__(self, nnodes, dim, device,batch_size, alpha=3):
        super (static_graph_constructor, self).__init__ ()
        self.nnodes = nnodes
        self.emb1 = nn.Embedding (nnodes, dim)
        self.emb2 = nn.Embedding (nnodes, dim)
        self.batch_size = batch_size
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
        result = None
        nodevec1 = self.emb1 (id)
        nodevec2 = self.emb2 (id)
        # print(nodevec1.shape)

        m1 = torch.tanh (self.alpha * torch.mm (nodevec1, self.theta1))
        m2 = torch.tanh (self.alpha * torch.mm (nodevec2, self.theta2))
        # print(m1.shape)
        # print(m2.transpose(1,0).shape)
        adj = torch.relu (torch.tanh (self.alpha * torch.mm (m1, m2.transpose (1, 0))))
        # adj = adj.unsqueeze (0)
        # print (adj.shape)
        # for i in range(self.batch_size):

            # if result is None:
            #     result = adj
            # else:
            #     result = torch.cat((result,adj),0)
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
        # 存放每一个时间步上的gru的隐含状态构建的图
        adj_set = []
        # 第一次输入得到第1-168个gru的隐含状态（就是2：168+1）
        for i in range(self.nodes):
            input_i = inputs[:,:,i,:].squeeze(2).transpose (1, 2)
            # print(input_i.shape)
            output,h_gru = self.gru(input_i)
            # print("隐状态")
            # print(h_gru.shape,output.shape)
            h0_gru.append(h_gru)
            outputs.append(output)
            # adj_set.append(h_gru)

        # 得到最终的gru隐状态
        for j in range(self.horizon-1):
            for i in range(self.nodes):
                # print(h0_gru[i])
                output,h_gru = self.gru(outputs[i],h0_gru[i])
                # print(h_gru)
                h0_gru[i] = h_gru
                outputs[i] = output
        result_gru  = None
        for i in range(self.nodes):
            if result_gru is None:
                result_gru = h0_gru[i]
            else:
                result_gru = torch.cat((result_gru,h0_gru[i]),dim=2)
                print (result_gru.shape)

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
        return adj_set,result_gru.transpose(0,1)
class graph_interation (nn.Module):
    def __init__(self,k):
        super (graph_interation, self).__init__ ()
        self.k = k
    def forward(self, adj_set,adj_static):
        seq_len = len(adj_set)
        for i in range(seq_len):
            adj_sum = adj_static + adj_static.transpose (0, 1)
            adj_sum += torch.eye (adj_static.size (0))  # 添加单位矩阵

            # 得到了动态图的mask
            mask_d = torch.where (adj_sum > 0, torch.ones_like (adj_sum), torch.zeros_like (adj_sum))  # 大于0设为1，否则设为0

            topK,_ = torch.topk(adj_set[i],self.k,dim=1)

            min_adj,_ = torch.min(topK,dim=1)

            # 得到静态图的mask
            mask_s = torch.where (adj_set[i] >= min_adj, torch.ones_like (adj_set[i]), torch.zeros_like (adj_set[i]))
            print(adj_static.shape,mask_s.shape)
            # 通过mask进行交互
            adj_static = torch.mm(adj_static, mask_s)
            adj_set[i] = torch.mm(adj_set[i], mask_d)
        return adj_static,adj_set

class Static_Graph_Convolution(nn.Module):
    def __init__(self, beta,node):
        super(Static_Graph_Convolution, self).__init__()
        self.beta = beta
        # 权重矩阵
        self.W = nn.Parameter(torch.FloatTensor(node, node))

    def forward(self, A, h):
        D = self.calculate_degree_matrix (A)
        inv_D = self.calculate_inverse_degree_matrix (D)
        A += torch.eye (A.size (0))  # 添加单位矩阵
        A_st = torch.mm(A, inv_D)
        print(A_st.shape,h.shape)
        A_st_11 = torch.mm(A_st, h)
        print(h.shape,A_st.shape,A_st_11.shape)
        t1 = self.beta * h
        print(t1.shape)
        t2 = (1 - self.beta) * A_st
        print (t2.shape)
        temp  = self.beta * h + (1 - self.beta) * A_st
        print(temp.shape)
        temph = torch.matmul(temp, self.W)
        print(temph.shape)
        return temph

    def calculate_degree_matrix(self, A):
        # 计算每一行的和，即节点的度数
        degrees = torch.sum (A, dim=1)
        # 构造度矩阵 D
        D = torch.diag (degrees) + torch.eye (degrees.size (0))
        return D

    def calculate_inverse_degree_matrix(self,D):
        # 计算度矩阵的逆矩阵
        print(D)
        D_inv = torch.inverse (D)
        return D_inv
class Dynamic_Graph_Convolution(nn.Module):
    def __init__(self):
        super(Dynamic_Graph_Convolution, self).__init__()
    def forward(self, A, h):
        # 计算每行的指数形式的和
        row_exp_A = torch.sum (torch.exp (A), dim=1)

        # 将矩阵中的每个元素取指数
        matrix_exp = torch.exp (A)

        # 将矩阵中的每个元素除以相应行和的指数形式的和
        normalized_A = matrix_exp / row_exp_A.view (-1, 1)

        # 将归一化后的矩阵和矩阵 h 的同一行相乘并求和
        result = torch.sum (normalized_A * h, dim=1)

        return result


