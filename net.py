import layers
import torch.nn as nn
import torch

class Net(nn.Module):
    def __init__(self, device, horizon, nodes, seq_len, dim=1, alpha=0.5, k=1,beta=0.5,batch_size=4):
        super(Net, self).__init__()
        self.device = device
        self.horizon = horizon
        self.nodes = nodes
        self.seq_len = seq_len
        self.k = k
        self.alpha = alpha
        self.dim = dim
        self.h_gru = []
        self.adj_set = []
        self.batch_size = 4
        self.static_graph_constructor = layers.static_graph_constructor(self.nodes, self.dim, self.device,self.batch_size)
        self.dynamic_graph_constructor = layers.dynamic_graph_constructor(self.horizon, self.nodes, self.seq_len)
        self.graph_interation = layers.graph_interation(self.k,self.device)
        self.static_Graph_Convolution = layers.Static_Graph_Convolution(beta, self.nodes,self.device)
        self.dynamic_Graph_Convolution = layers.Dynamic_Graph_Convolution()
        self.gru = layers.gru_h(horizon, nodes, seq_len)
        self.prediction_Module = layers.Prediction_Module(self.horizon, self.nodes, self.seq_len,input_size=2*nodes)
    def forward(self,inputs):
        # print("开始")
        # 获得静态图
        static_graph = self.static_graph_constructor(inputs)
        # 获得动态图和gru的隐状态
        self.adj_set,self.h_gru = self.dynamic_graph_constructor(inputs)
        # print("完成静态图和动态图的构建")
        # print(static_graph.shape,self.adj_set[0].shape,self.h_gru[0].shape)
        # 静态图和动态图交互
        adj_static_list, adj_dynamic_list = self.graph_interation(self.adj_set,static_graph)
        # print("完成静态图和动态图的交互")
        # print(adj_static_list[0].shape,adj_dynamic_list[0].shape)
        # 静态图卷积
        h_s_gru = self.gru(inputs)
        h_st = self.static_Graph_Convolution(adj_static_list,h_s_gru)
        # print(h_st[0].shape)
        # print ("完成静态图卷积")
        # 动态图卷积
        h_dt = self.dynamic_Graph_Convolution(adj_dynamic_list,self.h_gru)
        # print ("完成动态图卷积")
        # print (h_dt[0].shape)
        y = self.prediction_Module(h_dt,h_st)
        return y.squeeze()