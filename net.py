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
        self.graph_interation = layers.graph_interation(self.k)
        self.static_Graph_Convolution = layers.Static_Graph_Convolution(beta, self.nodes)
        self.dynamic_Graph_Convolution = layers.Dynamic_Graph_Convolution()
    def forward(self,inputs):
        print("开始")
        x = torch.randint (0, self.nodes, size=(1, self.nodes)).squeeze ()
        # 获得静态图
        static_graph = self.static_graph_constructor(x)
        # 获得动态图和gru的隐状态
        self.adj_set,self.h_gru = self.dynamic_graph_constructor(inputs)
        # 静态图和动态图交互
        self.graph_interation(self.adj_set,static_graph)
        # 静态图卷积
        h_st = self.static_Graph_Convolution(static_graph,self.h_gru)
        print(h_st.shape)
        # 动态图卷积
        h_dt = self.dynamic_Graph_Convolution(self.adj_set[-1],self.h_gru)
        print (h_dt.shape)