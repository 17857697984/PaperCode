import layers
import torch.nn as nn
import torch

class Net(nn.Module):
    def __init__(self, device, horizon, nodes, seq_len, dim=1, alpha=0.5, k=1):
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
        self.static_graph_constructor = layers.static_graph_constructor(self.nodes, self.dim, self.device)
        self.dynamic_graph_constructor = layers.dynamic_graph_constructor(self.horizon, self.nodes, self.seq_len)
    def forward(self,inputs):
        x = torch.randint (0, self.seq_len, size=(1, self.seq_len)).squeeze ()
        static_graph = self.static_graph_constructor(x)
        self.adj_set,self.h_gru = self.dynamic_graph_constructor(inputs)
