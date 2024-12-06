import torch
from torch import nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.nn import global_mean_pool


# class GCN(torch.nn.Module):
#     def __init__(self, input_dim, hidden_channels, output_dim):
#         super(GCN, self).__init__()
#         torch.manual_seed(2022)
        
#         self.conv1 = GCNConv(input_dim, hidden_channels)
#         self.flat  = nn.Flatten()
#         self.lin=Linear(hidden_channels, output_dim)
        
#     def forward(self, x, edge_index, batch):
#         # 1. Obtain node embeddings 
#         x = self.conv1(x, edge_index)
#         x=x.relu()
#         print(x.shape)
#         test=x.view(batch[-1]+1, -1, x.shape[-1])
#         print(test.shape)
#         test=self.flat(test)
#         print(test.shape)
        
#         # 2. Readout layer
#         x = global_mean_pool(x, batch)
#         #print(x.shape)
        
#         # 3. Apply a final output
#         x=self.lin(x)
#         #print(x.shape)
#         return x.float()
    
    
    
    
# class GCN(torch.nn.Module):
#     """
#     Add 1 MLP layer after GCN 
#     """
#     def __init__(self, input_dim, num_node, hidden_channels_GNN, hidden_channels_MLP, output_dim):
#         super(GCN, self).__init__()
#         torch.manual_seed(2022)
#         self.conv1 = GCNConv(input_dim, hidden_channels_GNN)
#         self.flat=nn.Flatten()
#         self.lin1=Linear(hidden_channels_GNN*num_node, hidden_channels_MLP)
#         self.drop=nn.Dropout()
#         self.lin2=Linear(hidden_channels_MLP, output_dim)
        
#     def forward(self, x, edge_index, batch):
#         # 1. Obtain node embeddings 
#         x = self.conv1(x, edge_index)
#         x=x.relu()
#         x=x.view(batch[-1]+1, -1, x.shape[-1])
        
        
#         # 2. flatten nodes for each graph candidate 
#         x=self.flat(x)
        
        
#         # 3. Apply a hidden_layer
#         x=self.lin1(x)
#         x=x.relu()
#         x=self.drop(x)
        
        
#         # 4. Apply a final output layer
#         x=self.lin2(x)
        
#         return x.float()
    
    
# class GCN_3(torch.nn.Module):
#     """
#     Add 1 MLP layer after GCN 
#     """
#     def __init__(self, input_dim, num_node, hidden_channels_GNN, hidden_channels_MLP, output_dim):
#         super(GCN_3, self).__init__()
#         torch.manual_seed(2022)
#         self.conv1 = GCNConv(input_dim, hidden_channels_GNN[0])
#         self.conv2 = GCNConv(hidden_channels_GNN[0], hidden_channels_GNN[1])
#         self.conv3 = GCNConv(hidden_channels_GNN[1], hidden_channels_GNN[2])
        
#         self.flat=nn.Flatten()
#         self.lin1=Linear(hidden_channels_GNN[-1]*num_node, hidden_channels_MLP)
#         self.drop=nn.Dropout()
#         self.lin2=Linear(hidden_channels_MLP, output_dim)
        
#     def forward(self, x, edge_index, batch):
#         # 1. Obtain node embeddings 
#         x = self.conv1(x, edge_index)
#         x=x.relu()
        
#         x = self.conv2(x, edge_index)
#         x=x.relu()
        
#         x = self.conv3(x, edge_index)
#         x=x.relu()
        
        
#         x=x.view(batch[-1]+1, -1, x.shape[-1])
        
        
#         # 2. flatten nodes for each graph candidate 
#         x=self.flat(x)
        
        
#         # 3. Apply a hidden_layer
#         x=self.lin1(x)
#         x=x.relu()
#         x=self.drop(x)
        
        
#         # 4. Apply a final output layer
#         x=self.lin2(x)
        
#         return x.float()

class GCN(torch.nn.Module):
    """
    Add 1 MLP layer after GCN, generalized GCN version
    """
    def __init__(self, 
                 input_dim, 
                 num_node, 
                 hidden_channels_GNN, 
                 hidden_channels_MLP,
                 output_dim,
                 dropout_rate=0.2,):
        
        super(GCN, self).__init__()
        torch.manual_seed(2022)
#         self.conv1 = GCNConv(input_dim, hidden_channels_GNN[0])
#         self.conv2 = GCNConv(hidden_channels_GNN[0], hidden_channels_GNN[1])
#         self.conv3 = GCNConv(hidden_channels_GNN[1], hidden_channels_GNN[2])
#         self.conv=[]
#         for i in range(1, len(hidden_channels_GNN)):
#             temp=GCNConv(hidden_channels_GNN[i-1], hidden_channels_GNN[i])
#             self.conv.append(temp)
        self.num_conv=len(hidden_channels_GNN)
    
        self.conv1=GCNConv(input_dim, hidden_channels_GNN[0])
        for i in range(1, len(hidden_channels_GNN)):
            dim1=hidden_channels_GNN[i-1]
            dim2=hidden_channels_GNN[i]
            
            exec('self.conv{} = GCNConv({},{})'.format(i+1,dim1,dim2))
        #exec('var{} = {}'.format(i, i))
    
        #self.conv1=GCNConv(hidden_channels_GNN[0], hidden_channels_GNN[1])
        self.flat=nn.Flatten()
        self.lin1=Linear(hidden_channels_GNN[-1]*num_node, hidden_channels_MLP)
        self.drop=nn.Dropout(dropout_rate)
        self.lin2=Linear(hidden_channels_MLP, output_dim)
        
    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        for i in range(1, self.num_conv+1):
            x = eval('self.conv{}'.format(i))(x, edge_index)
            x=x.relu()      
        
        x=x.view(batch[-1]+1, -1, x.shape[-1])
        
        
        # 2. flatten nodes for each graph candidate 
        x=self.flat(x)
        
        
        # 3. Apply a hidden_layer
        x=self.lin1(x)
        x=x.relu()
        x=self.drop(x)
        
        
        # 4. Apply a final output layer
        x=self.lin2(x)
        
        return x.float()
    
    
class SAGE_GCN(torch.nn.Module):
    """
    Add 1 MLP layer after SAGEGCN, most generalized SAGEGCN version 
    """
    def __init__(self, 
                 input_dim, 
                 num_node, 
                 hidden_channels_GNN, 
                 hidden_channels_MLP, 
                 output_dim,
                 dropout_rate=0.2,):
        
        super(SAGE_GCN, self).__init__()
        torch.manual_seed(2022)
#         self.conv1 = GCNConv(input_dim, hidden_channels_GNN[0])
#         self.conv2 = GCNConv(hidden_channels_GNN[0], hidden_channels_GNN[1])
#         self.conv3 = GCNConv(hidden_channels_GNN[1], hidden_channels_GNN[2])
#         self.conv=[]
#         for i in range(1, len(hidden_channels_GNN)):
#             temp=GCNConv(hidden_channels_GNN[i-1], hidden_channels_GNN[i])
#             self.conv.append(temp)
        self.hidden_channels_GNN=hidden_channels_GNN
        self.conv1=SAGEConv(input_dim, hidden_channels_GNN[0])
        for i in range(1, len(hidden_channels_GNN)):
            dim1=hidden_channels_GNN[i-1]
            dim2=hidden_channels_GNN[i]
            
            exec('self.conv{} = SAGEConv({},{})'.format(i+1,dim1,dim2))
        #exec('var{} = {}'.format(i, i))
    
        #self.conv1=GCNConv(hidden_channels_GNN[0], hidden_channels_GNN[1])
        self.flat=nn.Flatten()
        self.lin1=Linear(hidden_channels_GNN[-1]*num_node, hidden_channels_MLP)
        self.drop=nn.Dropout(dropout_rate)
        self.lin2=Linear(hidden_channels_MLP, output_dim)
        
    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        for i in range(1, len(self.hidden_channels_GNN)+1):
            x = eval('self.conv{}'.format(i))(x, edge_index)
            x=x.relu()      
        
        x=x.view(batch[-1]+1, -1, x.shape[-1])
        
        
        # 2. flatten nodes for each graph candidate 
        x=self.flat(x)
        
        
        # 3. Apply a hidden_layer
        x=self.lin1(x)
        x=x.relu()
        x=self.drop(x)
        
        
        # 4. Apply a final output layer
        x=self.lin2(x)
        
        return x.float()    

# class SAGE_GCN(torch.nn.Module):
#     """
#     Add 1 MLP layer after GCN 
#     """
#     def __init__(self, input_dim, num_node, hidden_channels_GNN, hidden_channels_MLP, output_dim):
#         super(SAGE_GCN, self).__init__()
#         torch.manual_seed(2022)
#         self.conv1 = SAGEConv(input_dim, hidden_channels_GNN[0])
#         self.conv2 = SAGEConv(hidden_channels_GNN[0], hidden_channels_GNN[1])
#         self.conv3 = SAGEConv(hidden_channels_GNN[1], hidden_channels_GNN[2])
        
#         self.flat=nn.Flatten()
#         self.lin1=Linear(hidden_channels_GNN[-1]*num_node, hidden_channels_MLP)
#         self.drop=nn.Dropout()
#         self.lin2=Linear(hidden_channels_MLP, output_dim)
        
#     def forward(self, x, edge_index, batch):
#         # 1. Obtain node embeddings 
#         x = self.conv1(x, edge_index)
#         x=x.relu()
        
#         x = self.conv2(x, edge_index)
#         x=x.relu()
        
#         x = self.conv3(x, edge_index)
#         x=x.relu()
        
        
#         x=x.view(batch[-1]+1, -1, x.shape[-1])
        
        
#         # 2. flatten nodes for each graph candidate 
#         x=self.flat(x)
        
        
#         # 3. Apply a hidden_layer
#         x=self.lin1(x)
#         x=x.relu()
#         x=self.drop(x)
        
        
#         # 4. Apply a final output layer
#         x=self.lin2(x)
        
#         return x.float()


