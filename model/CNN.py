import torch
from torch import nn
from torch.nn import Linear, Conv2d, Sigmoid
import torch.nn.functional as F
#from torch_geometric.nn import global_mean_pool


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
    
    
    
    
# class CNN(torch.nn.Module):
#     """
#     Add 1 MLP layer after GCN 
#     """
#     def __init__(self, 
#                  input_dim, 
#                  num_nodes,
#                  num_emb,
#                  hidden_channels_CNN,
#                  hidden_channels_MLP=100,
#                  kernel_size=3, 
#                  stride=(1,1),
#                  padding=0,
#                  output_dim=1,
#                  ):
        
#         super(CNN, self).__init__()
#         torch.manual_seed(2022)
#         self.num_conv=5
#         self.conv1 = Conv2d(input_dim, 
#                             hidden_channels_CNN[0],
#                             kernel_size=(kernel_size,num_emb),
#                             stride=(1,1),
#                             padding=padding,
#                             groups=1,
#                             )
        
#         self.conv2 = Conv2d(hidden_channels_CNN[0], 
#                             hidden_channels_CNN[1],
#                             kernel_size=(kernel_size,1),
#                             stride=(1,1),
#                             padding=padding,
#                             groups=1,
#                             )
#         self.conv3 = Conv2d(hidden_channels_CNN[1], 
#                             hidden_channels_CNN[2],
#                             kernel_size=(kernel_size,1),
#                             stride=(1,1),
#                             padding=padding,
#                             groups=1,
#                             )
#         self.conv4 = Conv2d(hidden_channels_CNN[2], 
#                             hidden_channels_CNN[3],
#                             kernel_size=(kernel_size,1),
#                             stride=(1,1),
#                             padding=padding,
#                             groups=1,
#                             )
#         self.conv5 = Conv2d(hidden_channels_CNN[3], 
#                             hidden_channels_CNN[4],
#                             kernel_size=(kernel_size,1),
#                             stride=(1,1),
#                             padding=padding,
#                             groups=1,
#                             )
#         #self.conv1 = GCNConv(input_dim, hidden_channels_GNN)
#         #self.conv1 = GCNConv(input_dim, hidden_channels_GNN)
#         self.flat=nn.Flatten()
#         self.lin1=Linear(hidden_channels_CNN[-1]*(num_nodes-self.num_conv*(kernel_size-1)), hidden_channels_MLP)
#         self.drop=nn.Dropout()
#         self.lin2=Linear(hidden_channels_MLP, output_dim)
        
#     def forward(self, x):
#         # 1. Add extra dim for input 
#         x=x.unsqueeze(1)
#         #print(x.shape)
        
#         # 2. Apply CNN
#         x = self.conv1(x)
#         x=x.relu()
#         #print(x.shape)
        
#         x = self.conv2(x)
#         x=x.relu()
#         #print(x.shape)
        
#         x = self.conv3(x)
#         x=x.relu()
#         #print(x.shape)
        
#         x = self.conv4(x)
#         x=x.relu()
#         #print(x.shape)
        
#         x = self.conv5(x)
#         x=x.relu()
#         #print(x.shape)
        
#         # 2. flatten nodes for each graph candidate 
#         x=self.flat(x)
#         #print(x.shape)
        
#         # 3. Apply a hidden_layer
#         x=self.lin1(x)
#         x=x.relu()
#         x=self.drop(x)
        
        
#         # 4. Apply a final output layer
#         x=self.lin2(x)
        
#         return x.float()
    
    
class CNN(torch.nn.Module):
    """
    Add 1 MLP layer after GCN 
    """
    def __init__(self, 
                 input_dim, 
                 num_nodes,
                 num_emb,
                 hidden_channels_CNN,
                 hidden_channels_MLP=100,
                 dropout_rate=0.2,
                 kernel_size=3, 
                 stride=(1,1),
                 padding=0,
                 output_dim=1,
                 ):
        
        super(CNN, self).__init__()
        torch.manual_seed(2022)
        self.num_conv=len(hidden_channels_CNN)
        self.conv1 = Conv2d(input_dim, 
                            hidden_channels_CNN[0],
                            kernel_size=(kernel_size,num_emb),
                            stride=stride,
                            padding=padding,
                            groups=1,
                            )
        for i in range(1, len(hidden_channels_CNN)):
            dim1=hidden_channels_CNN[i-1]
            dim2=hidden_channels_CNN[i]
            
            exec('self.conv{} = Conv2d({},{},kernel_size=(kernel_size,1),stride=stride,padding=padding,groups=1,)'.format(i+1,dim1,dim2))
        
        #self.conv1 = GCNConv(input_dim, hidden_channels_GNN)
        #self.conv1 = GCNConv(input_dim, hidden_channels_GNN)
        self.flat=nn.Flatten()
        
        new_nodes=num_nodes
        for i in range(self.num_conv):
            new_nodes=int((new_nodes-kernel_size)/stride[0])+1
        
        #self.lin1=Linear(hidden_channels_CNN[-1]*(num_nodes-self.num_conv*(kernel_size-1)), hidden_channels_MLP)
        self.lin1=Linear(hidden_channels_CNN[-1]*new_nodes, hidden_channels_MLP)
        self.drop=nn.Dropout(dropout_rate)
        self.lin2=Linear(hidden_channels_MLP, output_dim)
        
    def forward(self, x):
        # 1. Add extra dim for input 
        #print(x.shape)
        x=x.unsqueeze(1)
        #print(x.shape)
        
        # 2. Apply CNN
        for i in range(1, self.num_conv+1):
            x = eval('self.conv{}'.format(i))(x)
            x=x.relu()  
            #print(x.shape)
        
        # 2. flatten nodes for each graph candidate 
        x=self.flat(x)
        #print(x.shape)
        
        # 3. Apply a hidden_layer
        x=self.lin1(x)
        x=x.relu()
        x=self.drop(x)
        
        
        # 4. Apply a final output layer
        x=self.lin2(x)
        
        return x.float()

class CNN_binary(torch.nn.Module):
    """
    Add 1 MLP layer after GCN 
    """
    def __init__(self, 
                 input_dim, 
                 num_nodes,
                 num_emb,
                 hidden_channels_CNN,
                 hidden_channels_MLP=100,
                 dropout_rate=0.2,
                 kernel_size=3, 
                 stride=(1,1),
                 padding=0,
                 output_dim=1,
                 ):
        
        super(CNN_binary, self).__init__()
        torch.manual_seed(2022)
        self.num_conv=len(hidden_channels_CNN)
        self.conv1 = Conv2d(input_dim, 
                            hidden_channels_CNN[0],
                            kernel_size=(kernel_size,num_emb),
                            stride=stride,
                            padding=padding,
                            groups=1,
                            )
        for i in range(1, len(hidden_channels_CNN)):
            dim1=hidden_channels_CNN[i-1]
            dim2=hidden_channels_CNN[i]
            
            exec('self.conv{} = Conv2d({},{},kernel_size=(kernel_size,1),stride=stride,padding=padding,groups=1,)'.format(i+1,dim1,dim2))
        
        #self.conv1 = GCNConv(input_dim, hidden_channels_GNN)
        #self.conv1 = GCNConv(input_dim, hidden_channels_GNN)
        self.flat=nn.Flatten()
        
        new_nodes=num_nodes
        for i in range(self.num_conv):
            new_nodes=int((new_nodes-kernel_size)/stride[0])+1
        
        #self.lin1=Linear(hidden_channels_CNN[-1]*(num_nodes-self.num_conv*(kernel_size-1)), hidden_channels_MLP)
        self.lin1=Linear(hidden_channels_CNN[-1]*new_nodes, hidden_channels_MLP)
        self.drop=nn.Dropout(dropout_rate)
        self.lin2=Linear(hidden_channels_MLP, output_dim)
        self.activation = Sigmoid()
        
        
    def forward(self, x):
        # 1. Add extra dim for input 
        #print(x.shape)
        x=x.unsqueeze(1)
        #print(x.shape)
        
        # 2. Apply CNN
        for i in range(1, self.num_conv+1):
            x = eval('self.conv{}'.format(i))(x)
            x=x.relu()  
            #print(x.shape)
        
        # 2. flatten nodes for each graph candidate 
        x=self.flat(x)
        #print(x.shape)
        
        # 3. Apply a hidden_layer
        x=self.lin1(x)
        x=x.relu()
        x=self.drop(x)
        
        
        # 4. Apply a final output layer
        x=self.lin2(x)
        
        x=self.activation(x)
        return x.float()
        