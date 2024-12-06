import torch
from torch import nn
from torch.nn import Linear, Conv2d, Sigmoid
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Batch
from torch_geometric.data import Data



class CNNGCN(torch.nn.Module):
    """
    Add 1 MLP layer after GCN 
    """
    def __init__(self, 
                 input_dim, 
                 num_nodes,
                 num_emb,
                 edge_index,
                 hidden_channels_CNN=[128],
                 hidden_channels_GNN=[128],
                 hidden_channels_MLP=100,
                 dropout_rate=0.2,
                 kernel_size=3, 
                 stride=(1,1),
                 padding=0,
                 output_dim=1,
                 ):
        
        super(CNNGCN, self).__init__()
        torch.manual_seed(2022)
        self.num_conv=len(hidden_channels_CNN)
        self.num_gconv=len(hidden_channels_GNN)
        self.edge_index=edge_index
        
        
        # initialize CNN layers
        self.conv1 = Conv2d(input_dim, 
                            hidden_channels_CNN[0],
                            kernel_size=(kernel_size,num_emb),
                            stride=(1,1),
                            padding=padding,
                            groups=1,
                            )
        for i in range(1, len(hidden_channels_CNN)):
            dim1=hidden_channels_CNN[i-1]
            dim2=hidden_channels_CNN[i]
            exec('self.conv{} = Conv2d({},{},kernel_size=(kernel_size,1),stride=(1,1),padding=padding,groups=1,)'.format(i+1,dim1,dim2))
        
        # initialize GCN layers
        self.gconv1=GCNConv(hidden_channels_CNN[-1], 
                           hidden_channels_GNN[0])
        for i in range(1, len(hidden_channels_GNN)):
            dim1=hidden_channels_GNN[i-1]
            dim2=hidden_channels_GNN[i]
            exec('self.gconv{} = GCNConv({},{})'.format(i+1,dim1,dim2))
            
        
        self.flat=nn.Flatten()
        self.lin1=Linear(hidden_channels_GNN[-1]*(num_nodes), hidden_channels_MLP)
        self.drop=nn.Dropout(dropout_rate)
        self.lin2=Linear(hidden_channels_MLP, output_dim)
        
        
    def forward(self, x):
        # 1. Add extra dim for input 
        #print(x.shape)
        batch=x.shape[0]
        x=x.unsqueeze(1)
        #print(x.shape)
        
        # 2. Apply CNN output [num_batch, hidden_channel, protein_length-kernel_size+1, emb_dim - emb_dim + 1 (1 dim)]
        for i in range(1, self.num_conv+1):
            x = eval('self.conv{}'.format(i))(x)
            x = x.relu()  
            #print(x.shape)
          
        # 3. Apply GCN
        x = x.squeeze(3)  # drop the last dimension 1
        #print(x.shape)
        x = x.permute(0,2,1)
        #print(x.shape)
        
        G=[]
        for i in x:
            g=Data(x=i,edge_index=self.edge_index)
            G.append(g)
        G=Batch.from_data_list(G)
        
        x=G.x # very important, otherwise the layer always train the x right after CNN, so multilayer GNN equals to only one layer
        for i in range(1, self.num_gconv+1):
            x = eval('self.gconv{}'.format(i))(x, G.edge_index)
            x = x.relu() 
        
        #print(x.shape)
        x = x.view((batch, -1, x.shape[-1]))
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

class CNNGCN_binary(torch.nn.Module):
    """
    Add 1 MLP layer after GCN 
    """
    def __init__(self, 
                 input_dim, 
                 num_nodes,
                 num_emb,
                 edge_index,
                 hidden_channels_CNN=[128],
                 hidden_channels_GNN=[128],
                 hidden_channels_MLP=100,
                 dropout_rate=0.2,
                 kernel_size=3, 
                 stride=(1,1),
                 padding=0,
                 output_dim=1,
                 ):
        
        super(CNNGCN_binary, self).__init__()
        torch.manual_seed(2022)
        self.num_conv=len(hidden_channels_CNN)
        self.num_gconv=len(hidden_channels_GNN)
        self.edge_index=edge_index
        
        
        # initialize CNN layers
        self.conv1 = Conv2d(input_dim, 
                            hidden_channels_CNN[0],
                            kernel_size=(kernel_size,num_emb),
                            stride=(1,1),
                            padding=padding,
                            groups=1,
                            )
        for i in range(1, len(hidden_channels_CNN)):
            dim1=hidden_channels_CNN[i-1]
            dim2=hidden_channels_CNN[i]
            exec('self.conv{} = Conv2d({},{},kernel_size=(kernel_size,1),stride=(1,1),padding=padding,groups=1,)'.format(i+1,dim1,dim2))
        
        # initialize GCN layers
        self.gconv1=GCNConv(hidden_channels_CNN[-1], 
                           hidden_channels_GNN[0])
        for i in range(1, len(hidden_channels_GNN)):
            dim1=hidden_channels_GNN[i-1]
            dim2=hidden_channels_GNN[i]
            exec('self.gconv{} = GCNConv({},{})'.format(i+1,dim1,dim2))
            
        
        self.flat=nn.Flatten()
        self.lin1=Linear(hidden_channels_GNN[-1]*(num_nodes), hidden_channels_MLP)
        self.drop=nn.Dropout(dropout_rate)
        self.lin2=Linear(hidden_channels_MLP, output_dim)
        self.activation = Sigmoid()
        
    def forward(self, x):
        # 1. Add extra dim for input 
        #print(x.shape)
        batch=x.shape[0]
        x=x.unsqueeze(1)
        #print(x.shape)
        
        # 2. Apply CNN output [num_batch, hidden_channel, protein_length-kernel_size+1, emb_dim - emb_dim + 1 (1 dim)]
        for i in range(1, self.num_conv+1):
            x = eval('self.conv{}'.format(i))(x)
            x = x.relu()  
            #print(x.shape)
          
        # 3. Apply GCN
        x = x.squeeze(3)  # drop the last dimension 1
        #print(x.shape)
        x = x.permute(0,2,1)
        #print(x.shape)
        
        G=[]
        for i in x:
            g=Data(x=i,edge_index=self.edge_index)
            G.append(g)
        G=Batch.from_data_list(G)
        
        x=G.x # very important, otherwise the layer always train the x right after CNN, so multilayer GNN equals to only one layer
        for i in range(1, self.num_gconv+1):
            x = eval('self.gconv{}'.format(i))(x, G.edge_index)
            x = x.relu() 
        
        #print(x.shape)
        x = x.view((batch, -1, x.shape[-1]))
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
    
