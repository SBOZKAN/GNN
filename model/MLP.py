import torch
from torch import nn
from torch.nn import Linear, Sigmoid
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool



# class MLP(nn.Module):
#     """
#     Multilayer Perceptron
#     """
    
#     def __init__(self,input_dim, hidden_channels, output_dim, dropout_rate=0.1):
#         super().__init()__()
#         self.layer=nn.Sequential(
#             nn.BatchNorm1d(input_dim),
#             nn.Linear(input_dim,hidden_channels),
#             nn.Dropout(dropout_rate)
#             nn.BatchNorm1d(hidden_channels),
#             nn.ReLU(),
#             nn.Linear(hidden_channels,1)
#         )
#     def forward(self,x):
#         x=self.layer(x)
#         return x

class MLP(nn.Module):
    """
    Multilayer Perceptron
    """
    
    def __init__(self,input_dim, 
                 hidden_channels, 
                 output_dim, 
                 dropout_rate=0.2, 
                 dropout=True, 
                 batch_norm=True, 
                 final_activation=None
                ):
        
        super().__init__()

        
        modules=[nn.Flatten()]
        prev_dim=input_dim
        for cur_dim in hidden_channels:
            modules.append(nn.Linear(prev_dim,cur_dim))
            if dropout:
                modules.append(nn.Dropout(dropout_rate))
            if batch_norm:
                modules.append(nn.BatchNorm1d(cur_dim))
            modules.append(nn.ReLU())
            prev_dim=cur_dim
            
        modules.append(nn.Linear(prev_dim,output_dim))
        
        if final_activation:
            modules.append(final_activation)   
        
        self.layer=nn.Sequential(*modules)    
        
            
    def forward(self,x):
        x=self.layer(x)
        return x.float()
    
    
class MLP_mask(nn.Module):
    """
    Multilayer Perceptron
    """
    
    def __init__(self,input_dim, 
                 hidden_channels, 
                 output_dim,
                 mask,
                 dropout_rate=0.2, 
                 dropout=True, 
                 batch_norm=True, 
                 final_activation=None
                ):
        
        super().__init__()

        self.mask=mask
        modules=[nn.Flatten()]
        prev_dim=input_dim
        for cur_dim in hidden_channels:
            modules.append(nn.Linear(prev_dim,cur_dim))
            if dropout:
                modules.append(nn.Dropout(dropout_rate))
            if batch_norm:
                modules.append(nn.BatchNorm1d(cur_dim))
            modules.append(nn.ReLU())
            prev_dim=cur_dim
            
        modules.append(nn.Linear(prev_dim,output_dim))
        
        if final_activation:
            modules.append(final_activation)   
        
        self.layer=nn.Sequential(*modules)    
        
            
    def forward(self,x):
        ######## mask #########
        batch, num_nodes, num_emb=x.shape
        x=x.view(batch,-1)
        mask=self.mask.repeat_interleave(num_emb)
        x=x*mask
        x=x.view(batch,num_nodes,num_emb)
        #################
        
        
        x=self.layer(x)
        return x.float()  
    
    
class MLP_norm(nn.Module):
    """
    Multilayer Perceptron
    """
    
    def __init__(self,input_dim, 
                 hidden_channels, 
                 output_dim, 
                 dropout_rate=0.2, 
                 dropout=True, 
                 batch_norm=True, 
                 final_activation=None
                ):
        
        super().__init__()

        
        modules=[nn.Flatten(),nn.BatchNorm1d(263*40)]
        prev_dim=input_dim
        for cur_dim in hidden_channels:
            modules.append(nn.Linear(prev_dim,cur_dim))
            if dropout:
                modules.append(nn.Dropout(dropout_rate))
            if batch_norm:
                modules.append(nn.BatchNorm1d(cur_dim))
            modules.append(nn.ReLU())
            prev_dim=cur_dim
            
        modules.append(nn.Linear(prev_dim,output_dim))
        
        if final_activation:
            modules.append(final_activation)   
        
        self.layer=nn.Sequential(*modules)    
        
            
    def forward(self,x):
        x=self.layer(x)
        return x.float()
    
    
class MLP_binary(nn.Module):
    """
    Multilayer Perceptron
    """
    
    def __init__(self,input_dim, 
                 hidden_channels, 
                 output_dim, 
                 dropout_rate=0.2, 
                 dropout=True, 
                 batch_norm=True, 
                 final_activation=None
                ):
        
        super().__init__()

        
        modules=[nn.Flatten()]
        prev_dim=input_dim
        for cur_dim in hidden_channels:
            modules.append(nn.Linear(prev_dim,cur_dim))
            if dropout:
                modules.append(nn.Dropout(dropout_rate))
            if batch_norm:
                modules.append(nn.BatchNorm1d(cur_dim))
            modules.append(nn.ReLU())
            prev_dim=cur_dim
            
        modules.append(nn.Linear(prev_dim,output_dim))
        
        if final_activation:
            modules.append(final_activation)   
        
        self.layer=nn.Sequential(*modules)    
        self.activation = Sigmoid()
            
    def forward(self,x):
        x=self.layer(x)
        x=self.activation(x)
        return x.float()