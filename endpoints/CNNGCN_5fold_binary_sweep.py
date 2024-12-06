import os
from os.path import abspath, join, isdir
import sys
sys.path.append('/scratch/nhuynh9/GNN_PyTorch/')  
if not isdir("notebooks"):
    # if there's a "notebooks" directory in the cwd, we've already set the cwd so no need to do it again
    os.chdir("..")
from utils.utils import read_pdb, create_graph_pdb, create_graph_DCI, create_graph_DCIasym
import pandas as pd    
import numpy as np
import torch
from model.GNN import GCN, SAGE_GCN
from model.MLP import MLP
from model.CNNGCN import CNNGCN, CNNGCN_binary
from backend.Jin_PyTorchTrainer import PytorchTrainer_MLP, PytorchTrainer_GNN
from torch_geometric.data import Data
from torch.utils.data import TensorDataset
from utils.utils import split_data
from encode import encode as enc

# very important, fit graph data, also fit regular x, y data, using torch.utils.data.DataLoader intead will not read graph data
from torch_geometric.loader import DataLoader
import argparse


def main():
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--protein', type=str)
    parser.add_argument('-n_type','--neighbour_type', type=str) 
    parser.add_argument('-cut_off','--cut_off', type=str)
    parser.add_argument('-type_c','--type_c', nargs='?', const='abs', type=str)
    parser.add_argument('-high_couple','--high_couple', nargs='?', const=0.6, type=float)
    
    parser.add_argument('-b','--batch_size', type=int)
    parser.add_argument('-e','--encoder', type=str)
    parser.add_argument('-lc','--c_layer', type=int)
    parser.add_argument('-x','--c_hidden_c', type=int)
    parser.add_argument('-k','--kernel', type=int)           # kernel size
    parser.add_argument('-s','--stride', type=int)
    parser.add_argument('-gl','--g_layer', type=int)
    parser.add_argument('-y','--g_hidden_c', type=int)
    
    
    args = parser.parse_args()
    
    protein=args.protein
    n_type=args.neighbour_type
    cut_off=float(args.cut_off)
    c_type=args.type_c  # only for dci asymmetry ['abs','pos','neg']
    high_couple=args.high_couple
    
    batch_size = args.batch_size
    encoder    = args.encoder
    c_layer      = args.c_layer
    c_hidden_c   = args.c_hidden_c
    kernel     = args.kernel
    stride     = args.stride
    g_layer      = args.g_layer
    g_hidden_c   = args.g_hidden_c

    
    # parameters in the training process
    encoder='one_hot,aa_index' # can be change when new encoder finished
    epochs=300
    lr=0.0001
    #batch_size=128
    early_stop=True
    patience=30
    save_history=True
    kfold=5
    device='cuda'
    hidden_channels_CNN=[c_hidden_c]*c_layer
    hidden_channels_GNN=[g_hidden_c]*g_layer
    threshold=0.05
    
    # set random seed to make it repeatable  
    torch.manual_seed(2022)
    # train and validation data path
    path='./data/{}/Processed_data/'.format(protein)
    # path to save model and all history file
    save_path='./results/{}/sweep_binary/CNNGCN_{}_{}_{}_{}_batch-{}_encoder-{}_clayer-{}_chidden_c-{}_kernel-{}_stride-{}_glayer-{}_ghidden_c-{}_threshold-{}/'.format(protein,n_type,cut_off,c_type,high_couple,batch_size,encoder,c_layer,c_hidden_c,kernel,stride,g_layer,g_hidden_c,threshold)
    # path where temporory checkpoint.py from early_stopping is saved. Very important, must be specified 
    # if multiple cross-validation job is running
    es_checkpoint_path='./results/{}/sweep_binary/CNNGCN_{}_{}_{}_{}_batch-{}_encoder-{}_clayer-{}_chidden_c-{}_kernel-{}_stride-{}_glayer-{}_ghidden_c-{}_threshold-{}/'.format(protein,n_type,cut_off,c_type,high_couple,batch_size,encoder,c_layer,c_hidden_c,kernel,stride,g_layer,g_hidden_c,threshold)
    
    if n_type in ['pctdci_asym', 'dci_asym']:
        save_path='./results/{}/sweep_binary/CNNGCN_{}_{}_{}_{}_batch-{}_encoder-{}_clayer-{}_chidden_c-{}_kernel-{}_stride-{}_glayer-{}_ghidden_c-{}_threshold-{}/'.format(protein,n_type,cut_off,c_type,high_couple,batch_size,encoder,c_layer,c_hidden_c,kernel,stride,g_layer,g_hidden_c,threshold)
        es_checkpoint_path='./results/{}/sweep_binary/CNNGCN_{}_{}_{}_{}_batch-{}_encoder-{}_clayer-{}_chidden_c-{}_kernel-{}_stride-{}_glayer-{}_ghidden_c-{}_threshold-{}/'.format(protein,n_type,cut_off,c_type,high_couple,batch_size,encoder,c_layer,c_hidden_c,kernel,stride,g_layer,g_hidden_c,threshold)
    
    # dict to save all history and performance 
    validation_loss_min={}
    validation_loss_history={}
    loss_history={}
    
    # create es_checkpoint_path if not exists
    if es_checkpoint_path:
        # create path if not exists
        if not os.path.exists(es_checkpoint_path):
            os.makedirs(es_checkpoint_path)
    
    # define graph info
    qmap={'pctdci': 'pctfdfi', 'dci': 'fdfi'}
    Res_ID, coor, wt=read_pdb('./data/{}/wt.pdb'.format(protein))
    #cut_off=5
    if n_type=='pdb':
        graph=create_graph_pdb(coor,cut_off,Res_ID,save_path=save_path,)
    elif n_type in ['pctdci', 'dci']:
        graph=create_graph_DCI(path='data/{}'.format(protein),
                               name='wt',
                               cutoff=cut_off,
                               Res_ID=Res_ID,
                               q=qmap[n_type],
                               inverse=True,
                               save_path=save_path,
                               )
    elif n_type in ['pctdci_asym','dci_asym']:
        graph=create_graph_DCIasym(path='data/{}'.format(protein),
                         name='wt', 
                         Res_ID=Res_ID, 
                         cutoff=cut_off, 
                         c_type=c_type,
                         q=qmap[n_type.split('_')[0]],
                         common_neighbour=None,
                         save_path=save_path,
                         high_couple=high_couple,
                        )
    else:
        graph=pd.read_csv(n_type)
        #save_path='./results/{}/5fold/CNNGCN_{}_{}_{}_{}_{}_{}/'.format(protein,hidden_channels_CNN,hidden_channels_GNN,n_type.split('/')[-1],cut_off,c_type,high_couple)
        #es_checkpoint_path='./results/{}/5fold/CNNGCN_{}_{}_{}_{}_{}_{}/'.format(protein,hidden_channels_CNN,hidden_channels_GNN,n_type.split('/')[-1],cut_off,c_type,high_couple)
    edge_index=torch.tensor(np.array([graph.source.values,graph.target.values]))
    edge_index=edge_index.to(device)
    

    for fold in range(kfold):

        print('==================================================================')
        print('fold {} start'.format(fold))
        print('==================================================================')
        
        # read raw X_train X_valid 5_fold data
        train=pd.read_csv('{}train_fold{}.csv'.format(path,fold), index_col=0)
        valid=pd.read_csv('{}valid_fold{}.csv'.format(path,fold), index_col=0)

        y_train=train.score.values
        y_valid=valid.score.values
        
        y_train[y_train>=threshold]=1
        y_train[y_train<threshold]=0
        y_valid[y_valid>=threshold]=1
        y_valid[y_valid<threshold]=0
        
        seq_train=train.loc[:,train.columns[3:]].values
        seq_train=[''.join(i) for i in seq_train]
        seq_valid=valid.loc[:,valid.columns[3:]].values
        seq_valid=[''.join(i) for i in seq_valid]

        
        X_train=enc.encode(encoding=encoder, char_seqs=list(seq_train))
        X_valid=enc.encode(encoding=encoder, char_seqs=list(seq_valid))
        
        # change numpy to tensor
        X_train=torch.Tensor(X_train)  
        y_train=torch.Tensor(y_train).unsqueeze(1)    
        X_valid=torch.Tensor(X_valid)    
        y_valid=torch.Tensor(y_valid).unsqueeze(1)    

        # create regular dataset
        train_dataset = TensorDataset(X_train,y_train)
        valid_dataset = TensorDataset(X_valid,y_valid)
        
        # data loader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        
        # define model, loss and optimizer
        kernel_size=3
        padding=(kernel_size//2,0)
        num_nodes=X_train.shape[1]
        num_emb=X_train.shape[-1]
        # define model, loss and optimizer
        model = CNNGCN_binary(input_dim=1, 
                    num_nodes=num_nodes,
                    num_emb=num_emb,
                    edge_index=edge_index,
                    hidden_channels_CNN=hidden_channels_CNN,
                    hidden_channels_GNN=hidden_channels_GNN,    
                    kernel_size=kernel_size,
                    padding=padding, # need to be adjusted by kernel_size
                    output_dim=1)
        
        
        optimizer=torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.BCELoss()
        
        # define trainer and fit model
        trainer=PytorchTrainer_MLP(optimizer,
                                   loss=criterion,
                                   epochs=epochs,
                                   device=device,
                                   )
        trainer.fit(model,
                    train_loader,
                    val_loader=valid_loader, 
                    save_history=True, 
                    early_stop=early_stop, 
                    patience=patience,
                    es_checkpoint=es_checkpoint_path+'checkpoint.pt',
                    )



        # save train loss history
        loss_history['fold_{}'.format(fold)]= trainer.loss_history
        # save validation loss history
        validation_loss_history['fold_{}'.format(fold)]= trainer.val_loss_history
        # save minimum validation loss
        validation_loss_min['fold_{}'.format(fold)]= trainer.evaluate(model,valid_loader)
        # save best model for validation
        #best_models['fold_{}'.format(fold)] = model

        ###############################
        # infer the test data
        ###############################
        test=pd.read_csv('data/{}/Processed_data/test.csv'.format(protein), index_col=0)
        y_test=test.score.values
        seq_test=test.loc[:,test.columns[3:]].values
        seq_test=[''.join(i) for i in seq_test]
        #if encoder in ['one_hot', 'aa_index', 'one_hot,aa_index']:
        X_test=enc.encode(encoding=encoder, char_seqs=list(seq_test))

        # change numpy to tensor    
        X_test=torch.Tensor(X_test)  
        y_test=torch.Tensor(y_test).unsqueeze(1)  
        # create regular dataset
        test_dataset = TensorDataset(X_test,y_test)
        # data loader
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False) # shuffle must be False 
        model.eval()
        if device=='cpu':                                                   
            device= torch.device('cpu')                                             
        else:                                                                   
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        y_pred=torch.Tensor([])
        y_pred=y_pred.to(device)
        for data in test_loader:
            x,y=data
            x=x.to(device)
            y=y.to(device)
            y=y.squeeze(1)
            out=model(x).squeeze(1)
            y_pred=torch.cat((y_pred,out))
        
        y_pred=y_pred.cpu().detach().numpy()
        pred_data=test.loc[:,['variant','num_variant','score']]
        pred_data['y_pred']=y_pred
        y_t=pred_data.score.values
        y_t[y_t>=threshold]=1
        y_t[y_t<threshold]=0
        pred_data['score_binary']=y_t
        
        # save all history data if save_path is specified
        if save_path:
            # create path if not exists
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # save loss history
            loss_history_df=pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in loss_history.items() ]))
            loss_history_df['epoch']=range(len(loss_history_df))
            pd.DataFrame(loss_history_df).to_csv('{}loss_history.csv'.format(save_path))

            # save validation loss history
            validation_loss_history_df=pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in validation_loss_history.items() ]))
            validation_loss_history_df['epoch']=range(len(validation_loss_history_df))
            pd.DataFrame(validation_loss_history_df).to_csv('{}validation_loss_history.csv'.format(save_path))

            # save minimum loss 
            validation_loss_min_df=pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in validation_loss_min.items() ]))
            pd.DataFrame(validation_loss_min_df).to_csv('{}validation_loss_min.csv'.format(save_path))

            # save best model
            #for k,model in best_models.items():
            torch.save(model, '{}best_model_{}.pt'.format(save_path,fold))
            
            # save the predict test data
            pred_data.to_csv('{}test_prediction_{}.csv'.format(save_path, fold))
            
if __name__ == '__main__':
    main()
