import os
from os.path import abspath, join, isdir
import sys
sys.path.append('/scratch/nhuynh9/GNN_PyTorch/')  
if not isdir("notebooks"):
    # if there's a "notebooks" directory in the cwd, we've already set the cwd so no need to do it again
    os.chdir("..")
from utils.utils import read_pdb, create_graph_pdb, create_graph_DCI
import pandas as pd    
import numpy as np
import torch
from model.GNN import GCN, SAGE_GCN
from model.CNN import CNN, CNN_binary
from model.MLP import MLP
from backend.Jin_PyTorchTrainer import PytorchTrainer_MLP, PytorchTrainer_GNN
from torch_geometric.data import Data
from torch.utils.data import TensorDataset
from utils.utils import split_data
import pandas as pd
from encode import encode as enc

# very important, fit graph data, also fit regular x, y data, using torch.utils.data.DataLoader intead will not read graph data
from torch_geometric.loader import DataLoader
import argparse


def main():
    
    
    parser = argparse.ArgumentParser()
    #parser.add_argument('-n_type','--neighbour_type', type=str) 
    parser.add_argument('-p','--protein', type=str)
    parser.add_argument('-b','--batch_size', type=int)
    parser.add_argument('-e','--encoder', type=str)
    parser.add_argument('-l','--layer', type=int)            # num of layer
    parser.add_argument('-hidden','--hidden_c', type=int)         # hidden channels of CNN layer
    parser.add_argument('-k','--kernel', type=int)           # kernel size
    parser.add_argument('-s','--stride', type=int)           # stride
    parser.add_argument('-t','--threshold', type=float)
    
    args = parser.parse_args()
    
    protein    = args.protein
    batch_size = args.batch_size
    encoder    = args.encoder
    layer      = args.layer
    hidden_c   = args.hidden_c
    kernel     = args.kernel
    stride     = args.stride
    threshold    = args.threshold
    
    # parameters in the training process
    #encoder='one_hot,aa_index' # can be change when new encoder finished
    epochs=500
    lr=0.0001
    #batch_size=64
    early_stop=True
    patience=50
    save_history=True
    kfold=5
    device='cuda'
    hidden_channels=[100]
    hidden_channels_CNN=[hidden_c]*layer

    
    
    # set random seed to make it repeatable  
    torch.manual_seed(2022)
    # train and validation data path
    path='./data/{}/Processed_data/'.format(protein)
    # path to save model and all history file
    save_path='./results/{}/sweep_binary/CNN_batch-{}_encoder-{}_layer-{}_hidden_c-{}_kernel-{}_stride-{}_threshold-{}/'.format(protein,batch_size,encoder,layer,hidden_c,kernel,stride,threshold)
    # path where temporory checkpoint.py from early_stopping is saved. Very important, must be specified 
    # if multiple cross-validation job is running
    es_checkpoint_path='./results/{}/sweep_binary/CNN_batch-{}_encoder-{}_layer-{}_hidden_c-{}_kernel-{}_stride-{}_threshold-{}/'.format(protein,batch_size,encoder,layer,hidden_c,kernel,stride,threshold)
    # dict to save all history and performance 
    validation_loss_min={}
    validation_loss_history={}
    loss_history={}

    # create es_checkpoint_path if not exists
    if es_checkpoint_path:
        # create path if not exists
        if not os.path.exists(es_checkpoint_path):
            os.makedirs(es_checkpoint_path)


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

        #if encoder in ['one_hot', 'aa_index', 'one_hot,aa_index']:
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
        num_nodes=X_train.shape[-2]
        num_emb=X_train.shape[-1]
        # define model, loss and optimizer
        model = CNN_binary(input_dim=1, 
                    num_nodes=num_nodes,
                    num_emb=num_emb,
                    hidden_channels_CNN=hidden_channels_CNN,
                    kernel_size=kernel,
                    stride=(stride,1),
                    padding='valid',
                    output_dim=1,
                    )
        optimizer=torch.optim.Adam(model.parameters(), lr=lr)
        #criterion = torch.nn.MSELoss()
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
            #model.to('cpu')
            #model.eval()
            torch.save(model, '{}best_model_{}.pt'.format(save_path,fold))
            
            # save the predict test data
            pred_data.to_csv('{}test_prediction_{}.csv'.format(save_path, fold))
            
if __name__ == '__main__':
    main()
