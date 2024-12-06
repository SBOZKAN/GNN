import torch
from torch_geometric.loader import DataLoader
import numpy as np
from encode import encode as enc

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss



class PytorchTrainer_MLP:
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 loss: torch.nn.modules.loss,
                 epochs=1,
                 device='cpu',
                ):
        self.optimizer = optimizer
        self.loss = loss
        self.epochs = epochs
        self.loss_history = []
        self.val_loss_history=[]
        self.val_loss_min=np.Inf
        #self.device=device 
        if device=='cpu':                                                   
            self.device= torch.device('cpu')                                             
        else:                                                                   
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def train(self,
              model: torch.nn.Module,
              train_loader: DataLoader,):
        
#         if self.device=='cpu':                                                   
#             device='cpu'                                                        
#         else:                                                                   
#             device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        model.train()
        model.to(self.device)

        for data in train_loader:  # Iterate in batches over the training dataset.
            x,y = data
            x=x.to(self.device)
            y=y.to(self.device)
            out = model(x)  # Perform a single forward pass.
            loss = self.loss(out, y)  # Compute the loss.
            loss.backward()  # Derive gradients.
            self.optimizer.step()  # Update parameters based on gradients.
            self.optimizer.zero_grad()  # Clear gradients.
    
    def evaluate(self,
                 model,
                 data_loader: DataLoader):
        
        model.eval()
        loss_total=0
        num=0
        for data in data_loader:
            x,y = data
            x=x.to(self.device)
            y=y.to(self.device)
            out = model(x)
            loss = self.loss(out,y).item()
            loss_total+=loss*len(data)
            num+=len(data)
            
        return loss_total/num   
        
    
    def fit(self,
            model: torch.nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader=None,
            test_loader: DataLoader=None,
            save_history=False,
            early_stop=False,
            patience=7,
            es_checkpoint='checkpoint.pt',
           ):
        if early_stop:
            early_stopping = EarlyStopping(patience=patience, verbose=True, path=es_checkpoint)
        
        for epoch in range(self.epochs):
            print("epoch: ", epoch)
            self.train(model,train_loader)
            
            train_loss=self.evaluate(model,train_loader)
            val_loss=self.evaluate(model,val_loader)
            if save_history:
                self.loss_history.append(train_loss)
                if val_loader:
                    self.val_loss_history.append(val_loss)
                        
            if early_stop:
                early_stopping(val_loss, model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
                    
        if early_stop:
            # load the last checkpoint with the best model
            model.load_state_dict(torch.load(early_stopping.path)) 
        
        self.val_loss_min=early_stopping.val_loss_min        
        
        
class PytorchTrainer_MLP_Encoding:
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 loss: torch.nn.modules.loss,
                 encoder=None,
                 epochs=1,
                 device='cpu',
                ):
        self.optimizer = optimizer
        self.loss = loss
        self.encoder = encoder
        self.epochs = epochs
        self.loss_history = []
        self.val_loss_history=[]
        self.val_loss_min=np.Inf
        #self.device=device 
        if device=='cpu':                                                   
            self.device= torch.device('cpu')                                             
        else:                                                                   
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def train(self,
              model: torch.nn.Module,
              train_loader: DataLoader,):
        
#         if self.device=='cpu':                                                   
#             device='cpu'                                                        
#         else:                                                                   
#             device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        model.train()
        model.to(self.device)

        for data in train_loader:  # Iterate in batches over the training dataset.
            x,y = data
            if self.encoder:
                x=enc.encode(encoding=self.encoder, char_seqs=list(x))    
            x=torch.Tensor(x)
            y=torch.Tensor(y)
            x=x.to(self.device)
            y=y.to(self.device)
            out = model(x)  # Perform a single forward pass.
            loss = self.loss(out, y)  # Compute the loss.
            loss.backward()  # Derive gradients.
            self.optimizer.step()  # Update parameters based on gradients.
            self.optimizer.zero_grad()  # Clear gradients.
    
    def evaluate(self,
                 model,
                 data_loader: DataLoader):
        
        model.eval()
        loss_total=0
        num=0
        for data in data_loader:
            x,y = data
            if self.encoder:
                x=enc.encode(encoding=self.encoder, char_seqs=list(x))
            x=torch.Tensor(x)
            y=torch.Tensor(y)
            x=x.to(self.device)
            y=y.to(self.device)
            out = model(x)
            loss = self.loss(out,y).item()
            loss_total+=loss*len(data)
            num+=len(data)
            
        return loss_total/num   
        
    
    def fit(self,
            model: torch.nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader=None,
            test_loader: DataLoader=None,
            save_history=False,
            early_stop=False,
            patience=7,
            es_checkpoint='checkpoint.pt',
           ):
        if early_stop:
            early_stopping = EarlyStopping(patience=patience, verbose=True, path=es_checkpoint)
        
        for epoch in range(self.epochs):
            print("epoch: ", epoch)
            self.train(model,train_loader)
            
            train_loss=self.evaluate(model,train_loader)
            val_loss=self.evaluate(model,val_loader)
            if save_history:
                self.loss_history.append(train_loss)
                if val_loader:
                    self.val_loss_history.append(val_loss)
                        
            if early_stop:
                early_stopping(val_loss, model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
        
        print('state_1')
        if early_stop:
            # load the last checkpoint with the best model
            model.load_state_dict(torch.load(early_stopping.path)) 
            print('state_2')
            
        self.val_loss_min=early_stopping.val_loss_min           
        print('state_3')
        
        
class PytorchTrainer_GNN:
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 loss: torch.nn.modules.loss,
                 epochs=1,
                 device='cpu', 
                ):
        self.optimizer = optimizer
        self.loss = loss
        self.epochs = epochs
        self.loss_history = []
        self.val_loss_history=[]
        self.val_loss_min=np.Inf
        #self.device=device 
        if device=='cpu':                                                   
            self.device= torch.device('cpu')                                             
        else:                                                                   
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def train(self,
              model: torch.nn.Module,
              train_loader: DataLoader,):
        
#         if self.device=='cpu':                                                   
#             device='cpu'                                                        
#         else:                                                                   
#             device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        model.train()
        model.to(self.device)

        for data in train_loader:  # Iterate in batches over the training dataset.
            data.to(self.device)
            out = model(data.x.float(), data.edge_index, data.batch)  # Perform a single forward pass.
            loss = self.loss(out, data.y)  # Compute the loss.
            loss.backward()  # Derive gradients.
            self.optimizer.step()  # Update parameters based on gradients.
            self.optimizer.zero_grad()  # Clear gradients.
    
    def evaluate(self,
                 model,
                 data_loader: DataLoader):
        
#         if self.device=='cpu':                                                   
#             device='cpu'                                                        
#         else:                                                                   
#             device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model.eval()
        loss_total=0
        num=0
        for data in data_loader:
            data.to(self.device)
            out = model(data.x.float(), data.edge_index, data.batch)
            loss = self.loss(out,data.y).item()
            loss_total+=loss*len(data)
            num+=len(data)
            
        return loss_total/num   
        
    
    def fit(self,
            model: torch.nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader=None,
            test_loader: DataLoader=None,
            save_history=False,
            early_stop=False,
            patience=7,
            es_checkpoint='checkpoint.pt',
           ):
        if early_stop:
            early_stopping = EarlyStopping(patience=patience, verbose=True, path=es_checkpoint)
        
        for epoch in range(self.epochs):
            print("epoch: ", epoch)
            self.train(model,train_loader)
            
            train_loss=self.evaluate(model,train_loader)
            val_loss=self.evaluate(model,val_loader)
            if save_history:
                self.loss_history.append(train_loss)
                if val_loader:
                    self.val_loss_history.append(val_loss)
                        
            if early_stop:
                early_stopping(val_loss, model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
        
        if early_stop:
            # load the last checkpoint with the best model
            model.load_state_dict(torch.load(early_stopping.path)) 
        
        self.val_loss_min=early_stopping.val_loss_min
        

class PytorchTrainer_MLP_test:
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 loss: torch.nn.modules.loss,
                 epochs=1,
                 device='cpu',
                ):
        self.optimizer = optimizer
        self.loss = loss
        self.epochs = epochs
        self.loss_history = []
        self.val_loss_history=[]
        self.val_loss_min=np.Inf
        #self.device=device 
        if device=='cpu':                                                   
            self.device= torch.device('cpu')                                             
        else:                                                                   
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def train(self,
              model: torch.nn.Module,
              train_loader: DataLoader,):
        
#         if self.device=='cpu':                                                   
#             device='cpu'                                                        
#         else:                                                                   
#             device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        model.train()
        model.to(self.device)

        for data in train_loader:  # Iterate in batches over the training dataset.
            x,y = data
            x=x.to(self.device)
            y=y.to(self.device)
            out = model(x)  # Perform a single forward pass.
            loss = self.loss(out, y)  # Compute the loss.
            loss.backward()  # Derive gradients.
            self.optimizer.step()  # Update parameters based on gradients.
            self.optimizer.zero_grad()  # Clear gradients.
    
    def evaluate(self,
                 model,
                 data_loader: DataLoader,
                 epoch,
                 name):
        
        model.eval()
        loss_total=0
        num=0
        
        y_test=torch.Tensor([])
        y_pred=torch.Tensor([])
        y_test=y_test.to(self.device)
        y_pred=y_pred.to(self.device)
        
        for data in data_loader:
            x,y = data
            x=x.to(self.device)
            y=y.to(self.device)
            out = model(x)
            loss = self.loss(out,y).item()
            loss_total+=loss*len(data)
            num+=len(data)
            y_test=torch.cat((y_test,y))
            y_pred=torch.cat((y_pred,out))
            
        y_test=y_test.squeeze(1).cpu().detach().numpy()
        y_pred=y_pred.squeeze(1).cpu().detach().numpy()
        import matplotlib.pyplot as plt
        from scipy import stats
        
        fig, ax=plt.subplots()
        ax.scatter(y_test,y_pred)
        slope, intercept, r, p, stderr = stats.linregress(y_test, y_pred)
        ax.plot(y_test, intercept + slope * y_test, label='R=' + format(r,'.2f'), c='black' ) 
        mse=np.mean((y_pred-y_test)**2)
        mae=np.mean(abs(y_pred-y_test))
        ax.legend()
        ax.set_title('{}_{}_{}'.format(mse,mae,r))
        plt.savefig('{}_{}.png'.format(name,epoch))
        
        
        
        
        
        return loss_total/num   
        
    
    def fit(self,
            model: torch.nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader=None,
            test_loader: DataLoader=None,
            save_history=False,
            early_stop=False,
            patience=7,
            es_checkpoint='checkpoint.pt',
           ):
        if early_stop:
            early_stopping = EarlyStopping(patience=patience, verbose=True, path=es_checkpoint)
        
        for epoch in range(self.epochs):
            print("epoch: ", epoch)
            self.train(model,train_loader)
            
            train_loss=self.evaluate(model,train_loader,epoch,'train')
            val_loss=self.evaluate(model,val_loader,epoch, 'valid')
            
            #================================================
            #================================================
            from torch_geometric.data import Data
            from torch.utils.data import TensorDataset
            import matplotlib.pyplot as plt
            from scipy import stats
            path='./data/{}/Encoded_data/5fold/'.format('ube4b')
            fold=0
            X_test=np.load('{}X_train_fold{}.npy'.format(path,fold))
            y_test=np.load('{}y_train_fold{}.npy'.format(path,fold))
            X_test=torch.Tensor(X_test)  
            y_test=torch.Tensor(y_test).unsqueeze(1)  
            # create regular dataset
            test_dataset = TensorDataset(X_test,y_test)
            # data loader
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
            test_loader = train_loader
            
            fig, ax=plt.subplots(figsize=(8,10))      
            model.eval()
            y_pred=torch.Tensor([])
            y_pred=y_pred.to(torch.device('cuda'))
            y_test=torch.Tensor([])
            y_test=y_pred.to(torch.device('cuda'))
            for data in test_loader:
                x,y=data
                x=x.to(torch.device('cuda'))
                y=y.to(torch.device('cuda'))
                out=model(x).squeeze(1)
                y_pred=torch.cat((y_pred,out))
                y_test=torch.cat((y_pred,y))
            #y_pred_all=torch.cat((y_pred_all,y_pred), dim=1)
            y_pred=y_pred.cpu().detach().numpy()
            y_test=y_test.cpu().detach().numpy()
            #y_pred_all.append(y_pred)

            #y_pred_ensem=np.array(y_pred_all).mean(axis=0)
            #ax=fig.add_subplot(len(folders),1,idx)
            ax.scatter(y_test,y_pred, s=2)
            slope, intercept, r, p, stderr = stats.linregress(y_test, y_pred)
            ax.plot(y_test, intercept + slope * y_test, label='R=' + format(r,'.2f'), c='black' ) 
            mse=np.mean((y_test-y_pred)**2)
            mae=np.mean(abs(y_test-y_pred))
            ax.set_title('{}_{}'.format(mse,mae))
            plt.savefig('correlation_{}.png'.format(epoch))
            #=======================================================
            #=======================================================
            
            
            
            if save_history:
                self.loss_history.append(train_loss)
                if val_loader:
                    self.val_loss_history.append(val_loss)
                        
            if early_stop:
                early_stopping(val_loss, model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
            
            
            
        if early_stop:
            # load the last checkpoint with the best model
            model.load_state_dict(torch.load(early_stopping.path)) 
        
        self.val_loss_min=early_stopping.val_loss_min               