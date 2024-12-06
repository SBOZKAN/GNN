import numpy as np
import pandas as pd
from torch.utils.data import random_split
import torch
import matplotlib.pyplot as plt

def split_data(dataset, test_size=0.25, random_state=None):
    # split data into train and test/val
    random_state = 2022 if random_state is None else random_state
    length=len(dataset)
    test_num=int(length*test_size)
    train_dataset, test_dataset=random_split(dataset, lengths=[length-test_num,test_num], 
                                             generator=torch.Generator().manual_seed(random_state))
    return train_dataset,test_dataset



def read_pdb(pdb_file):
    f = open(pdb_file,'r')   #read pdb file
    pdb = f.readlines()
    out=[]
    Res_ID={}     # a list which match the Res number with the new ID
    coor={}
    wt={}
    new_id=0
    for line in pdb:
        if line[:4] == "ATOM":
            atom = line[0:6]
            atomserialnumber = int(line[6:11])
            atomname = line[12:16]
            atomname=atomname.strip()
            alternatelocationindicator = line[16:17]
            residuename = line[17:20]
            chainidentifier = line[21:22]
            chainidentifier = chainidentifier.strip()
            resnumber = line[22:26]
            resnumber=int(resnumber.strip())
            codeforinsertionofresidues = line[26:27]
            orthogonalcoordinatesforx = float(line[30:38])
            orthogonalcoordinatesfory = float(line[38:46])
            orthogonalcoordinatesforz = float(line[46:54])
            occupancy = float(line[54:60])
            bfactor = float(line[60:66])
            elementsymbol = line[76:78]
            chargeontheatom = line[78:80]

            #pctdci=dfi_wt.query("ResI == %d"%resnumber)['pctdfi'].values[0]
            #bfactor=pctdci
            
            if atomname =='CA':
                Res_ID[resnumber]=new_id
                coor[new_id]=np.array([orthogonalcoordinatesforx,orthogonalcoordinatesfory,orthogonalcoordinatesforz])
                wt[resnumber]=residuename
                new_id += 1
#             oneliner = "%-6s%5d %4s%1s%3s %1s%4s%1s   %8.3f%8.3f%8.3f%6.2f%6.2f          %2s%2s\n" % \
#                 (atom,
#                 atomserialnumber,
#                 atomname,
#                 alternatelocationindicator,
#                 residuename,
#                 chainidentifier,
#                 int(resnumber),
#                 codeforinsertionofresidues,
#                 orthogonalcoordinatesforx,
#                 orthogonalcoordinatesfory,
#                 orthogonalcoordinatesforz,
#                 occupancy,
#                 bfactor,
#                 elementsymbol,
#                 chargeontheatom)
#             out.append(oneliner)
    return Res_ID, coor, wt


def create_graph_pdb(coor, 
                     cutoff, 
                     Res_ID, 
                     common_neighbour=None, 
                     save_path=None):
    id_list = list(coor.keys())
    source, target, dist = [], [], []
    for i in range(len(id_list)):
        for j in range(len(id_list)):
            source.append(i)
            target.append(j)
            dist.append(np.sqrt(np.mean((coor[i]-coor[j])**2)))
            
    graph = pd.DataFrame({'source':source, 'target': target, 'couple': dist})
    # define common neighbour                                                   
    if common_neighbour:                                                        
        #common_neighbour=[5,6,7]                                               
        min_c=min(graph.couple)                                                 
        for i in Res_ID.values():                                               
            for j in common_neighbour:                                          
                idx=graph.loc[(graph['source']==i)&(graph['target']==j)]['couple'].index
                graph.loc[idx,'couple']=min_c                                   
    ############## 
    graph=graph.loc[(graph['couple']<=cutoff) & (graph['source']!=graph['target'])]
    graph['couple']=5/graph['couple']
    
    # save if save_path specified
    if save_path:
        adj=[[0]*len(Res_ID) for i in range(len(Res_ID))]
        total=0
        for index, row in graph.iterrows():
            i=int(row['source'])
            j=int(row['target'])
            c=row['couple']
            adj[i][j]=1
            total+=1

        for i in range(len(adj)):
            adj[i][i]=1

        import seaborn as sns
        from matplotlib.colors import LinearSegmentedColormap
        sns.set()
        f, ax = plt.subplots(figsize=(10, 8))

        my_colormap = LinearSegmentedColormap.from_list("", ["white","red",])
        ax = sns.heatmap(adj, cmap=my_colormap,vmin=0, vmax=1)
        #y_label=ax.get_yticklabels()
        #ax.set_xticklabels(np.arange(1,len(adj)+1,2))
        #ax.set_yticklabels(np.arange(1,len(adj)+1,2))
        ax.set_title('num_edges: {}'.format(total))
        plt.show()
        plt.savefig('{}graph_map.png'.format(save_path))
    
    return graph


def create_graph_DCI(path, 
                     name, 
                     cutoff, 
                     Res_ID, 
                     q='pctfdfi', 
                     inverse=True, 
                     common_neighbour=None, 
                     save_path=None):
    
    source, target, dci = [], [], []
    for i in Res_ID:
        data=pd.read_csv(path+'/DCI/{}/{}-dfianalysis.csv'.format(i,name))[q].values
        for j in range(len(data)):
            source.append(Res_ID[i])
            target.append(j)
            dci.append(data[j])       
    graph = pd.DataFrame({'source':source, 'target': target, 'couple': dci})
    
    
    if inverse:
        graph = pd.DataFrame({'source':target, 'target': source, 'couple': dci})

    # define common neighbour                                                   
    if common_neighbour:                                                        
        #common_neighbour=[5,6,7]                                               
        max_c=max(graph.couple)                                                 
        for i in Res_ID.values():                                               
            for j in common_neighbour:                                          
                idx=graph.loc[(graph['source']==i)&(graph['target']==j)]['couple'].index
                graph.loc[idx,'couple']=max_c                                   
    ############## 

    graph=graph.loc[(graph['couple']>=cutoff) & (graph['source']!=graph['target'])]
    
    # save if save_path specified
    if save_path:
        adj=[[0]*len(Res_ID) for i in range(len(Res_ID))]
        total=0
        for index, row in graph.iterrows():
            i=int(row['source'])
            j=int(row['target'])
            c=row['couple']
            adj[i][j]=1
            total+=1

        for i in range(len(adj)):
            adj[i][i]=1

        import seaborn as sns
        from matplotlib.colors import LinearSegmentedColormap
        sns.set()
        f, ax = plt.subplots(figsize=(10, 8))

        my_colormap = LinearSegmentedColormap.from_list("", ["white","red",])
        ax = sns.heatmap(adj, cmap=my_colormap,vmin=0, vmax=1)
        #y_label=ax.get_yticklabels()
        #ax.set_xticklabels(np.arange(1,len(adj)+1,2))
        #ax.set_yticklabels(np.arange(1,len(adj)+1,2))
        ax.set_title('num_edges: {}'.format(total))
        plt.show()
        plt.savefig('{}graph_map.png'.format(save_path))
        
    return graph

def create_graph_DCIasym(path, 
                     name, 
                     Res_ID, 
                     cutoff, 
                     c_type='abs',
                     q='pctfdfi',
                     common_neighbour=None,
                     save_path=None,
                     high_couple=None,
                    ):

    source, target, dci = [], [], []
    for i in Res_ID:
        data=pd.read_csv(path+'/DCI/{}/{}-dfianalysis.csv'.format(i,name))[q].values
        for j in range(len(data)):
            source.append(Res_ID[i])
            target.append(j)
            dci.append(data[j]) 

    df1=pd.DataFrame({'source': source, 'target':target, 'dci':dci})
    df2=pd.DataFrame({'target': source, 'source':target, 'dci2':dci})
    df=df1.merge(df2, on=['source','target'])
    df['dci_asym']=df.dci-df.dci2
    
    # filter based on dci
    if high_couple:
        if c_type=='abs':
            df=df.loc[(df.dci>high_couple) | (df.dci2>high_couple),:]
        elif c_type=='pos':
            df=df.loc[(df.dci>high_couple),:]
        else:
            df=df.loc[(df.dci2>high_couple),:]
    df.drop(columns=['dci','dci2'],inplace=True)
    
    # filter based on dci asymmetry
    if c_type=='abs':
        graph=df.loc[abs(df.dci_asym)>cutoff]
    elif c_type=='pos':
        graph=df.loc[df.dci_asym>cutoff]
    else:
        graph=df.loc[df.dci_asym<-cutoff]
    
    # define common neighbour                                                   
    if common_neighbour:                                                        
        #common_neighbour=[5,6,7]                                               
        max_c=max(graph.couple)                                                 
        for i in Res_ID.values():                                               
            for j in common_neighbour:                                          
                idx=graph.loc[(graph['source']==i)&(graph['target']==j)]['couple'].index
                graph.loc[idx,'couple']=max_c                                   
    ############## 
    
    # save if save_path specified
    if save_path:
        adj=[[0]*len(Res_ID) for i in range(len(Res_ID))]
        total=0
        for index, row in graph.iterrows():
            i=int(row['source'])
            j=int(row['target'])
            c=row['dci_asym']
            adj[i][j]=1
            total+=1

        for i in range(len(adj)):
            adj[i][i]=1

        import seaborn as sns
        from matplotlib.colors import LinearSegmentedColormap
        sns.set()
        f, ax = plt.subplots(figsize=(10, 8))

        my_colormap = LinearSegmentedColormap.from_list("", ["white","red",])
        ax = sns.heatmap(adj, cmap=my_colormap,vmin=0, vmax=1)
        #y_label=ax.get_yticklabels()
        #ax.set_xticklabels(np.arange(1,len(adj)+1,2))
        #ax.set_yticklabels(np.arange(1,len(adj)+1,2))
        ax.set_title('num_edges: {}'.format(total))
        plt.show()
        plt.savefig('{}graph_map.png'.format(save_path))
        
    return graph
