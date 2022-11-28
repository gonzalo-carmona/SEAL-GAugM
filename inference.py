import numpy as np
import pandas as pd
import torch
import torch_geometric
import os
import scipy.sparse as ssp
from models import *
from seal_link_pred import SEALDataset, SEALDynamicDataset
from utils import *
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Dataset, InMemoryDataset, DataLoader
from tqdm import tqdm
import multiprocessing as mp
import time
import itertools as it
import pickle
from custom_datasets import *
import argparse
    
def findsubsets(s, n):
    return list(it.combinations(s, n))

class CustomDataset(InMemoryDataset):
    def __init__(self, root, data, num_hops, use_coalesce=False,
        node_label='drnl', ratio_per_hop=1.0, max_nodes_per_hop=None,
        directed=False):
        self.data = data
        self.num_hops = num_hops
        self.use_coalesce = use_coalesce
        self.node_label = node_label
        self.ratio_per_hop = ratio_per_hop
        self.max_nodes_per_hop = max_nodes_per_hop
        self.directed = directed
        super(CustomDataset, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        if self.percent == 100:
            name = 'SEAL_{}_data'.format(self.split)
        else:
            name = 'SEAL_{}_data_{}'.format(self.split, self.percent)
        name += '.pt'
        return [name]

    def process(self):
        nodes = set(range(self.data.num_nodes))
        all_edge = torch.Tensor(np.array(findsubsets(nodes, 2))).t().type(torch.int64)
        
        if self.use_coalesce:  # compress mutli-edge into edge with weight
            self.data.edge_index, self.data.edge_weight = coalesce(
                self.data.edge_index, self.data.edge_weight,
                self.data.num_nodes, self.data.num_nodes)
        if 'edge_weight' in self.data:
            edge_weight = self.data.edge_weight.view(-1)
        else:
            edge_weight = torch.ones(self.data.edge_index.size(1), dtype=int)
        A = ssp.csr_matrix(
            (edge_weight, (self.data.edge_index[0], self.data.edge_index[1])),
            shape=(self.data.num_nodes, self.data.num_nodes)
        )

        if self.directed:
            A_csc = A.tocsc()
        else:
            A_csc = None

        subgraphs = extract_enclosing_subgraphs(
            all_edge, A, self.data.x, 1, self.num_hops, self.node_label,
            self.ratio_per_hop, self.max_nodes_per_hop, self.directed, A_csc
        )

        torch.save(self.collate(subgraphs), self.processed_paths[0])
        del subgraphs

class SEALDynamicCustomDataset(Dataset):
    def __init__(self, root, data, num_hops,
                 use_coalesce=False, node_label='drnl', ratio_per_hop=1.0,
                 max_nodes_per_hop=None, directed=False, **kwargs):
        self.data = data
        self.num_hops = num_hops
        self.use_coalesce = use_coalesce
        self.node_label = node_label
        self.ratio_per_hop = ratio_per_hop
        self.max_nodes_per_hop = max_nodes_per_hop
        self.directed = directed
        super(SEALDynamicCustomDataset, self).__init__(root)

        nodes = set(range(self.data.num_nodes))
        all_edge = torch.Tensor(np.array(findsubsets(nodes, 2))).t().type(torch.int64)
        
        self.links = all_edge.t().tolist()
        self.labels = [1] * all_edge.size(1)

        if self.use_coalesce:  # compress mutli-edge into edge with weight
            self.data.edge_index, self.data.edge_weight = coalesce(
                self.data.edge_index, self.data.edge_weight,
                self.data.num_nodes, self.data.num_nodes)

        if 'edge_weight' in self.data:
            edge_weight = self.data.edge_weight.view(-1)
        else:
            edge_weight = torch.ones(self.data.edge_index.size(1), dtype=int)
        self.A = ssp.csr_matrix(
            (edge_weight, (self.data.edge_index[0], self.data.edge_index[1])),
            shape=(self.data.num_nodes, self.data.num_nodes)
        )
        if self.directed:
            self.A_csc = self.A.tocsc()
        else:
            self.A_csc = None

    def __len__(self):
        return len(self.links)

    def len(self):
        return self.__len__()

    def get(self, idx):
        src, dst = self.links[idx]
        y = self.labels[idx]
        tmp = k_hop_subgraph(src, dst, self.num_hops, self.A, self.ratio_per_hop,
                             self.max_nodes_per_hop, node_features=self.data.x,
                             y=y, directed=self.directed, A_csc=self.A_csc)
        data = construct_pyg_graph(*tmp, self.node_label)

        return data

if __name__=='__main__':
    
    st = time.time()
    parser = argparse.ArgumentParser(description='Prediccion de aristas')
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--dynamic_inference', type=str, default='y')
    parser.add_argument('--dynamic_train', action='store_true')
    parser.add_argument('--num_hops', type=int, default='2')
    parser.add_argument('--train_percent', type=float, default='100')
    parser.add_argument('--best_run', type=int, default='25')
    parser.add_argument('--k', type=int, default='30')
    parser.add_argument('--batch_size', type=int, default='32')
    parser.add_argument('--save_appendix', type=str, default='defaultparams')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    argdataset = args.dataset
    dynamic = args.dynamic_inference
    best_run = args.best_run
    
    if dynamic not in ['y', 'n']:
        raise Exception("""Argumento dynamic no válido. Escribir 'y' o 'n'.""")
    
    if best_run not in range(51)[1:]:
        raise Exception("""Argumento best_run no válido. El modelo se entrena durante 50 épocas,
        por lo que el que el argumento debe ser un número entero entre 1 y 50, ambos inclusive.
        A la hora de entrenarlo con seal_link_pred.py, el script informa de que run ha sido la mejor
        en el conjunto de validación, mediante la variable "Highest Eval point". Es recomendable escoger esa run.""")
    
    if dynamic=='y':
        dataset_class = 'SEALDynamicDataset'
    else:
        dataset_class = 'SEALDataset'
    
    datasetlist = ['Cora', 'CiteSeer', 'BlogCatalog', 'AirUSA', 'Flickr', 'PPI', 'KarateClub']
    
    if argdataset not in datasetlist:
        raise Exception("""Se introdujo un dataset no válido. Introducir un dataset
        de los de la siguiente lista: {}""".format(datasetlist))
    
    if argdataset=='Cora' or argdataset=='CiteSeer':
        path = os.path.join('dataset', argdataset)
        dataset = Planetoid(path, argdataset)
    else:
        path = os.path.join('dataset', argdataset)
        dataset = eval(argdataset)(path, argdataset)
    
    appendix = args.save_appendix
    try:
        model_statedict = torch.load("results\\{}{}\\run1_model_checkpoint{}.pth".format(argdataset, appendix, best_run))
    except:
        raise Exception("""El modelo SEAL no ha sido entrenado todavía para este dataset.
        Es necesario ejecutar seal_link_pred con el dataset correspondiente (Cora, CiteSeer, PPI, AirUSA, Flickr, BlogCatalog), y los siguientes argumentos:
        --num_hops 2, --use_feature, --save_appendix defaultparams, --train_node_embedding. Añadir --num_workers, y el número de cores de la CPU.""")
    split_edge = do_edge_split(dataset, False)
    data = dataset[0]
    data.edge_index = split_edge['train']['edge'].t()

    num_hops = args.num_hops
    train_percent = args.train_percent
    use_coalesce = False
    node_label = 'drnl'
    ratio_per_hop = 1.0
    max_nodes_per_hop = None
    directed = False

    train_dataset = eval(dataset_class)(
            path,
            data,
            split_edge,
            num_hops=num_hops,
            percent=train_percent,
            split='train',
            use_coalesce=use_coalesce,
            node_label=node_label,
            ratio_per_hop=ratio_per_hop,
            max_nodes_per_hop=max_nodes_per_hop,
            directed=directed,
        )
    
    hidden_channels = 32
    num_layers = 3
    sortpool_k = args.k
    dynamic_train = args.dynamic_train
    use_feature = True
    max_z = 1000
    emb = torch.nn.Embedding(data.num_nodes, hidden_channels).to(device)
    
    model = DGCNN(hidden_channels, num_layers, max_z, sortpool_k,
                            train_dataset, dynamic_train, use_feature=use_feature,
                            node_embedding=emb).to(device)
    model.load_state_dict(model_statedict)
    
    print("Preparando argumentos")
    data = dataset[0]
    
    inference_dataset = SEALDynamicCustomDataset(
        path,
        data,
        num_hops=num_hops,
        use_coalesce=use_coalesce,
        node_label=node_label,
        ratio_per_hop=ratio_per_hop,
        max_nodes_per_hop = max_nodes_per_hop,
        directed=directed
    )
    
    print("Comenzando cálculo de probabilidades")
    
    batch_size=args.batch_size
    inference_loader = DataLoader(inference_dataset, batch_size=batch_size, num_workers = mp.cpu_count()-2)
    
    prob_df = pd.DataFrame(columns=list(range(data.num_nodes)), index = range(data.num_nodes))
    s = torch.nn.Sigmoid()
    
    for data in tqdm(inference_loader, ncols=70):
        x = data.x
        edgebatch = data.edge_index
        edge_weight = None
        node_id = data.node_id
        logits = model(data.z, edgebatch, data.batch, x, edge_weight, node_id).view(-1).cpu()
        probas = s(logits)
        labels = list(data.z)
        positions = [index for index, value in enumerate(labels) if value==1]
        for i in range(len(probas)):
           src = node_id[positions[0]].item()
           dst = node_id[positions[1]].item()
           positions = positions[2:]
           prob_df.loc[src, dst] = probas[i].item()
           prob_df.loc[dst, src] = probas[i].item()

    print("completando la diagonal del dataframe")

    prob_df.fillna(0)
    
    print("Guardando resultados")
    
    values = prob_df.values
    pickle.dump(values, open('data/edge_probabilities/{}_graph_2_logits.pkl'.format(argdataset), 'wb'))
    en=time.time()
    secs = en-st
    mins = secs//60
    secs = secs%60
    hours = mins//60
    mins = mins%60
    print("Tarea finalizada. Tiempo transcurrido:{} h, {} min, {} secs".format(hours, mins, secs))
