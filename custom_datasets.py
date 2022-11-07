import torch
from torch_geometric.data import Data, InMemoryDataset
import scipy.sparse as scsp
import numpy as np

class Flickr(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.name = name

    @property
    def raw_file_names(self):
        return ['Flickr_adj.pkl', 'Flickr_features.pkl', 'Flickr_labels.pkl']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        pass

    def process(self):
        adj = scsp.find(np.load('dataset/Flickr/raw/Flickr_adj.pkl', allow_pickle=True))
        edge_index = torch.Tensor(np.array([adj[0], adj[1]])).type(torch.int64)
        feats = np.load('dataset/Flickr/raw/Flickr_features.pkl', allow_pickle=True).todense()
        x = torch.Tensor(feats).type(torch.int64)
        labels = np.load('dataset/Flickr/raw/Flickr_labels.pkl', allow_pickle=True)
        y = torch.Tensor(labels).type(torch.int64)
        data_list = [Data(x=x, edge_index=edge_index, y=y)]
        
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
    
    def __repr__(self) -> str:
        return f'{self.name}()'
        
class BlogCatalog(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.name = name
        
    @property
    def raw_file_names(self):
        return ['BlogCatalog_adj.pkl', 'BlogCatalog_features.pkl', 'BlogCatalog_labels.pkl']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        pass

    def process(self):
        adj = scsp.find(np.load('dataset/BlogCatalog/raw/BlogCatalog_adj.pkl', allow_pickle=True))
        edge_index = torch.Tensor(np.array([adj[0], adj[1]])).type(torch.int64)
        feats = np.load('dataset/BlogCatalog/raw/BlogCatalog_features.pkl', allow_pickle=True).todense()
        x = torch.Tensor(feats).type(torch.int64)
        labels = np.load('dataset/BlogCatalog/raw/BlogCatalog_labels.pkl', allow_pickle=True)
        y = torch.Tensor(labels).type(torch.int64)
        data_list = [Data(x=x, edge_index=edge_index, y=y)]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
            
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
    
    def __repr__(self) -> str:
        return f'{self.name}()'
        
class PPI(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.name = name

    @property
    def raw_file_names(self):
        return ['PPI_adj.pkl', 'PPI_features.pkl', 'PPI_labels.pkl']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        pass

    def process(self):
        adj = scsp.find(np.load('dataset/PPI/raw/PPI_adj.pkl', allow_pickle=True))
        edge_index = torch.Tensor(np.array([adj[0], adj[1]])).type(torch.int64)
        feats = np.load('dataset/PPI/raw/PPI_features.pkl', allow_pickle=True)
        x = torch.Tensor(feats).type(torch.int64)
        labels = np.load('dataset/PPI/raw/PPI_labels.pkl', allow_pickle=True)
        y = torch.Tensor(labels).type(torch.int64)
        data_list = [Data(x=x, edge_index=edge_index, y=y)]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
    
    def __repr__(self) -> str:
        return f'{self.name}()'

class AirUSA(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.name = name

    @property
    def raw_file_names(self):
        return ['AirUSA_adj.pkl', 'AirUSA_features.pkl', 'AirUSA_labels.pkl']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        pass

    def process(self):
        adj = scsp.find(np.load('dataset/AirUSA/raw/AirUSA_adj.pkl', allow_pickle=True))
        edge_index = torch.Tensor(np.array([adj[0], adj[1]])).type(torch.int64)
        feats = np.load('dataset/AirUSA/raw/AirUSA_features.pkl', allow_pickle=True)
        x = torch.Tensor(feats).type(torch.int64)
        labels = np.load('dataset/AirUSA/raw/AirUSA_labels.pkl', allow_pickle=True)
        y = torch.Tensor(labels).type(torch.int64)
        data_list = [Data(x=x, edge_index=edge_index, y=y)]
        
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
    
    def __repr__(self) -> str:
        return f'{self.name}()'