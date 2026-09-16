import os
import torch
from torch.utils.data import Dataset
import medmnist
from medmnist import INFO
from ml.datasets.preprocessing import get_preprocessing_transform

class MedMNISTDataset(Dataset):
    def __init__(self, split='train', data_flag='pneumoniamnist', dataset_root='./data', transform=None):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            data_flag (str): MedMNIST dataset flag, e.g., 'pneumoniamnist'.
            dataset_root (str): Root directory where datasets are stored.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.split = split
        self.data_flag = data_flag
        self.dataset_root = dataset_root
        
        info = INFO[data_flag]
        DataClass = getattr(medmnist, info['python_class'])
        
        # Download will only happen if it doesn't exist in root
        self.dataset = DataClass(split=self.split, download=True, root=self.dataset_root)
        
        if transform:
            self.transform = transform
        else:
            self.transform = get_preprocessing_transform(train=(split == 'train'))
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img, label = self.dataset[idx]
        
        if self.transform:
            img = self.transform(img)
            
        # label is returned as a 1D numpy array e.g., [1]. Convert to float32 tensor
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        return img, label_tensor
