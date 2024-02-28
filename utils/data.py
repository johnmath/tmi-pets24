from torch.utils.data import Dataset
from torchvision.datasets import CIFAR100
import numpy as np
        
class CoarseCIFAR100(Dataset):
    """
    Splits CIFAR100 into its coarse labels
    """
    def __init__(self, train=True, transform=None, root="CIFAR100"):
        self._original_set = CIFAR100(root="CIFAR100", 
                                      download=True, 
                                      train=train, 
                                      transform=transform)
        
        self._coarse_labels = np.array([4,  1, 14,  8,  0,  6,  7,  7, 18,  3,  
                                        3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                                        6, 11,  5, 10,  7,  6, 13, 15,  3, 15,  
                                        0, 11,  1, 10, 12, 14, 16,  9, 11,  5, 
                                        5, 19,  8,  8, 15, 13, 14, 17, 18, 10, 
                                        16, 4, 17,  4,  2,  0, 17,  4, 18, 17, 
                                        10, 3,  2, 12, 12, 16, 12,  1,  9, 19,  
                                        2, 10,  0,  1, 16, 12,  9, 13, 15, 13, 
                                        16, 19,  2,  4,  6, 19,  5,  5,  8, 19, 
                                        18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
                
    def __len__(self):
        return len(self._original_set)
    
    def __getitem__(self, idx):
        return (self._original_set[idx][0], 
                self._coarse_labels[self._original_set[idx][1]])
    
        

