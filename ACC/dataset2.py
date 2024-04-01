import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
import random
random.seed(0)

'''
all_data = np.load("dataset/critical_data.npy", allow_pickle=True).item()
label = all_data['label']
max_len = 0
for i in range(507958):
    idx = 0
    for j in range(191):
        if np.abs(label[i,idx,0]) < 1e-3:
            label[i,idx:191,:] = label[i,idx+1:192,:]
        else:
            idx += 1
    if idx > max_len:
        max_len = idx
        print(max_len)
    if i % 1000 == 0:
        print(i)
all_data['label'] = label
np.save("dataset/critical_data.npy", all_data)
'''

class uncritical_data(Dataset):
    def __init__(self, range, transform = None, target_transform = None):
        self.transform = transform
        self.target_transform = target_transform
        self.all_data = np.load("/mnt/mnt1/linxuan/nnv/ACC/dataset/uncritical_data.npy", allow_pickle=True).item()
        self.data = torch.tensor(self.all_data['data'][range,...],dtype=torch.float32)
        self.label = torch.tensor(self.all_data['label'][range,...],dtype=torch.float32)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        
        data = self.data[index,...]
        label = self.label[index,...]
        
        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            label = self.target_transform(data)

        return data, label

def get_dataset(batch_size):
    order = np.array(range(1763451))
    random.shuffle(order)

    train_dataset = uncritical_data(order[:1400000])
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0)
     
    test_dataset = uncritical_data(order[1400000:1600000])
    test_loader = DataLoader(dataset=test_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=0)
    
    val_dataset = uncritical_data(order[1600000:])
    val_loader = DataLoader(dataset=val_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=0)
    
    return train_loader, test_loader, val_loader
