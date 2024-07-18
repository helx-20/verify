import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
import random
import json
import sys
random.seed(0)

class critical_data(Dataset):
    def __init__(self, idx_range, transform = None, target_transform = None, append = False):
        self.transform = transform
        self.target_transform = target_transform
        self.data_path = "/mnt/mnt1/linxuan/nnv/ACC/new_train_data3_critical_json"
        self.append_data_path = '/mnt/mnt1/linxuan/nnv/ACC/new_train_data3_append_critical_json'
        self.all_data_list = idx_range
        if append:
            self.all_data_list = np.array(self.all_data_list.tolist() + [i for i in range(255717*2)])
            
    def __len__(self):
        return len(self.all_data_list)

    def __getitem__(self, index):
        if index < 593528:
            idx = self.all_data_list[index] + 1
            with open(os.path.join(self.data_path,'Uin_{}.json'.format(idx)), 'r') as file:
                data = json.load(file)
                data_V = torch.tensor(data['Uin_V'],dtype=torch.float32)
                data_C = torch.tensor(data['Uin_C'])
                row = torch.tensor(data_C[:,0]) - 1
                col = torch.tensor(data_C[:,1]) - 1
                C_value = torch.tensor(data_C[:,2],dtype=torch.float32)
                data_C_d = torch.zeros(323,164,dtype=torch.float32)
                data_C_d[row,col] = C_value
                data_C_d[:,163] = torch.tensor(data['Uin_d'],dtype=torch.float32)
            with open(os.path.join(self.data_path,'Rc_convex_{}.json'.format(idx)), 'r') as file:
                label = torch.tensor(json.load(file),dtype=torch.float32).view(-1)
        else:
            idx = self.all_data_list[index] + 1
            with open(os.path.join(self.append_data_path,'Uin_{}.json'.format(idx)), 'r') as file:
                data = json.load(file)
                data_V = torch.tensor(data['Uin_V'],dtype=torch.float32)
                data_C = torch.tensor(data['Uin_C'])
                row = torch.tensor(data_C[:,0]) - 1
                col = torch.tensor(data_C[:,1]) - 1
                C_value = torch.tensor(data_C[:,2],dtype=torch.float32)
                data_C_d = torch.zeros(323,164,dtype=torch.float32)
                data_C_d[row,col] = C_value
                data_C_d[:,163] = torch.tensor(data['Uin_d'],dtype=torch.float32)
            with open(os.path.join(self.append_data_path,'Rc_convex_{}.json'.format(idx)), 'r') as file:
                label = torch.tensor(json.load(file),dtype=torch.float32).view(-1)

        return data_V.transpose(0,1), data_C_d.transpose(0,1), label

def get_dataset(batch_size):
    order = np.array(range(593528))
    random.shuffle(order)

    train_dataset = critical_data(order[:500000])
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0)
     
    test_dataset = critical_data(order[500000:545000])
    test_loader = DataLoader(dataset=test_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=0)
    
    val_dataset = critical_data(order[545000:])
    val_loader = DataLoader(dataset=val_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=0)
    
    return train_loader, test_loader, val_loader


#train, _, _2 = get_dataset(1024)
#for batch_id, (data_V, data_C_d, target) in enumerate(train): 
#    pass
    #if data_V.shape[1] != 164 or data_C_d.shape[2] != 323 or target.shape[1] != 2:
    #    print(data_V.shape,data_C_d.shape,target.shape)

