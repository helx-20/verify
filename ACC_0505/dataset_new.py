import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
import random
import json
import sys
random.seed(0)

'''
data_path = "/mnt/mnt1/linxuan/nnv/ACC/new_train_data_critical_json"
max_nVar = 0
max_nC = 0
for i in range(255717):
    with open(os.path.join(data_path,'Uin_{}.json'.format(i+1)), 'r') as file:
        data = json.load(file)
        nVar = len(data['Uin_V'][0]) - 1
        nC = len(data['Uin_C'])
        print(nVar,nC)
        if nVar > max_nVar:
            max_nVar = nVar
            print(max_nVar,max_nC)
        if nC > max_nC:
            max_nC = nC
            print(max_nVar,max_nC)
'''

class critical_data(Dataset):
    def __init__(self, idx_range, append = False):
        self.data_path = "/mnt/mnt1/linxuan/nnv/ACC/new_train_data2_critical_json"
        self.append_data_path = '/mnt/mnt1/linxuan/nnv/ACC/new_train_data2_append_critical_json'
        self.all_data_list = idx_range
        self.data_V_dict = []
        self.data_C_dict = []
        self.label_dict = []
        for index in range(len(idx_range)):
            idx = self.all_data_list[index] + 1
            with open(os.path.join(self.data_path,'Uin_{}.json'.format(idx)), 'r') as file:
                data = json.load(file)
                data_V = torch.tensor(data['Uin_V'],dtype=torch.float32)
                row = torch.tensor(data['Uin_C'][:][0]) - 1
                col = torch.tensor(data['Uin_C'][:][1]) - 1
                C_value = torch.tensor(data['Uin_C'][:][2],dtype=torch.float32)
                data_C_d = torch.zeros(323,164,dtype=torch.float32)
                data_C_d[row,col] = C_value
                data_C_d[:,163] = torch.tensor(data['Uin_d'],dtype=torch.float32)
            with open(os.path.join(self.data_path,'Rc_convex_{}.json'.format(idx)), 'r') as file:
                label = torch.tensor(json.load(file),dtype=torch.float32).view(-1)
            self.data_V_dict.append(data_V.transpose(0,1))
            self.data_C_dict.append(data_C_d.transpose(0,1))
            self.label_dict.append(label)
        if append:
            for index in range(int((len(os.listdir(self.append_data_path)))/3)):
                idx = index + 1
                with open(os.path.join(self.append_data_path,'Uin_{}.json'.format(idx)), 'r') as file:
                    data = json.load(file)
                    data_V = torch.tensor(data['Uin_V'],dtype=torch.float32)
                    row = torch.tensor(data['Uin_C'][:][0]) - 1
                    col = torch.tensor(data['Uin_C'][:][1]) - 1
                    C_value = torch.tensor(data['Uin_C'][:][2],dtype=torch.float32)
                    data_C_d = torch.zeros(323,164,dtype=torch.float32)
                    data_C_d[row,col] = C_value
                    data_C_d[:,163] = torch.tensor(data['Uin_d'],dtype=torch.float32)
                with open(os.path.join(self.append_data_path,'Rc_convex_{}.json'.format(idx)), 'r') as file:
                    label = torch.tensor(json.load(file),dtype=torch.float32).view(-1)
                self.data_V_dict.append(data_V.transpose(0,1))
                self.data_C_dict.append(data_C_d.transpose(0,1))
                self.label_dict.append(label)
        self.data_V_dict = torch.stack(self.data_V_dict,dim=0)
        self.data_C_dict = torch.stack(self.data_C_dict,dim=0)
        self.label_dict = torch.stack(self.label_dict,dim=0)

    def __len__(self):
        return self.label_dict.shape[0]

    def __getitem__(self, index):
        data_V = self.data_V_dict[index,...]
        data_C = self.data_C_dict[index,...]
        label = self.label_dict[index,...]
        
        return data_V, data_C, label

def get_dataset(batch_size):
    order = np.array(range(255717))
    random.shuffle(order)

    train_dataset = critical_data(order[:200000], append=True)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0)
     
    test_dataset = critical_data(order[200000:225000])
    test_loader = DataLoader(dataset=test_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=0)
    
    val_dataset = critical_data(order[225000:])
    val_loader = DataLoader(dataset=val_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=0)
    
    return train_loader, test_loader, val_loader

'''
data_C_dict = []
train, _, _2 = get_dataset(256)
#for batch_id, (data_V, data_C_d, target) in enumerate(train): 
#    data_C_dict.append(data_C_d)
    #torch.save(data_C_d,'test.pt')
'''
