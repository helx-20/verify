import os
import numpy as np
import json
import torch
path = '/mnt/mnt1/linxuan/nnv/ACC/new_train_data3_uncritical_json'
name_list = os.listdir(path)
exist_list = np.zeros((int(len(name_list)/3)))
del_list = []
for i in range(int(len(name_list)/3)):
    idx = i + 1
    with open(os.path.join(path,'Uin_{}.json'.format(idx)), 'r') as file:
        data = json.load(file)
    with open(os.path.join(path,'Rc_convex_{}.json'.format(idx)), 'r') as file:
        label = torch.tensor(json.load(file),dtype=torch.float32).view(-1)
    if torch.tensor(data['Uin_d'],dtype=torch.float32).shape[0] != 323 or torch.tensor(data['Uin_V'],dtype=torch.float32).shape[1] != 164 or label.shape[0] != 2:
        print(torch.tensor(data['Uin_d'],dtype=torch.float32).shape[0],torch.tensor(data['Uin_V'],dtype=torch.float32).shape[1],label.shape[0])
        del_list.append(idx)
        os.remove(os.path.join(path,'Rc_{}.json'.format(idx)))
        os.remove(os.path.join(path,'Rc_convex_{}.json'.format(idx)))
        os.remove(os.path.join(path,'Uin_{}.json'.format(idx)))