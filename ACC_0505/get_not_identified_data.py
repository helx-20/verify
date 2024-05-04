import json

import numpy as np
import math
import time
import os,sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import time
from dataset import get_dataset
from model.critical_net import Critical_Net

device = torch.device("cuda:1")

print('model established')
model = Critical_Net(input_dim_V=5, input_dim_C=323, output_dim=2, m_tokens_in=164)
model.load_state_dict(torch.load("saved_model/critical_net2_state_dict.pth"))
model = model.to(device)
print(model)

def a_in_b(a, b):
    in_area_size = 0
    out_area_size = 0
    for i in range(int(len(a)/2)):
        a_start = a[2*i]
        a_end = a[2*i+1]
        in_area_size_tmp = in_area_size
        for j in range(int(len(b)/2)):
            b_start = b[2*j]
            b_end = b[2*j+1]
            if (b_start < a_start and b_end > a_start and b_start < a_end and b_end > a_end):
                in_area_size += a_end - a_start
                break
            elif b_start < a_start and b_end > a_start:
                in_area_size += b_end - a_start
            elif b_start < a_end and b_end > a_end:
                in_area_size += a_end - b_start
            else:
                in_area_size += b_end - b_start
        out_area_size += (a_end - a_start) - (in_area_size - in_area_size_tmp)

    return in_area_size, out_area_size

def rearrange_data(data):
    current_idx = 1
    rearranged_data = data
    while 1:
        if current_idx + 1 >= len(rearranged_data):
            break
        else:
            if rearranged_data[current_idx] >= rearranged_data[current_idx+2]:
                rearranged_data.pop(current_idx+1)
                rearranged_data.pop(current_idx+1)
            elif rearranged_data[current_idx] >= rearranged_data[current_idx+1]:
                rearranged_data[current_idx] = rearranged_data[current_idx+2]
                rearranged_data.pop(current_idx+1)
                rearranged_data.pop(current_idx+1)
            else:
                current_idx += 2

    return rearranged_data

def transfer_range(data): # data = 48 * 2
    data = data.view(-1,2)
    transfered_data = []
    for i in range(data.shape[0]):
        if data[i,1] == 0:
            break
        data_start = (data[i,0] - data[i,1]).item()
        data_end = (data[i,0] + data[i,1]).item()
        if len(transfered_data) == 0:
            transfered_data.append(data_start)
            transfered_data.append(data_end)
        else:
            inserted = False
            for j in range(int(len(transfered_data)/2)):
                transfered_data_start = transfered_data[2*j]
                transfered_data_end = transfered_data[2*j+1]
                if data_start <= transfered_data_start:
                    transfered_data.insert(2*j,data_start)
                    transfered_data.insert(2*j+1,data_end)
                    inserted = True
                    break
            if not inserted:
                transfered_data.append(data_start)
                transfered_data.append(data_end)
        transfered_data = rearrange_data(transfered_data)

    return transfered_data

def judge_in_area(output, target):
    identified = 0 # target in output
    not_identified = 0 # target not in output
    wrongly_identified = 0 # output not in target
    for i in range(output.shape[0]):
        output_tmp = transfer_range(output[i,...])
        target_tmp = transfer_range(target[i,...])
        identified_tmp, not_identified_tmp = a_in_b(target_tmp, output_tmp)
        _, wrongly_identified_tmp = a_in_b(output_tmp, target_tmp)
        identified += identified_tmp
        not_identified += not_identified_tmp
        wrongly_identified += wrongly_identified_tmp

    return identified, not_identified, wrongly_identified

def test(device, test_loader, criterion, epoch):
    test_loss = 0 
    correct = 0 
    ttime = 0
    idx = 0
    identified = 0 # target in output
    not_identified = 0 # target not in output
    wrongly_identified = 0 # output not in target
    not_identified_data = []
    with torch.no_grad():
        for data_V, data_C_d, target in test_loader: 
            data_V, data_C_d, target = data_V.to(device), data_C_d.to(device), target.to(device)
            output = model(data_V, data_C_d)
            for i in range(output.shape[0]):
                identified_tmp, not_identified_tmp, wrongly_identified_tmp = judge_in_area(output[i,...].view(1,2).cpu(), target[i,...].view(1,2).cpu())
                identified += identified_tmp
                not_identified += not_identified_tmp
                wrongly_identified += wrongly_identified_tmp
                idx += 1
                thr = 0.005
                if not_identified_tmp > thr or wrongly_identified_tmp > thr:
                    not_identified_data.append(data_V[i,0,:].tolist())
                    #print(data_V[i,0,:])
                    print(len(not_identified_data)/idx)
                    print(idx)

    with open('/mnt/mnt1/linxuan/nnv/ACC/new_train_data2_critical_not-identified_json/data_list3.json', 'w') as file:
        json.dump(not_identified_data, file)
    
    print("identified:",identified)
    print("not identified:",not_identified)
    print("wrongly identified:",wrongly_identified)

    return 0
          
def main():

    BATCH_SIZE = 512
    Epoch = 1

    train_loader,test_loader,val_loader = get_dataset(BATCH_SIZE)
    criterion = nn.L1Loss()

    loss = test(device, val_loader, criterion, 1)
    print("final loss:", loss)

main()