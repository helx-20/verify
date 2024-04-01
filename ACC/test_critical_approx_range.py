import numpy as np
import math
import time
import os,sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import time
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
import random
random.seed(0)

device = torch.device("cuda:0")

batch_size = 256
order = np.array(range(219186))
random.shuffle(order)
    
all_data = np.load("/mnt/mnt1/linxuan/nnv/ACC/dataset/critical_data4.npy", allow_pickle=True).item()
data = torch.tensor(all_data['data'][order[180000:200000],...],dtype=torch.float32)
label = torch.tensor(all_data['label'][order[180000:200000],...],dtype=torch.float32)

all_data_approx = np.load("/mnt/mnt1/linxuan/nnv/ACC/dataset/critical_data4_approx.npy", allow_pickle=True).item()
data_approx = torch.tensor(all_data_approx['data'][order[180000:200000],...],dtype=torch.float32)
label_approx = torch.tensor(all_data_approx['label'][order[180000:200000],...],dtype=torch.float32)
print()

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

def test():

    identified = 0 # target in output
    not_identified = 0 # target not in output
    wrongly_identified = 0 # output not in target
    identified_tmp, not_identified_tmp, wrongly_identified_tmp = judge_in_area(label_approx, label)
    identified += identified_tmp
    not_identified += not_identified_tmp
    wrongly_identified += wrongly_identified_tmp
    print("identified:",identified)
    print("not identified:",not_identified)
    print("wrongly identified:",wrongly_identified)
          
start_time = time.time()
test()
print("spend time:", time.time()-start_time)