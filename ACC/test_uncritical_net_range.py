import numpy as np
import math
import time
import os,sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import time
from dataset2 import get_dataset
from model.uncritical_net import define_uncritical_net

device = torch.device("cuda:0")

print('model established')
model = define_uncritical_net(model='transformer', input_dim=2, output_dim=2, m_tokens_in=5, m_tokens_out=1, transformer_out_feature_dim=256)
model.load_state_dict(torch.load("saved_model/uncritical_net.pth").state_dict())
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

def transfer_range(data): # data = 1 * 2
    transfered_data = []
    data_start = (data[0,0] - data[0,1]).item()
    data_end = (data[0,0] + data[0,1]).item()
    transfered_data.append(data_start)
    transfered_data.append(data_end)
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

def test(model, device, test_loader, criterion, epoch):
    test_loss = 0 
    correct = 0 
    identified = 0 # target in output
    not_identified = 0 # target not in output
    wrongly_identified = 0 # output not in target
    with torch.no_grad():
        for data, target in test_loader: 
            data, target = data.to(device), target.to(device) 
            output = model(data)
            identified_tmp, not_identified_tmp, wrongly_identified_tmp = judge_in_area(output.cpu(), target.cpu())
            identified += identified_tmp
            not_identified += not_identified_tmp
            wrongly_identified += wrongly_identified_tmp
            print("identified:",identified)
            print("not identified:",not_identified)
            print("wrongly identified:",wrongly_identified)
            #print(target, output)
            #for i in range(10):
            #    print(output[i,0], target[i,0])
            #print('next')
            loss = criterion(output, target)
            test_loss += loss.item()

    test_loss /= len(test_loader.dataset) 
    print('\nTest set: Epoch: {}, Average loss: {:.8f}\n'.format(epoch, test_loss))
    return test_loss
          
def main(model):

    BATCH_SIZE = 256
    Epoch = 1

    train_loader,test_loader,val_loader = get_dataset(BATCH_SIZE)
    criterion = nn.L1Loss()

    loss = test(model, device, test_loader, criterion, 1)
    print("final loss:", loss)

start_time = time.time()
main(model)
print("spend time:", time.time()-start_time)