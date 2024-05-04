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
model.load_state_dict(torch.load("saved_model/critical_net8_state_dict.pth"))
model = model.to(device)
print(model)

def test(model, device, test_loader, criterion, epoch):
    test_loss = 0 
    correct = 0 
    ttime = 0
    idx = 0
    with torch.no_grad():
        for data_V, data_C_d, target in test_loader: 
            data_V, data_C_d, target = data_V.to(device), data_C_d.to(device), target.to(device)
            start_time = time.time()
            output = model(data_V, data_C_d)
            ttime += time.time() - start_time
            idx += data_V.shape[0]
            for i in range(10):
                print(output[i,0].item(), target[i,0].item())
                print(output[i,1].item(), target[i,1].item())
                print('next')
            print("time:",ttime/idx)
            loss = criterion(output, target)
            test_loss += loss.item()
    
    print("time:",ttime/idx)
    test_loss /= len(test_loader.dataset) 
    print('\nTest set: Epoch: {}, Average loss: {:.8f}\n'.format(epoch, test_loss))
    return test_loss
          
def main(model):

    BATCH_SIZE = 256
    train_loader,test_loader,val_loader = get_dataset(BATCH_SIZE)
    criterion = nn.L1Loss()
    loss = test(model, device, test_loader, criterion, 1)
    print("final loss:", loss)

main(model)