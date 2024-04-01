import numpy as np
import math
import time
import os,sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from dataset2 import get_dataset
from model.uncritical_net import define_uncritical_net

device = torch.device("cuda:0")

print('model established')
model = define_uncritical_net(model='transformer', input_dim=2, output_dim=2, m_tokens_in=5, m_tokens_out=1, transformer_out_feature_dim=256)
model.load_state_dict(torch.load("saved_model/uncritical_net.pth").state_dict())
model = model.to(device)
print(model)

def train(model, device, train_loader, optimizer, scheduler, criterion, epoch):
    model.train()
    for batch_id, (data, target) in enumerate(train_loader): 
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        #print(torch.mean(output).item(),torch.mean(target).item())
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()
        
    scheduler.step()

def test(model, device, test_loader, criterion, epoch):
    test_loss = 0 
    correct = 0 
    with torch.no_grad():
        for data, target in test_loader: 
            data, target = data.to(device), target.to(device) 
            output = model(data)
            #print(target, output)
            for i in range(10):
                print(output[i,0], target[i,0])
            print('next')
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

    loss = test(model, device, train_loader, criterion, 1)
    print("final loss:", loss)

start_time = time.time()
main(model)
print("spend time:", time.time()-start_time)