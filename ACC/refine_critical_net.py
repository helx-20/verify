import numpy as np
import math
import time
import os,sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from dataset import get_dataset
from model.critical_net import Critical_Net

device = torch.device("cuda:0")

print('model established')
model = Critical_Net(input_dim_V=5, input_dim_C=323, output_dim=2, m_tokens_in=164)
model.load_state_dict(torch.load("saved_model/critical_net_state_dict.pth"))
model = model.to(device)
print(model)

def train(model, device, train_loader, optimizer, scheduler, criterion, epoch, criterion2 = nn.L1Loss(reduction='sum')):
    model.train()
    for batch_id, (data_V, data_C_d, target) in enumerate(train_loader): 
        data_V, data_C_d, target = data_V.to(device), data_C_d.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data_V, data_C_d)
        r = 0
        critical = (target[:,0] - target[:,1] < output[:,0] - output[:,1] - r) + (target[:,0] + target[:,1] > output[:,0] + output[:,1] + r)
        criticality = torch.sum(torch.clamp(output[:,0]-output[:,1]-target[:,0]+target[:,1], min=0)) + torch.sum(torch.clamp(target[:,0]+target[:,1]-output[:,0]-output[:,1], min=0))
        loss = criterion(output, target) + 1.9e-2 * criticality
        print(criticality)
        print(loss.item())
        loss.backward()
        optimizer.step()
        
    scheduler.step()

def test(model, device, test_loader, criterion, epoch, criterion2 = nn.L1Loss(reduction='sum')):
    test_loss = 0 
    correct = 0 
    with torch.no_grad():
        for data_V, data_C_d, target in test_loader: 
            data_V, data_C_d, target = data_V.to(device), data_C_d.to(device), target.to(device)
            output = model(data_V, data_C_d)
            r = 0
            critical = (target[:,0] - target[:,1] < output[:,0] - output[:,1] - r) + (target[:,0] + target[:,1] > output[:,0] + output[:,1] + r)
            criticality = torch.sum(torch.clamp(output[:,0]-output[:,1]-target[:,0]+target[:,1], min=0)) + torch.sum(torch.clamp(target[:,0]+target[:,1]-output[:,0]-output[:,1], min=0))
            regularization = torch.sum(torch.clamp(output[:,1]-target[:,1], min=0))
            critical_place = torch.where(critical > 0)
            #loss = criterion(output, target) + 1e-1*criterion2(output[critical_place], target[critical_place])
            loss = criterion(output, target) + 1.9e-2 * criticality #+ 1e-3 * regularization
            test_loss += loss.item()

    test_loss /= len(test_loader.dataset) 
    print('\nTest set: Epoch: {}, Average loss: {:.8f}\n'.format(epoch, test_loss))
    return test_loss
          
def main(model):

    BATCH_SIZE = 512
    Epoch = 100

    train_loader,test_loader,val_loader = get_dataset(BATCH_SIZE)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50], gamma=0.2)
    criterion = nn.L1Loss()

    min_loss = 1000
    for e in range(Epoch):
        print("Epoch {} begin".format(e))
        train(model, device, train_loader, optimizer, scheduler, criterion, e)
        loss = test(model, device, val_loader, criterion, e)
        if loss < min_loss:
            min_loss = loss
            best_model = model
            torch.save(model.state_dict(),'saved_model/critical_net2_state_dict.pth')
            print("save")
        print("min_loss:", min_loss)
    final_loss = test(best_model, device, test_loader, criterion, 1)
    print("final loss:", final_loss)

start_time = time.time()
main(model)
print("spend time:", time.time()-start_time)