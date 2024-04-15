import numpy as np
import math
import time
import os,sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from dataset2 import get_dataset
from model.uncritical_net import Uncritical_Net

device = torch.device("cuda:1")

print('model established')
model = Uncritical_Net(input_dim_V=5, input_dim_C=323, output_dim=2, m_tokens_in=164)
model.load_state_dict(torch.load("saved_model/uncritical_net.pth").state_dict())
model = model.to(device)
print(model)

def train(model, device, train_loader, optimizer, scheduler, criterion, epoch):
    model.train()
    for batch_id, (data_V, data_C_d, target) in enumerate(train_loader): 
        data_V, data_C_d, target = data_V.to(device), data_C_d.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data_V, data_C_d)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
    scheduler.step()

def test(model, device, test_loader, criterion, epoch):
    test_loss = 0 
    correct = 0 
    with torch.no_grad():
        for data_V, data_C_d, target in test_loader: 
            data_V, data_C_d, target = data_V.to(device), data_C_d.to(device), target.to(device)
            output = model(data_V, data_C_d)
            loss = criterion(output, target)
            test_loss += loss.item()

    test_loss /= len(test_loader.dataset) 
    print('\nTest set: Epoch: {}, Average loss: {:.8f}\n'.format(epoch, test_loss))
    return test_loss
          
def main(model):

    BATCH_SIZE = 512
    Epoch = 150

    train_loader,test_loader,val_loader = get_dataset(BATCH_SIZE)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,100], gamma=0.2)
    criterion = nn.L1Loss()

    min_loss = 1000
    for e in range(Epoch):
        print("Epoch {} begin".format(e))
        train(model, device, train_loader, optimizer, scheduler, criterion, e)
        loss = test(model, device, val_loader, criterion, e)
        if loss < min_loss:
            min_loss = loss
            best_model = model
            torch.save(model,'saved_model/uncritical_net2.pth')
            print("save")
        print("min_loss:", min_loss)
    final_loss = test(best_model, device, test_loader, criterion, 1)
    print("final loss:", final_loss)

start_time = time.time()
main(model)
print("spend time:", time.time()-start_time)