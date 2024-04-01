import numpy as np
import math
import time
import os,sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from dataset import get_dataset
from model.critical_set_net import define_critical_set_net

device = torch.device("cuda:0")

print('model established')
model = define_critical_set_net(model='transformer', input_dim=2, output_dim=2, m_tokens_in=5, m_tokens_out=48, transformer_out_feature_dim=1024)
#model.load_state_dict(torch.load("saved_model/critical_set_net.pth").state_dict())
model = model.to(device)
print(model)

def train(model, device, train_loader, optimizer, scheduler, mask_criterion, criterion, criterion2, epoch):
    model.train()
    for batch_id, (data, target) in enumerate(train_loader): 
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        critical_place = torch.where(target != 0)
        set_loss = criterion(output[critical_place], target[critical_place])
        set_loss.backward()
        optimizer.step()
        
    scheduler.step()

def test(model, device, test_loader, mask_criterion, criterion, criterion2, epoch):
    test_loss = 0 
    correct = 0 
    with torch.no_grad():
        for data, target in test_loader: 
            data, target = data.to(device), target.to(device)
            output = model(data)
            critical_place = torch.where(target != 0)
            set_loss = criterion(output[critical_place], target[critical_place])
            test_loss += set_loss.item()

    test_loss /= len(test_loader.dataset) 
    print('\nTest set: Epoch: {}, Average loss: {:.8f}\n'.format(epoch, test_loss))
    return test_loss
          
def main(model):

    BATCH_SIZE = 256
    Epoch = 900

    train_loader,test_loader,val_loader = get_dataset(BATCH_SIZE)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300,600], gamma=0.2)
    #criterion = nn.MSELoss()
    mask_criterion = nn.MSELoss()
    criterion = nn.L1Loss()
    criterion2 = nn.L1Loss(reduction='sum')

    min_loss = 1000
    for e in range(Epoch):
        print("Epoch {} begin".format(e))
        train(model, device, train_loader, optimizer, scheduler, mask_criterion, criterion, criterion2, e)
        loss = test(model, device, val_loader, mask_criterion, criterion, criterion2, e)
        if loss < min_loss:
            min_loss = loss
            best_model = model
            torch.save(model,'saved_model/critical_set_net4.pth')
            print("save")
        print("min_loss:", min_loss)   
    final_loss = test(best_model, device, test_loader, mask_criterion, criterion, criterion2, 1)
    print("final loss:", final_loss)

start_time = time.time()
main(model)
print("spend time:", time.time()-start_time)