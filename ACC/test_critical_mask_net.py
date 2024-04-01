import numpy as np
import math
import time
import os,sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from dataset import get_dataset
import matplotlib.pyplot as plt
from model.critical_mask_net import define_critical_mask_net

device = torch.device("cuda:0")

print('model established')
model = define_critical_mask_net(model='transformer', input_dim=2, output_dim=2, m_tokens_in=5, m_tokens_out=48, transformer_out_feature_dim=1024)
model.load_state_dict(torch.load("saved_model/critical_mask_net.pth").state_dict())
model = model.to(device)
print(model)

def train(model, device, train_loader, optimizer, scheduler, criterion, epoch):
    model.train()
    for batch_id, (data, target) in enumerate(train_loader): 
        data, target = data.to(device), target.to(device)
        mask_target = (target[:,:,0] != 0).float().view(target.shape[0], -1, 1)
        optimizer.zero_grad()
        mask = model(data)
        #print(torch.mean(output).item(),torch.mean(target).item())
        loss = criterion(mask, mask_target)

        loss.backward()
        optimizer.step()
        
    scheduler.step()

def test(model, device, test_loader, criterion, epoch):
    test_loss = 0 
    correct = 0 
    TP = 0
    FN = 0
    FP = 0
    thresh = torch.tensor(range(0,101,5))/100
    TP = torch.zeros(thresh.shape[0])
    FN = torch.zeros(thresh.shape[0])
    FP = torch.zeros(thresh.shape[0])
    with torch.no_grad():
        for data, target in test_loader: 
            data, target = data.to(device), target.to(device) 
            mask_target = (target[:,:,0] != 0).float().view(target.shape[0], -1, 1)
            mask = model(data)
            for i in range(thresh.shape[0]):
                TP[i] += torch.sum((mask.cpu() >= thresh[i]) * (mask_target.cpu() == 1))
                FN[i] += torch.sum((mask.cpu() < thresh[i]) * (mask_target.cpu() == 1))
                FP[i] += torch.sum((mask.cpu() >= thresh[i]) * (mask_target.cpu() == 0))
            #for i in range(10):
            #    print(mask[0,i], mask_target[0,i])
            #print('next')
            loss = criterion(mask, mask_target)
            test_loss += loss.item()

    recall = TP/(TP+FN)
    precision = TP/(TP+FP)
    test_loss /= len(test_loader.dataset) 
    print('\nTest set: Epoch: {}, Average loss: {:.8f}\n'.format(epoch, test_loss))
    print("recall:",recall[int(thresh.shape[0]/2)])
    print("precision:",precision[int(thresh.shape[0]/2)])
    plt.plot(recall, precision)
    size = 12
    plt.xlabel("recall",fontsize=size)
    plt.ylabel("precision",fontsize=size)
    plt.title("mask net")
    plt.savefig("critical_mask_net.png",dpi=300)
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