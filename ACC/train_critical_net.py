import numpy as np
import math
import time
import os,sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from dataset import get_dataset
from model.critical_net import define_critical_net
from model.critical_net2 import define_critical_net2

device = torch.device("cuda:1")

print('model established')
model = define_critical_net(model='transformer', input_dim=2, output_dim=2, m_tokens_in=5, m_tokens_out=48, transformer_out_feature_dim=1024)
#model = define_critical_net2(model='transformer', input_dim=2, output_dim=2, m_tokens_in=5, m_tokens_out=48)
#model = torch.load("saved_model/critical_net.pth")
model = model.to(device)
print(model)

def train(model, device, train_loader, optimizer, scheduler, mask_criterion, criterion, criterion2, epoch):
    model.train()
    for batch_id, (data, target) in enumerate(train_loader): 
        data, target = data.to(device), target.to(device)
        mask_target = (target[:,:,0] != 0).float().view(target.shape[0],-1, 1)
        separate_train = False
        if separate_train:
            optimizer.zero_grad()
            mask, output = model(data)
            mask_loss = mask_criterion(mask, mask_target)
            mask_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            mask, output = model(data)
            set_loss = criterion(output, target)
            set_loss.backward()
            #loss = 5*mask_loss + set_loss
            #loss.backward()
            optimizer.step()
        else:
            optimizer.zero_grad()
            mask, output = model(data)
            mask_loss = mask_criterion(mask, mask_target)
            set_loss = criterion(output, target)
            loss = 20*mask_loss + set_loss
            loss.backward()
            #print(model[1].mask_net.blocks[0].attn.proj_k.weight.grad)
            #print(model[1].mask_net.blocks[0].attn.q.grad)
            optimizer.step()
        
    scheduler.step()

def test(model, device, test_loader, mask_criterion, criterion, criterion2, epoch):
    test_loss = 0 
    correct = 0 
    with torch.no_grad():
        for data, target in test_loader: 
            data, target = data.to(device), target.to(device)
            mask_target = (target[:,:,0] != 0).float().view(target.shape[0], -1, 1)
            mask, output = model(data)
            mask_loss = mask_criterion(mask, mask_target)
            thr = 1e-1 #(epoch > 5) * 1e-3 + (epoch > 20) * 1e-2 + (epoch > 50) * 1e-1
            critical_place = torch.where(((torch.abs(target)>thr)*(torch.abs(output)<thr)+(torch.abs(target)<thr)*(torch.abs(output)>thr))>0)
            set_loss = criterion(output, target) # + (epoch>5) * 1e-3 * criterion2(output[critical_place], target[critical_place])
            loss = 20*mask_loss + set_loss
            test_loss += loss.item()

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
            torch.save(model,'saved_model/critical_net7.pth')
            print("save")
        print("min_loss:", min_loss)   
    final_loss = test(best_model, device, test_loader, mask_criterion, criterion, criterion2, 1)
    print("final loss:", final_loss)

start_time = time.time()
main(model)
print("spend time:", time.time()-start_time)