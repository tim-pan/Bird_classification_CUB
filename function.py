'''
this file includes parts of functions that should be used in homework1
'''

from tqdm import tqdm
import random
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR

#create tensor2class dict for us to convert tensor to class easily
with open('./dataset/classes.txt') as fp:
    li = fp.read().split()
    tensor2class = dict(zip(range(0, 200), li))
    
def ind2class(tensor):
    #convert tensor to class
    #for example
    #input:tensor[1, 2, 3]
    #output will be:['1. blue_bird', '2. green bird, '3. yellow_bird']
    final_label = []
    for i in tensor:
        final_label.append(tensor2class[i.item()])
    return final_label

def evaluate(model, data_loader):
    '''
    loss:cross entropy
    this function aims to evalute the model performance wrt a specific data loader
    *input: 
    model, data loader
    *output:
    accuracy, loss, predicted index(eg:[0, 1, 2, 1])
    '''
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    model.eval()
    acc = 0
    num = 0
    total_loss = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device).float(), y.to(device)
            out = model(x)
            loss = criterion(out, y.long())
            total_loss += loss.item()
            num += y.size(0)
            
            pred_ind = torch.max(out, 1).indices.view(-1)
            acc += (pred_ind == y).sum().item()
            
        acc /= num
        # total_loss /= len(data_loader)
        total_loss /= num
        pred_cla = ind2class(pred_ind)
        return acc, total_loss, pred_cla

def evaluate_new(model, data_loader):
    '''
    loss:cross entropy+cos loss
    this function aims to evalute the model performance wrt a specific data loader
    *input: 
    model, data loader
    *output:
    accuracy, loss, predicted index(eg:[0, 1, 2, 1])
    '''
    criterion = nn.CosineSimilarity()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    model.eval()
    acc = 0
    num = 0
    total_loss = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device).float(), y.to(device)
            out = model(x)
            #calculate loss part
            onehot_y = F.one_hot(y.view(-1).long(), num_classes=200)
            loss_cos = torch.sum(1 - criterion(out, onehot_y))
            loss_cro = nn.CrossEntropyLoss()(out, y.long())
            loss = loss_cos + loss_cro
            loss = loss_cos
            total_loss += loss.item()
            num += y.size(0)
            #eval
            pred_ind = torch.max(out, 1).indices.view(-1)
            acc += (pred_ind == y).sum().item()
            
        acc /= num
        # total_loss /= len(data_loader)
        total_loss /= num
        pred_cla = ind2class(pred_ind)
        return acc, total_loss, pred_cla

def train(model, optimizer, scheduler, criterion, train_loader, dev_loader, EPOCHS, MODELNAME):
    '''
    training procedure based on cross entropy loss(or your preferable loss)
    *input:
    model: model
    optimizer: SGD or Adam or st else
    scheduler: make learning rate designable, for eg, ExponentialLR
    criterion:default is cross entropy loss
    train_loader: training set
    dev_loader: validation set
    EPOCHS: total epochs
    MODELNAME: the place where you save your model
    '''
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion=criterion
    
    optimizer = optimizer
    scheduler = scheduler
    
    model.to(device)
    model.train()
    
    best_acc = 0
    for epoch in range(EPOCHS):
        total_loss = 0
        train_acc=0
        num = 0
        for x, y in tqdm(train_loader):
            x = x.to(device).float()
            y = y.to(device)
            
            out = model(x)#[bs, 200]
            
            #calculate loss part
            loss = criterion(out, y.long())
            total_loss += loss.item()
            num += y.size(0)
            
            #calculate train acc
            pred_ind = torch.max(out, 1).indices.view(-1)
            train_acc += (pred_ind == y).sum().item()
            
            #backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step() 

        epoch_loss = total_loss / num
        train_acc /= num
        test_acc, test_loss, _ = evaluate(model, dev_loader)
        # if best_acc<test_acc:
            # best_acc = test_acc
        torch.save(model.state_dict(), f'./model-para/{MODELNAME}/ep{epoch} acc{train_acc:.2f} {test_acc:.2f}.pt')
        
        if (epoch + 1) % 1 == 0:
            print(f'epoch {epoch+1}%%%%%%%%%%%%%%%%%%%%')
            print(f"train set===>loss: {epoch_loss:.4f}   acc: {100* train_acc:.2f}%")
            print(f"dev set=====>loss: {test_loss:.4f}   acc: {100* test_acc:.2f}%")
            print()
            
def train_new(model, optimizer, scheduler, criterion, train_loader, dev_loader, EPOCHS, MODELNAME):
    '''
    training procedure based on cos loss + cross entropy loss
    *input:
    model: model
    optimizer: SGD or Adam or st else
    scheduler: make learning rate designable, for eg, ExponentialLR
    criterion:default is cross entropy loss+cos loss
    train_loader: training set
    dev_loader: validation set
    EPOCHS: total epochs
    MODELNAME: the place where you save your model
    '''
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    criterion=criterion
    optimizer = optimizer
    scheduler = scheduler
    
    model.to(device)
    model.train()
    
    # best_acc = 0
    for epoch in range(EPOCHS):
        total_loss = 0
        train_acc=0
        num = 0
        for x, y in tqdm(train_loader):
            x = x.to(device).float()
            y = y.to(device)
            
            out = model(x)#[bs, 200]
            
            #calculate loss part
            onehot_y = F.one_hot(y.view(-1).long(), num_classes=200)
            loss_cos = torch.sum(1 - criterion(out, onehot_y))
            loss_cro = nn.CrossEntropyLoss()(out, y.long())
            loss = loss_cos + loss_cro
            loss = loss_cos
            total_loss += loss.item()
            num += y.size(0)
            
            #calculate train acc
            pred_ind = torch.max(out, 1).indices.view(-1)
            train_acc += (pred_ind == y).sum().item()
            
            #backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step() 

        epoch_loss = total_loss / num
        train_acc /= num
        test_acc, test_loss, _ = evaluate_new(model, dev_loader)
        # if best_acc<test_acc:
            # best_acc = test_acc
        torch.save(model.state_dict(), f'./model-para/{MODELNAME}/ep{epoch} acc{train_acc:.2f} {test_acc:.2f}.pt')
        
        if (epoch + 1) % 1 == 0:
            print(f'epoch {epoch+1}%%%%%%%%%%%%%%%%%%%%')
            print(f"train set===>loss: {epoch_loss:.4f}   acc: {100* train_acc:.2f}%")
            print(f"dev set=====>loss: {test_loss:.4f}   acc: {100* test_acc:.2f}%")
            print()
            
def class2txt(sub):
    '''
    write your true labels to "answer.txt"
    *input:
    sub:submission, a list of predicted classes of testing data, for eg: ['1. yellow _bird', '2. red_bird']
    *output:
    no output, but generate a "answer.txt" file 
    '''
    #input:the list of predicted classes of testing data
    #note:please sure your testing data order is correct, shuffle=false
    with open('./dataset/testing_img_order.txt') as f:
         test_images = [x.strip() for x in f.readlines()] # names of all the testing images
    submission = []
    for img, predicted_class in zip(test_images, sub):# image order is important to your result
        submission.append([img, predicted_class])
    
    np.savetxt('answer.txt', submission, fmt='%s')    