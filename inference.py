'''
please read readme.md first.

'''
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import birdloader
from birdloader import BirdImage as BirdImage
import function as fn
'''
this procedure will generate an answer.txt in your current folder, 
please back to your current folder, and check answer.txt 
'''
#prepare data
testset = BirdImage('test')
test_loader = DataLoader(dataset = testset,
                        batch_size = 16,
                        shuffle = False,
                        num_workers = 0,
                        pin_memory = True)
#load data
model1 = models.resnet152(pretrained=False)
model1.fc = nn.Linear(2048, 200)

model2 = models.resnet152(pretrained=False)
model2.fc = nn.Linear(2048, 200)

model3 = models.resnet152(pretrained=False)
model3.fc = nn.Linear(2048, 200)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model1.to(device)
model1.eval()
model1.load_state_dict(torch.load('./model-para/ep8 acc1.00 0.65.pt', map_location=device))

model2.to(device)
model2.eval()
model2.load_state_dict(torch.load('./model-para/ep30 acc0.99 0.65.pt', map_location=device))

model3.to(device)
model3.eval()
model3.load_state_dict(torch.load('./model-para/ep8_finetune.pt', map_location=device))


#start evaluating
final = torch.empty(0).to(device)
for x in test_loader:
  x = x.to(device).float()
  out1 = model1(x)
  out2 = model2(x)
  out3 = model3(x)
  out = out1 + out2 + out3
  out = torch.max(out, 1).indices.view(-1)
  final = torch.cat([final, out], dim=0)
  
#convert index to bird name
final = fn.ind2class(final)
#write final to answer.txt
fn.class2txt(final)

 