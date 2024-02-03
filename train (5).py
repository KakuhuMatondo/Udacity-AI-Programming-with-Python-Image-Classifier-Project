import argparse
import numpy as np
import torch
from torchvision import datasets,transforms, models

from torch import nn, optim

from collections import OrderedDict
import time
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib
import json





parser=argparse.ArgumentParser()

parser.add_argument( 'data_directory', action='store',default='flowers',help='Set directory to load training data')

parser.add_argument('--save_dir', action='store', default='.',dest='save_dir', help='Sat directory to sace checkpoint')

parser.add_argument('--arch', action='store', default='vgg19', dest='arch', help='Choose architecture')

parser.add_argument('--learning_rate',action='store',default= 0.001, dest='learning_rate', help='Choose learning rate')

parser.add_argument('--hidden_units', action='store', nargs=3, type=list, default=[4096,4096,1000], dest='hidden_units', help='choose hidden units')

parser.add_argument('--epochs', action='store', default=5, dest='epochs',help='Choose number of ephochs')

parser.add_argument('--gpu', action='store_true', default= False, dest='gpu', help='Use GPU for training set to true' )

parse_results=parser.parse_args()

data_dir=parse_results.data_directory
save_dir=parse_results.save_dir
arch=parse_results.arch
learning_rate=parse_results.learning_rate
hidden_units=parse_results.hidden_units
epochs=int(parse_results.epochs)

if torch.cuda.is_available() and parse_results.gpu:
      gpu=parse_results.gpu
else: 
      gpu= False
gpu=parse_results.gpu

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
train_transforms = transforms.Compose([
                              transforms.RandomRotation(30),
                              transforms.Resize(224),
                              transforms.CenterCrop(224),
                              transforms.ToTensor(),  
                              transforms.Normalize((0.485,0.465,0.406),
                                                   (0.229,0.224,0.225))
                              ])
test_transforms = transforms.Compose([
                              transforms.Resize(224),
                              transforms.CenterCrop(224),
                              transforms.ToTensor(),  
                              transforms.Normalize((0.485,0.465,0.406), (0.229,0.224,0.225))
                              ])
                              
valid_transforms = transforms.Compose([
                              transforms.Resize(224),
                              transforms.CenterCrop(224),
                              transforms.ToTensor(),  
                              transforms.Normalize((0.485,0.465,0.406), (0.229,0.224,0.225))
                              ])
# TODO: Load the datasets with ImageFolder
image_datasets={}
image_datasets['train_data'] = datasets.ImageFolder( train_dir, transform=train_transforms)
image_datasets['test_data'] = datasets.ImageFolder(test_dir, transform=test_transforms)
image_datasets['valid_data']= datasets.ImageFolder(valid_dir, transform=valid_transforms)
# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(image_datasets['train_data'], batch_size=64,shuffle=True)
testloader = torch.utils.data.DataLoader(image_datasets['test_data'] , batch_size=64)
validloader= torch.utils.data.DataLoader(image_datasets['valid_data'], batch_size=64)


model= models.vgg19(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

from collections import OrderedDict
classifier = nn.Sequential(
          nn.Linear(in_features=25088, out_features=hidden_units[0], bias=True),
          nn.ReLU(inplace=True),
          nn.Dropout(p=0.5, inplace=False),
          nn.Linear(in_features=hidden_units[0], out_features=hidden_units[1], bias=True),
          nn.ReLU(inplace=True),
          nn.Linear(in_features=hidden_units[1], out_features=hidden_units[2], bias=True),
          nn.ReLU(inplace=True),
          nn.Dropout(p=0.5, inplace=False),
          nn.Linear(in_features=hidden_units[2], out_features=102, bias=True),
          nn.LogSoftmax(dim=1)
  )
    
model.classifier = classifier
device=torch.device('cuda' if gpu else 'cpu')
model.to(device)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

epochs = epochs
steps = 0
running_loss = 0
print_every = 5
for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            validation_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    validation_loss += batch_loss.item()
                    
                    
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {validation_loss/len(validloader):.3f}.. "
                  f"Validation accuracy: {accuracy/len(validloader):.3f}")
            running_loss = 0
            model.train()
            
test_loss=0
accuracy=0

with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    test_loss += batch_loss.item()
                    
                    
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                print(
                  f"Test loss: {test_loss/len(testloader):.3f}.. "
                  f"Test accuracy: {accuracy/len(testloader):.3f}")
                model.train()
                
model.class_to_idx=image_datasets['train_data'].class_to_idx
checkpoint={ 'input_size' : 25088,
            'output_size' : 102,
            'model': model,
            'state_dict': model.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict,
            'criterion': criterion,
            'epochs' : epochs,
            'class_to_idx' : model.class_to_idx}


torch.save(checkpoint, save_dir +'/checkpoint.pth')
if save_dir=='.':
    save_dir_name='current folder'
else:
    save_dir_name=save_dir + "folder"

print(f' Checkpoint is saved to {save_dir_name}')

def load_checkpoint(file):
    checkpoint=torch.load(file, device)
    model=checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    return model

    model=checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    return model
saved_model=load_checkpoint(save_dir)
print(saved_model)
