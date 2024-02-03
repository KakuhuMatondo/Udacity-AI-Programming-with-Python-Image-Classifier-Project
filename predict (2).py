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
def load_checkpoint(file):
    model= models.vgg19(pretrained=True)
    
    checkpoint=torch.load(file)
    lr=checkpoint['learning_rate']
    model.classifier=checkpoint['classifier']
    model.class_to_idx=checkpoint['class_to_idx']
    model.state_dict(checkpoint['state_dict'])
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    optimizer.state_dict(checkpoint['optimizer_state_dict'])
    input_size=checkpoint['input_size']
    output_size=checkpoint['output_size']
    epoch=checkpoint['epoch']
    
    return model, optimizer, input_size, output_size, epoch

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
   
    l,h =image.size
    ratio=l/h
    if ratio>1:
        image=image.resize((round(256/ratio),256))
    else:
        image=image.resize((256,round(256/ratio)))
    l,h=image.size
    new_l=224
    new_h=224
    image.crop((((l-new_l)/2),((h-new_h)/2),((l+new_l)/2),((h+new_h)/2)))
       
        
    np_image = np.array(image)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image=(np_image-mean)/std
        
    return np_image.transpose((2,0,1))
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    image=Image.open(image_path)
    np_image=process_image(image)
    model.to(device)
    model.eval()
    with torch.no_grad():
            images=torch.from_numpy(np_image)
            images=images.unsqueeze(0)
            images=images.type(torch.FloatTensor)
            images=images.to(device)
            output=model.forward(images)
            class_probabilities=torch.exp(output)
            probabilities, indices =torch.topk(class_probabilities,topk)
            probabilities=[probability for probability in probabilities[0]]
            conversion={i : j for j, i in model.class_to_idx.items()}
            classes=[conversion[int(k)] for k in indices[0]]
            
            return probabilities, classes
        
parser=argparse.ArgumentParser()
parser.add_argument('image_path', action='store',nargs='?', default="/1/image_06743.jpg", help='path to image')
parser.add_argument('checkpoint_path', action='store',default='checkpoint.pth', help='checkpoint path')
parser.add_argument('--topk',action='store',default=5, help='Top flower probabilities', dest='topk')
parser.add_argument('--category_names', action='store', default='cat_to_name.json',help='get json file', dest='category_names')
parser.add_argument('--gpu', action='store_true', default=False, help='select GPU', dest='gpu')

arg=parser.parse_args()
arg_image_path=arg.image_path
arg_checkpoint_path=arg.checkpoint_path
arg_topk=arg.topk
arg_category_names=arg.category_names

if torch.cuda.is_available() and arg.gpu:
    arg_gpu=arg.gpu
else: 
    arg_gpu=False
    
arg_gpu=arg.gpu

device=torch.device('cuda' if arg_gpu else 'cpu')

with open(arg_category_names, 'r') as f:
    cat_to_name=json.load(f)
model, optimizer, input_size, output_size,epoch=load_checkpoint(arg_checkpoint_path)
model.eval()
idx_to_class={ i:j for j, i in model.class_to_idx.items()}

probabilities, classes = predict(arg_image_path,model,arg_topk)
print('This flower is : ')
for i in range(arg_topk):
    print('{}         {:.3f}%'.format(cat_to_name[idx_to_class[classes[0,i].item()]], probabilities[0,i].item()))
    