import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50
import torch.optim as optim
from torchvision.utils import save_image
import torch.utils.data as Data
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
from PIL import Image
import json


def adv_example(img_tensor):
    """
    Create an adversarial example
    """

    # Find the prediction of true class
    pred_vector = model(img_tensor) 
    max_value, max_class = pred_vector[0].max(0)
    print("true class: ", image_class[str(max_class.item())][1])

    e = 5./255 # Set the maximum limit of the perturbation 
    delta = torch.zeros_like(img_tensor, requires_grad=True) # The small perturbation that will be added on the original image tensor
    opt = optim.SGD([delta], lr=0.01) # optimizer

    for t in range(50):
        pred_vector = model(img_tensor + delta)
        loss = -nn.CrossEntropyLoss()(pred_vector, torch.LongTensor([101])) # "Maximize" the loss for true class so that the probability of being in that class will be extremely small with the modified image.

        # Zero the parameter gradients
        opt.zero_grad()

        # Backward
        loss.backward()

        # Updates the noise
        opt.step()

        # Clamp the noise to be withing the limit
        delta.data.clamp_(-e, e)

    max_value, max_class = pred_vector[0].max(0) # Find the max class from the prediction
    print("true class probability:", nn.Softmax(dim=1)(pred_vector)[0,101].item())
    print("predicted class: ", image_class[str(max_class.item())][1])
    print("predicted class probability:", nn.Softmax(dim=1)(pred_vector)[0,max_class.item()].item())

    plt.imshow((img_tensor + delta)[0].detach().numpy().transpose(1,2,0))
    plt.savefig('adv_example.png')

def target_adv_example(img_tensor):

    # Find the prediction of true class
    pred_vector = model(img_tensor)
    max_value, max_class = pred_vector[0].max(0)
    print("true class: ", image_class[str(max_class.item())][1])

    e = 5./255 # Set the maximum limit of the perturbation 
    delta = torch.zeros_like(img_tensor, requires_grad=True) # The small perturbation that will be added on the original image tensor
    opt = optim.SGD([delta], lr=0.02) # optimizer
    for t in range(50):
        pred_vector = model(img_tensor + delta)
        loss = -nn.CrossEntropyLoss()(pred_vector, torch.LongTensor([101])) + nn.CrossEntropyLoss()(pred_vector, torch.LongTensor([466])) # "Maximize" the loss for true class so that the probability of being in that class will be extremely small with the modified image. On top of that, we also "minimize" the loss for the aimed class to increase the probability of being in that class.

        # Zero the parameter gradients
        opt.zero_grad()

        # Backward
        loss.backward()

        # Updates the noise
        opt.step()

        # Clamp the noise to be withing the limit
        delta.data.clamp_(-e, e)

    max_value, max_class = pred_vector[0].max(0) # Find the max class from the prediction
    print("true class probability:", nn.Softmax(dim=1)(pred_vector)[0,101].item())
    print("predicted class: ", image_class[str(max_class.item())][1])
    print("predicted class probability:", nn.Softmax(dim=1)(pred_vector)[0,max_class.item()].item())

    plt.imshow((img_tensor + delta)[0].detach().numpy().transpose(1,2,0))
    plt.savefig('target_adv_example.png')


if __name__=='__main__':

    with open('imagenet_class_index.json', 'r') as json_file:
        data=json_file.read()

    # parse file
    image_class = json.loads(data)
    
    # pre-trained model
    model = resnet50(pretrained=True)
    model.eval()

    img = Image.open('Elephant2.jpg')
    pre_process = transforms.Compose([
       transforms.Resize(224),
       transforms.ToTensor(),
    ])
    img_tensor = pre_process(img)[None,:,:,:]
    plt.imshow(img_tensor[0].numpy().transpose(1,2,0))
    plt.savefig('original_img.png')
    
    adv_example(img_tensor)
    target_adv_example(img_tensor)
