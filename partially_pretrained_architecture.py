import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim

def gen_alexnet(num_classes):

    # correct for binary classification configurations
    if num_classes == 2: 
        num_classes = 1 
    
    alexnet_model = models.alexnet(pretrained=True)

    conv_layers = alexnet_model.features

    # Enable further training on pre-tranied conv weights
    for param in conv_layers.parameters():
        param.requires_grad = True

    classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(256 * 3 * 3, 2048), 
        nn.LeakyReLU(),
        nn.Dropout(0.2),
        nn.Linear(2048, 2048),
        nn.LeakyReLU(),
        nn.Dropout(0.2),
        nn.Linear(2048, num_classes), # OBS! Adjust this number to fit the numbe of classes 
        nn.LogSoftmax(dim=1)
    )

    AlexNet = nn.Sequential(
        conv_layers,
        classifier
    )

    return AlexNet
