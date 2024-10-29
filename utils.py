import os
import sys
import torch
import torch.nn as nn
import torchvision.models as models
#from architecture import AlexNet # Import our architecture which should be located in the biggan-am directory 
### Import our AlexNet architecture 
#from partially_pretrained_architecture import genAlexNet

top_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(top_dir)

from robustness import datasets, model_utils

dim_z_dict = {128: 120, 256: 140, 512: 128}
attn_dict = {128: "64", 256: "128", 512: "64"}
max_clamp_dict = {128: 0.83, 256: 0.61}
min_clamp_dict = {128: -0.88, 256: -0.59}
DATA_PATH_DICT = {
    "CIFAR": "/path/tools/cifar",
    "RestrictedImageNet": "/mnt/raid/qi/ILSVRC2012_img_train/ImageNet",
    "ImageNet": "/path/tools/imagenet",
    "H2Z": "/path/tools/horse2zebra",
    "A2O": "/path/tools/apple2orange",
    "S2W": "/path/tools/summer2winter_yosemite",
}


def get_config(resolution):
    return {
        "G_param": "SN",
        "D_param": "SN",
        "G_ch": 96,
        "D_ch": 96,
        "D_wide": True,
        # Embeddings from a new dataset is shared with the embeddings the model is trained on. 
        "G_shared": True,
        #This is the size of the class-embedding
        "shared_dim": 128,
        "dim_z": dim_z_dict[resolution],
        "hier": True,
        "cross_replica": False,
        "mybn": False,
        "G_activation": nn.ReLU(inplace=True),
        "G_attn": attn_dict[resolution],
        "norm_style": "bn",
        "G_init": "ortho",
        "skip_init": True,
        "no_optim": True,
        "G_fp16": False,
        "G_mixed_precision": False,
        "accumulate_stats": False,
        "num_standing_accumulations": 16,
        "G_eval_mode": True,
        "BN_eps": 1e-04,
        "SN_eps": 1e-04,
        "num_G_SVs": 1,
        "num_G_SV_itrs": 1,
        "resolution": resolution,
        "n_classes": 1000,
    }


def load_mit(model_name):
    model_file = f"{model_name}_places365.pth.tar"
    if not os.access(model_file, os.W_OK):
        weight_url = f"http://places2.csail.mit.edu/models_places365/{model_file}"
        os.system(f"wget {weight_url}")

    model = models.__dict__[model_name](num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {
        str.replace(k, "module.", ""): v for (k, v) in checkpoint["state_dict"].items()
    }
    model.load_state_dict(state_dict)

    return model


def load_madrylab_imagenet(arch):
    data = "ImageNet"
    dataset_function = getattr(datasets, data)
    dataset = dataset_function(DATA_PATH_DICT[data])
    model_kwargs = {
        "arch": arch,
        "dataset": dataset,
        "resume_path": f"madrylab_models/{data}.pt",
        "state_dict_path": "model",
    }
    (model, _) = model_utils.make_and_restore_model(**model_kwargs)

    return model


def load_net(model_name, model_path = None, num_classes = None, discriminator_save_dir = None):
   
    print(f"Loading {model_name} classifier...")
        
    def AlexNet(num_classes):
        
        alexnet_model = models.alexnet(pretrained=True)

        conv_layers = alexnet_model.features

        # Enable further training on pre-tranied conv weights
        for param in conv_layers.parameters():
            param.requires_grad = True

        classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 2048), 
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 2048),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, num_classes), 
            nn.LogSoftmax(dim=1)
        )

        alexnet = nn.Sequential(
            conv_layers,
            classifier
        )

        return(alexnet)
    
    alexnet_model = models.alexnet(pretrained=True)

    conv_layers = alexnet_model.features

    # enable further training on pre-tranied conv weights
    for param in conv_layers.parameters():
        param.requires_grad = True

    classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(256 * 3 * 3, 2048), 
        nn.LeakyReLU(),
        nn.Dropout(0.5),
        nn.Linear(2048, 2048),
        nn.LeakyReLU(),
        nn.Dropout(0.5),
        nn.Linear(2048, 30), 
        nn.LogSoftmax(dim=1)
    )

    alexnet = nn.Sequential(
        conv_layers,
        classifier
    )
    
    if model_name == "resnet50":
        return models.resnet50(pretrained=True)

    elif model_name == "alexnet":
        return models.alexnet(pretrained=True)

    elif model_name == "alexnet_conv5":
        return models.alexnet(pretrained=True).features

    elif model_name == "inception_v3":
        # Modified the original file in torchvision/models/inception.py!!!
        return models.inception_v3(pretrained=True)

    elif model_name == "mit_alexnet":
        return load_mit("alexnet")

    elif model_name == "mit_resnet18":
        return load_mit("resnet18")

    elif model_name == "madrylab_resnet50":
        return load_madrylab_imagenet("resnet50")
    #Our model
    elif model_name == "alexnet_Tetration_1":  
        model = models.alexnet(pretrained=False)

        # model.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        # sjunde lagret, convert from convolutional to linear layer? 5 for 6 classes. 
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 5)
        path = "Tetration_Alexnet_statedict2.pth"

        # load the state dictionary
        checkpoint = torch.load(path, map_location=torch.device('cpu'))

        # Load the adapted state dict into the model
        model.load_state_dict(checkpoint)
        
        return model
    
    elif model_name == "modelV4": 
        
        model = AlexNet(5)
        
        # Load the state dict (replace 'path_to_your_state_dict.pth' with your actual file path)
        state_dict = torch.load('Tetration_Alexnet_full_modelV4.pth')

        model.load_state_dict(state_dict)
        return model
    
    elif model_name == "binary_flowerV1": 
        
        model = AlexNet(num_classes =2)
        
        # Load the state dict (replace 'path_to_your_state_dict.pth' with your actual file path)
        state_dict = torch.load('Binary_flowersV1_dict.pth')

        model.load_state_dict(state_dict)
        return model
    elif model_name == "AlexNetMODV2": 

        model = AlexNet

        # Load the state dictionary
        model.load_state_dict(torch.load("StateDictV4.pth", map_location='cpu'))

        return model
    
    elif model_name == "pul_nod": 

        alexnet_model = models.alexnet(pretrained=True)

        conv_layers = alexnet_model.features

        # enable further training on pre-tranied conv weights
        for param in conv_layers.parameters():
            param.requires_grad = True

        classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 2048), 
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 2048),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 2), 
            nn.LogSoftmax(dim=1)
        )

        alexnet = nn.Sequential(
            conv_layers,
            classifier
        )

        # Load the state dictionary
        alexnet.load_state_dict(torch.load("pul_nod_model_dict.pth", map_location='cpu'))

        return alexnet
    
    elif model_name == "extended_flower": 
        # Load the state dictionary
        alexnet.load_state_dict(torch.load("10_classes_pul_nod_model_FINAL_DICT.pth", map_location='cpu'))

        return alexnet
    
    elif model_name == "50_classes_model": 
        # Load the state dictionary
        alexnet.load_state_dict(torch.load("50_classes_model_DICT.pth", map_location='cpu'))

        return alexnet

    elif model_name == "10_classes_model": 
        # Load the state dictionary
        alexnet.load_state_dict(torch.load("/mnt/c/Users/Gästkonto/Documents/Programmering/projekt/TetrationAI/classifiers/biggan_discriminator_own_dataset_state_dictionary.pth", map_location='cpu'))

        return alexnet
    
    elif model_name == "30_classes_model": 
        # Load the state dictionary
        alexnet.load_state_dict(torch.load("/mnt/c/Users/Gästkonto/Documents/Programmering/projekt/TetrationAI/classifiers/biggan_discriminator_30_own_dataset_state_dictionary.pth", map_location='cpu'))

        return alexnet
    
    elif model_path: 
        
        model = AlexNet(num_classes = num_classes)
        
        # Load the state dict (replace 'path_to_your_state_dict.pth' with your actual file path)
        save_model_dict_path = os.path.join(discriminator_save_dir, f'{model_name}_state_dictionary.pth')
        
        state_dict = torch.load(save_model_dict_path)

        model.load_state_dict(state_dict)
        return model



    else:
        raise ValueError(f"{model_name} is not a supported classifier...")