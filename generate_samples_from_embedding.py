import BigGAN
import numpy as np
import sys

from torchvision.utils import save_image
from utils import *


def main():
    # Set up cudnn.benchmark for free speed.
    torch.backends.cudnn.benchmark = True
    device = "cuda:0"

    print("Loading the BigGAN generator model...", flush=True) # check directory structure 
    resolution = 128
    config = get_config(resolution)
    G = BigGAN.Generator(**config)
    G.load_state_dict(
        torch.load("biggan-am/pretrained_weights/138k/G_ema.pth"), strict=False
    )
    G = nn.DataParallel(G).to(device)
    G.eval()

    #################### REPLACED
    """ torch.backends.cudnn.benchmark = True
        device = "cpu"  # Change this line to use the CPU
        print("Loading the BigGAN generator model...", flush=True)
        resolution = 256
        
        ###### The condig dictionary is found in the utils.py ##########
        config = get_config(resolution)
        ##### config is a ductionary 
        G = BigGAN.Generator(**config)
        G.load_state_dict(
            torch.load("pretrained_weights/biggan_256_weights.pth"), strict=False
        )
        G = G.to(device)  # Remove nn.DataParallel
        G.eval() """
    
    #################### REPLACED
    #data_source = sys.argv[1]  # "imagenet" or "places".
    #target = sys.argv[2]  # Filename found in "imagenet" or "places" directory.
    #class_embedding = np.load(f"{data_source}/{target}.npy")
    class_embedding = np.load("biggan-am/places/02_rose.npy") # check if this is true after directory reorganization 
    class_embedding = torch.tensor(class_embedding)

    # Number of preference. Determines number of images generated
    z_num = 1
    repeat_class_embedding = class_embedding.repeat(z_num, 1).to(device)

    # Generate random noize 
    # dim_z_dict found in utils.py
    zs = torch.randn((z_num, dim_z_dict[resolution]), requires_grad=False).to(device)

    ##################### This is where the forward method is called ######################
    gan_images_tensor = G(zs, repeat_class_embedding)

    save_dir = "samples"
    print(f"Saving class embedding samples in {save_dir}.", flush=True)
    os.makedirs(save_dir, exist_ok=True)
    final_image_path = f"{save_dir}/places_02_rose3.jpg"
    save_image(gan_images_tensor, final_image_path, normalize=True, nrow=4)


if __name__ == "__main__":
    main()

    