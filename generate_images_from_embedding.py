import BigGAN
import numpy as np
import sys
from datetime import datetime
import torch

from torchvision.utils import save_image
from utils import *

os.chdir("/mnt/c/Users/Gästkonto/Documents/Programmering/projekt/TetrationAI")

# Check if it worked
print("Current Working Directory:", os.getcwd())
# Import other necessary libraries like BigGAN, get_config, etc.

def sample_images_save(z_num, embedding_path, save_dir, resolution=128, batch_size=10):
    """
    z_num: number of images to be generated 
    embedding_path: path to file with embeddings. 
    save_dir: path to directory where images are to be saved. 
    batch_size: number of images to generate per batch
    """    
    # Set up cudnn.benchmark for free speed.
    torch.backends.cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device", device)

    print("Loading the BigGAN generator model...", flush=True)
    config = get_config(resolution)
    
    G = BigGAN.Generator(**config)
    if resolution == 128:
        G.load_state_dict(
            torch.load("biggan-am/pretrained_weights/138k/G_ema.pth"), strict=False
        )

    G = torch.nn.DataParallel(G).to(device)
    G.eval()
    
    # Load embedding
    class_embedding = np.load(embedding_path)
    class_embedding = torch.tensor(class_embedding).to(device)

    # Calculate the number of batches
    num_batches = (z_num + batch_size - 1) // batch_size

    os.makedirs(save_dir, exist_ok=True)
    today_date = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H-%M-%S")

    for batch_idx in range(num_batches):
        # Calculate batch start and end indices
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, z_num)

        # Generate zs for the current batch
        zs = torch.randn((end_idx - start_idx, dim_z_dict[resolution]), requires_grad=False).to(device)
        repeat_class_embedding = class_embedding.repeat(end_idx - start_idx, 1)

        # Generate images for the current batch
        gan_images_tensor = G(zs, repeat_class_embedding)

        # save images for the current batch
        for i in range(gan_images_tensor.size(0)):
            single_image_tensor = gan_images_tensor[i].unsqueeze(0)
            final_image_path = f"{save_dir}/image_{start_idx + i:03d}_{today_date}_{time_str}.jpg"
            save_image(single_image_tensor, final_image_path, normalize=True)

    print(f"Saved {z_num} class embedding samples in {save_dir}.")

# Example usage
sample_images_save(10, embedding_path="final/optimal_params.npy", save_dir="samples", batch_size=5)
#"C:\Users\Johan\Documents\Tetration\final\00_2024-04-14_17-58-20.npy"