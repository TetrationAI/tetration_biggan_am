import BigGAN
import itertools
import numpy as np
import random
import time
import torch.nn.functional as F
import yaml
from datetime import datetime

from torch import optim
from torchvision.utils import save_image
from utils import *

os.chdir("/mnt/c/Users/Gästkonto/Documents/Programmering/projekt/TetrationAI")

# Check if it worked
print("Current Working Directory:", os.getcwd())

# No labels involved here since mean iss used as method of interpolation
def get_initial_embeddings(
    resolution,
    init_method,
    init_num,
    min_clamp,
    max_clamp,
    dim_z,
    G,
    net,
    target_class,
    noise_std,
    num_classes, 
):
    class_embeddings = np.load(f"biggan-am/biggan_embeddings_{resolution}.npy")
    class_embeddings = torch.from_numpy(class_embeddings)
    embedding_dim = class_embeddings.shape[-1]

    if init_method == "mean":

        mean_class_embedding = torch.mean(class_embeddings, dim=0)
        init_embeddings = mean_class_embedding.repeat(init_num, 1)
        init_embeddings += torch.randn((init_num, embedding_dim)) * 0.1

    elif init_method == "top":

        class_embeddings_clamped = torch.clamp(class_embeddings, min_clamp, max_clamp)

        num_samples = 10 # ? 
        avg_list = []
        for i in range(num_classes):
            class_embedding = class_embeddings_clamped[i]
            repeat_class_embedding = class_embedding.repeat(num_samples, 1)
            final_z = torch.randn((num_samples, dim_z), requires_grad=False)

            with torch.no_grad():
                gan_images_tensor = G(final_z, repeat_class_embedding)
                resized_images_tensor = nn.functional.interpolate(
                    gan_images_tensor, size=224
                )
                pred_logits = net(resized_images_tensor)

            pred_probs = nn.functional.softmax(pred_logits, dim=1)
            avg_target_prob = pred_probs[:, target_class].mean().item()
            avg_list.append(avg_target_prob)

        avg_array = np.array(avg_list)
        sort_index = np.argsort(avg_array)

        print(f"The top {init_num} classes: {sort_index[-init_num:]}")

        init_embeddings = class_embeddings[sort_index[-init_num:]]

    elif init_method == "random":

        index_list = random.sample(range(num_classes), init_num)
        print(f"The {init_num} random classes: {index_list}")
        init_embeddings = class_embeddings[index_list]

    elif init_method == "target": 

        init_embeddings = (
            class_embeddings[target_class].unsqueeze(0).repeat(init_num, 1)
        )
        init_embeddings += torch.randn((init_num, embedding_dim)) * noise_std

    return init_embeddings

# No labels involved here since loss is measured from the diversity within the generated images rather than comparing predictions to truth. 
def get_diversity_loss(
    half_z_num, zs, dloss_function, pred_probs, alexnet_conv5, resized_images_tensor
):
    pairs = list(itertools.combinations(range(len(zs)), 2))
    random.shuffle(pairs)

    first_idxs = []
    second_idxs = []
    for pair in pairs[:half_z_num]:
        first_idxs.append(pair[0])
        second_idxs.append(pair[1])

    denom = F.pairwise_distance(zs[first_idxs, :], zs[second_idxs, :])

    if dloss_function == "softmax":

        num = torch.sum(
            F.pairwise_distance(pred_probs[first_idxs, :], pred_probs[second_idxs, :])
        )

    elif dloss_function == "features":

        features_out = alexnet_conv5(resized_images_tensor)
        num = torch.sum(
            F.pairwise_distance(
                features_out[first_idxs, :].view(half_z_num, -1),
                features_out[second_idxs, :].view(half_z_num, -1),
            )
        )

    else:

        num = torch.sum(
            F.pairwise_distance(
                resized_images_tensor[first_idxs, :].view(half_z_num, -1),
                resized_images_tensor[second_idxs, :].view(half_z_num, -1),
            )
        )

    return num / denom


def run_biggan_am(
    init_embedding,
    device,
    lr,
    dr,
    state_z,
    n_iters,
    z_num,
    dim_z,
    steps_per_z,
    min_clamp,
    max_clamp,
    G,
    net,
    criterion,
    labels,
    dloss_function,
    half_z_num,
    alexnet_conv5,
    alpha,
    target_class,
    init_embedding_idx,
    intermediate_dir,
    use_noise_layer,
):
    optim_embedding = init_embedding.unsqueeze(0).to(device)
    optim_embedding.requires_grad_()
    optim_comps = {
        "optim_embedding": optim_embedding,
        "use_noise_layer": use_noise_layer,
    }
    optim_params = [optim_embedding]
    
    if use_noise_layer:
        noise_layer = nn.Linear(dim_z, dim_z).to(device)
        noise_layer.train()
        optim_params += [params for params in noise_layer.parameters()]
        optim_comps["noise_layer"] = noise_layer

    optimizer = optim.Adam(optim_params, lr=lr, weight_decay=dr)

    torch.set_rng_state(state_z)

    for epoch in range(n_iters):

        zs = torch.randn((z_num, dim_z), requires_grad=False).to(device)

        for z_step in range(steps_per_z):

            optimizer.zero_grad()

            if use_noise_layer:
                z_hats = noise_layer(zs)
            else:
                z_hats = zs
            
            #### "Random image tensor which is to be optimized" ####
            clamped_embedding = torch.clamp(optim_embedding, min_clamp, max_clamp)
            
            repeat_clamped_embedding = clamped_embedding.repeat(z_num, 1).to(device)
            
            # z_hats = random noice
            gan_images_tensor = G(z_hats, repeat_clamped_embedding)
            
            resized_images_tensor = nn.functional.interpolate(
                gan_images_tensor, size=128 #224 (this is the input size for imagenet Classifiers) When set to 128 it works! 
            )
            #print("resized shape", resized_images_tensor.shape)
            pred_logits = net(resized_images_tensor)
            # labels (from target-class) used to predict loss 
            #print("labels shape", labels.shape)
            #print("logits shape", pred_logits.shape)
            #pred_logits = pred_logits.view(pred_logits.size(0), pred_logits.size(1), -1).mean(dim=2)  # size is now [batch_size, num_classes] instead of [batch_size, image_size, num_classes, num_classes]
            #print("labels shape after:", labels.shape)
            #print("logits shape after:", pred_logits.shape)

            loss = criterion(pred_logits, labels)
         
            pred_probs = nn.functional.softmax(pred_logits, dim=1)

            if dloss_function:
                diversity_loss = get_diversity_loss(
                    half_z_num,
                    zs,
                    dloss_function,
                    pred_probs,
                    #### replace with our network, here, just the features, not the "NET". Kinda wrong, just used for feature loss. 
                    alexnet_conv5,
                    resized_images_tensor,
                )
                
                loss += -alpha * diversity_loss

            loss.backward()
            optimizer.step()

            avg_target_prob = pred_probs[:, target_class].mean().item()
            log_line = f"Embedding: {init_embedding_idx}\t"
            log_line += f"Epoch: {epoch:0=5d}\tStep: {z_step:0=5d}\t"
            log_line += f"Average Target Probability:{avg_target_prob:.4f}"
            print(log_line)

            if intermediate_dir:
                global_step_id = epoch * steps_per_z + z_step
                img_f = f"{init_embedding_idx}_{global_step_id:0=7d}.jpg"
                output_image_path = f"{intermediate_dir}/{img_f}"
                save_image(
                    gan_images_tensor, output_image_path, normalize=True, nrow=10
                )

            torch.cuda.empty_cache()

    return optim_comps

# No labels involved here
def save_final_samples(
    optim_comps,
    min_clamp,
    max_clamp,
    device,
    model,
    state_z,
    num_final,
    dim_z,
    G,
    repeat_original_embedding,
    final_dir,
    init_embedding_idx,
):
    optim_embedding = optim_comps["optim_embedding"]
    optim_embedding_clamped = torch.clamp(optim_embedding, min_clamp, max_clamp)
    repeat_optim_embedding = optim_embedding_clamped.repeat(4, 1).to(device)

    if optim_comps["use_noise_layer"]:
        optim_comps["noise_layer"].eval()

    optim_imgs = []
    if model not in {"mit_alexnet", "mit_resnet18"}:
        original_imgs = []

    torch.set_rng_state(state_z)

    for show_id in range(num_final):
        zs = torch.randn((num_final, dim_z), device=device, requires_grad=False)
        if optim_comps["use_noise_layer"]:
            with torch.no_grad():
                z_hats = optim_comps["noise_layer"](zs)

        else:
            z_hats = zs

        with torch.no_grad():
            optim_imgs.append(G(z_hats, repeat_optim_embedding))
            if model not in {"mit_alexnet", "mit_resnet18"}:
                original_imgs.append(G(z_hats, repeat_original_embedding))

    # get time and date if specified in the opts yaml file, else just get the current.  
    if optim_comps.get("today_date") and optim_comps.get("today_time"): 
        today_date = optim_comps["today_date"]
        today_time = optim_comps["today_time"]
    
    else: 
        today_date = datetime.now().strftime("%Y-%m-%d")
        today_time = datetime.now().strftime("%H-%M-%S")

    final_image_path = f"{final_dir}/{init_embedding_idx}_{today_date}_{today_time}.jpg"

    optim_imgs = torch.cat(optim_imgs, dim=0)
    save_image(optim_imgs, final_image_path, normalize=True, nrow=4)
    np.save(
       f"{final_dir}/_{today_date}_{today_time}.npy",
        optim_embedding_clamped.detach().cpu().numpy(),
    )
    if optim_comps["use_noise_layer"]:
        torch.save(
            optim_comps["noise_layer"].state_dict(),
            f"{final_dir}/{init_embedding_idx}_noise_layer.pth",
        )

    if model not in {"mit_alexnet", "mit_resnet18"}:
        original_image_path = f"{final_dir}/{init_embedding_idx}_{today_date}_{today_time}_original.jpg"
        original_imgs = torch.cat(original_imgs, dim=0)
        save_image(original_imgs, original_image_path, normalize=True, nrow=4)


def main2():
    opts = yaml.safe_load(open("biggan-am/opts.yaml"))

    # Set random seed.
    seed_z = opts["seed_z"]
    torch.manual_seed(seed_z)
    torch.cuda.manual_seed(seed_z)
    np.random.seed(seed_z)
    random.seed(seed_z)

    init_method = opts["init_method"]
    print(f"Initialization method: {init_method}")
    if init_method == "target":
        noise_std = opts["noise_std"]
        print(f"The noise std is: {noise_std}")
    else:
        noise_std = None

    # Load the models.
    # Set up cudnn.benchmark for free speed.
    torch.backends.cudnn.benchmark = True
    device = "cuda:0"

    print("Loading the BigGAN generator model...")
    resolution = opts["resolution"]
    config = get_config(resolution)
    start_time = time.time()
    G = BigGAN.Generator(**config)
    if resolution == 128:
        biggan_weights = "biggan-am/pretrained_weights/138k/G_ema.pth"
    else:
        biggan_weights = "biggan-am/pretrained_weights/biggan_256_weights.pth"

    G.load_state_dict(torch.load(f"{biggan_weights}"), strict=False)
    G = nn.DataParallel(G).to(device)
    G.eval()

    model_name = opts["model"]
    
    #### Additional for pipeline ####

    model_path = opts["model_path"], 
    
    num_classes2 = opts["num_classes"], 
    
    discriminator_save_dir = opts["discriminator_save_dir"]
    
    #################################

    net = nn.DataParallel(load_net(model_name= model_name, 
                                   model_path = model_path, 
                                   num_classes = num_classes2,
                                   discriminator_save_dir = discriminator_save_dir)).to(device)
    
    net.eval()

    z_num = opts["z_num"]
    dloss_function = opts["dloss_function"]
    if dloss_function:
        half_z_num = z_num // 2
        print(f"Using diversity loss: {dloss_function}")
        if dloss_function == "features":
            if model_name != "alexnet":
                alexnet_conv5 = nn.DataParallel(load_net("alexnet_conv5")).to(device)
                alexnet_conv5.eval()

            else:
                # Here, the features i.e feature maps from convolutional network, are extracted
                alexnet_conv5 = net.features

    else:
        half_z_num = alexnet_conv5 = None

    print(f"BigGAN initialization time: {time.time() - start_time}")

    # Set up optimization.
    init_num = opts["init_num"]
    dim_z = dim_z_dict[resolution]
    max_clamp = max_clamp_dict[resolution]
    min_clamp = min_clamp_dict[resolution]

    target_class = opts["target_class"]
    num_classes = opts["num_classes"]
    
    init_embeddings = get_initial_embeddings(
        resolution,
        init_method,
        init_num,
        min_clamp,
        max_clamp,
        dim_z,
        G,
        net,
        target_class,
        noise_std,
        num_classes, 
    )

    criterion = nn.CrossEntropyLoss()

    #OBS! target_class must match the output of the classifier! 
    labels = torch.LongTensor([target_class] * z_num).to(device)
    
    print("LABELS IN MAIN")
    print(labels)

    state_z = torch.get_rng_state()

    intermediate_dir = opts["intermediate_dir"]
    if intermediate_dir:
        print(f"Saving intermediate samples in {intermediate_dir}.")
        os.makedirs(intermediate_dir, exist_ok=True)

    final_dir = opts["final_dir"]
    if final_dir:
        print(f"Saving final samples in {final_dir}.")
        os.makedirs(final_dir, exist_ok=True)
        if model_name not in {"mit_alexnet", "mit_resnet18"}:
            original_embeddings = np.load(f"biggan-am/biggan_embeddings_{resolution}.npy")
            original_embeddings = torch.from_numpy(original_embeddings)
            original_embedding_clamped = torch.clamp(
                original_embeddings[target_class].unsqueeze(0), min_clamp, max_clamp
            )
            num_final = 4
            repeat_original_embedding = original_embedding_clamped.repeat(
                num_final, 1
            ).to(device)

        else:
            num_final = None
            repeat_original_embedding = None

    for (init_embedding_idx, init_embedding) in enumerate(init_embeddings):
        init_embedding_idx = str(init_embedding_idx).zfill(2)
        optim_comps = run_biggan_am(
            init_embedding,
            device,
            opts["lr"],
            opts["dr"],
            state_z,
            opts["n_iters"],
            z_num,
            dim_z,
            opts["steps_per_z"],
            min_clamp,
            max_clamp,
            G,
            net,
            criterion,
            labels,
            dloss_function,
            half_z_num,
            alexnet_conv5,
            opts["alpha"],
            target_class,
            init_embedding_idx,
            intermediate_dir,
            opts["use_noise_layer"],
        )
        
        if final_dir:
            save_final_samples(
                optim_comps,
                min_clamp,
                max_clamp,
                device,
                model_name,
                state_z,
                num_final,
                dim_z,
                G,
                repeat_original_embedding,
                final_dir,
                init_embedding_idx,
            )


if __name__ == "__main__":
    main2()
