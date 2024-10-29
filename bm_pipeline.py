import torch 
import torch
import matplotlib.pyplot as plt
import os

from generate_images_from_embedding import sample_images_save 
from generate_images_from_embedding import sample_images_save

# Set the working directory
os.chdir("/mnt/c/Users/Gästkonto/Documents/Programmering/projekt/TetrationAI")

# Check if it worked
print("Current Working Directory:", os.getcwd())

from first_demo_main.flower_set_classifier.main import main 
from biggan_am import main2 

import os
import shutil
import yaml
import time 
from datetime import datetime

"""
Directory structure: 

Tetration 
 - first_demo_main
    - flower_set_classifier
        - file 1 
        - file 2 
        - file 3 
        - file ... 
 - biggan-am
    - file 4
    - file 5 
    - file 6 
    - file ... 
 - more directories

 Now, from a file in biggan-am, ex file 4 i want to access functions in the files in flower_set_classifier, ex file 1. How can i import those functions? 
"""

"""
Structure: 
    1. Load state dicts from two mobels, on with complementary GAN data, and one without. 
        a. Find a proper classifier. #
        b. Create class embeddings for each class.  # 
        c. Create extended dataset using the class embeddings. # 
    2. Compute graphs of accuracy progression.  # 
    3. Compute SD around the "converges" values. 
    4. Compute ANOVA-test to determine similarity or differences. 
"""

################################# PIPELINE #################################
# Place 

""" 
1. Train classifier. 
2. Load classifier. 
3. Retrieve embeddings. 
4. Determine number of images to generate and where. 
5. Generate images and save in desired directory. 
"""

# 1 - Train classifier. 
def train_classifier(train_valid_data_path = 'datasets/flowers', 
                     test_data_path = None ,
                     model_name = "biggan_discriminator_own_dataset", 
                     num_original_classes = 5, 
                     discriminator_save_dir = "classifiers/"): 
    
    # run train classifier pipeline & save model and metrics in discriminator_save_dir
    main(train_data_path = train_valid_data_path, 
         test_data_path = test_data_path, 
         model_name = model_name, 
         num_original_classes = num_original_classes, 
         discriminator_save_dir = discriminator_save_dir)

    None

# 2 - Load classifier.
def load_classifier(model_name, 
                    model_path = False, 
                    num_classes = False, 
                    discriminator_save_dir = False, 
                    save_embeddings_dir = "final", 
                    yaml_file_path = 'biggan-am/opts.yaml'): 

    today_date = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H-%M-%S")

    # Load the YAML file
    with open(yaml_file_path, 'r') as file:
        config = yaml.safe_load(file)

    # Modify the values as needed
    # get the discriminator model
    config['model'] = model_name 
    # get the path to where the model is stored
    config["model_path"] = model_path
    # get the number of classes
    config["num_classes"] = num_classes
    # get the number of classes before training the model
    config["discriminator_save_dir"] = discriminator_save_dir
    # get the directory where to save embeddings
    config["final_dir"] = save_embeddings_dir
    # get date of today
    config["today_date"] = today_date
    # get current time
    config["time_str"] = time_str

    # Save the modified YAML back to the file
    with open(yaml_file_path, 'w') as file:
        yaml.safe_dump(config, file)

    print("YAML file updated successfully.")
    

# 3 - Retrieve embeddings run biggan_am. 
def retireve_embeddings_run_biggan_am(): 

    # run biggan_am 
    main2()

# 4 - Determine number of images to generate and where. 
def generate_images_from_embeddings(num_images_to_generate = 10, 
                                     save_embeddings_dir = "final", 
                                     yaml_file_path = 'biggan-am/opts.yaml', 
                                     save_dir = "samples/samplesA", 
                                     resolution=128, 
                                     batch_size=5): 

    with open(yaml_file_path, 'r') as file:
        config = yaml.safe_load(file)

    # get current date
    today_date = config["today_date"] 
    # get current time
    time_str = config["time_str"] 

    embedding_path = f"{save_embeddings_dir}/_{today_date}_{time_str}.npy"

    """
    z_num: number of images to be generated 
    embedding_path: path to file with embeddings. 
    save_dir: path to directory where images are to be saved. 
    batch_size: number of images to generate per batch
    """    

    sample_images_save(z_num = num_images_to_generate, 
                       embedding_path = embedding_path, 
                       save_dir = save_dir, 
                       resolution = resolution, 
                       batch_size = batch_size)