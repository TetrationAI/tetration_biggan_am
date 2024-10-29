from biggan_am import main2
import yaml

import yaml
import shutil

def finetune_parameters(settings, yaml_file_path):
    # keep the original for reference
    original_config_path = 'biggan-am/opts_original.yaml'

    for setting, parameters in settings.items(): 
        # copy original, this is so that changes does not "accumulate" 
        shutil.copyfile(original_config_path, yaml_file_path)

        # open second yaml, which is to be modified
        with open(yaml_file_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # update config with new parameters
        config['today_date'] = setting 
        for param in parameters: 
            # change the parameter
            config[setting] = param
            # set today time to the parameter for naming purposes
            config['today_time'] = param

            print("Setting Investigated: ", config["today_date"])
            print("Current Parameter: ", config["today_time"])

            # save yaml with updated parameters
            with open(yaml_file_path, 'w') as file:
                yaml.safe_dump(config, file)

            # run biggan-am with updated parameters
            main2()

def main3(): 
    # parameters to test 
    settings = {
    "target_class": [i for i in range(1,10)], # target classes
    "n_iters": [10,15,20,25,30,40,50], # epochs
    "z_num": [5,7,10,15,20,50], # batch sizes
    "steps_per_z":[10,15,20,25,30,40,50], # steps per epoch
    "lr":[0.01,0.05,0.1,0.15,0.2,0.5], # learning rate
    "noise_std": [0.1,0.15,0.2,0.5] # noise standard deviation
    }

    yaml_file_path = 'biggan-am/opts.yaml'
    
    finetune_parameters(settings = settings, 
                        yaml_file_path = yaml_file_path)
    
if __name__ == "__main__": 
     main3()

# OBS! Needs to return to old settings! 
# Names are not saved correctly! (i.e dates are still displayed, instead of parameters)