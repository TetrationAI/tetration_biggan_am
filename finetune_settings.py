from biggan_am import main2
import yaml

def finetune_parameters(settings, yaml_file_path):

    with open(yaml_file_path, 'r') as file:
            config = yaml.safe_load(file)
            
    for setting, parameters in settings.items(): 
        
        # store the setting in the embedding name 
        config['today_date'] = setting 
        for param in parameters: 
            config['today_time'] = param

            print("Setting Investigated: ", config["today_date"])
            print("Current Parameter: ", config["today_time"])

            # save yaml with configured parameters 
            with open(yaml_file_path, 'w') as file:
                yaml.safe_dump(config, file)

            # run biggan_am with new settings
            main2()

def main3(): 
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