import json
import torch

class Config:
    def __init__(self):
        self.train_images_dir = 'dataset/train'
        self.val_images_dir = 'dataset/val'
        self.img_size = 256
        self.n_input = 1  # L channel
        self.n_output = 3  # AB channels
        self.batch_size = 8
        self.start_epoch = 0
        self.epochs = 20
        self.num_workers = 4
        self.output_format = 'rgb'
        self.resnet_backbone = False
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.generator_lr = 2e-4
        self.discriminator_lr = 2e-4
        self.betas = (0.5, 0.999)
        self.l1_lambda = 80
        
        self.save_checkpoints_freq = 10  # how many epochs to wait to save the checkpoints
        self.save_images_freq = 1  # how many epochs to wait to save the generated images
        self.save_models = True  # whether to save the checkpoints of models
        self.load_models = False  # whether to load the checkpoints of models
        self.save_examples = True  # whether to save true and generated images
        
        self.checkpoints_dir = 'checkpoints'
        self.examples_dir = 'generated_images'
    
    def save_config(self, filename):
        config_dict = self.__dict__  # convert the attributes of the class to a dictionary
        config_dict['device'] = str(config_dict['device'])
        with open(filename, 'w') as file:
            json.dump(config_dict, file, indent=4)
    
    def load_config(self, filename):
        with open(filename, 'r') as file:
            config_dict = json.load(file)
            self.__dict__.update(config_dict)  # update the attributes of the class with the loaded dictionary
