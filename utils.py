import os
import warnings
from tqdm import tqdm

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import lab2rgb, rgb2lab

import torch
import torch.nn as nn
from config import Config

CONFIG = Config()


def _init_weights(model, initialization='normal', gain=0.02):
    """Initialize the weights of the model using the specified initialization method. More information on initialization methods here https://pytorch.org/docs/stable/nn.init.html

    Args:
        model (nn.Module): The model to initialize weights for
        initialization (str, optional): The initialization method to use. Defaults to 'normal'. Can be 'normal' or 'xavier'
        gain (float, optional): The gain value for the normal distribution. Defaults to 0.02
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if initialization == 'normal':
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif initialization == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(f'Initialization method {initialization} not implemented')
            
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)
    
    model.apply(init_func)
    return model


def init_model(model, initialization='normal', device='cpu'):
    model = _init_weights(model, initialization=initialization)
    model = model.to(device)
    return model


class AverageMeter:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.count, self.avg, self.sum = [0.] * 3
    
    def update(self, val, count=1):
        self.count += count
        self.sum += count * val
        self.avg = self.sum / self.count


def create_loss_meters():
    loss_D_fake = AverageMeter()
    loss_D_real = AverageMeter()
    loss_D = AverageMeter()
    loss_G_GAN = AverageMeter()
    loss_G_L1 = AverageMeter()
    loss_G = AverageMeter()
    
    return {
        'loss_D_fake': loss_D_fake,
        'loss_D_real': loss_D_real,
        'loss_D': loss_D,
        'loss_G_GAN': loss_G_GAN,
        'loss_G_L1': loss_G_L1,
        'loss_G': loss_G
    }


def postprocess_lab_to_rgb(L, AB):
    L = (L + 1.) * 50.  # denormalize L channel
    AB = AB * 128.  # denormalize AB channels
    lab = torch.cat([L, AB], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rgb_imgs = [lab2rgb(img) for img in lab]
    
    rgb_imgs = [np.clip(img * 255, 0, 255).astype('uint8') for img in rgb_imgs]
    return np.stack(rgb_imgs, axis=0)


def postprocess_rgb(images):
    images = images.permute(0, 2, 3, 1)
    images = (images + 1.0) * 127.5
    images = images.clamp(0, 255)
    images = images.cpu().numpy()
    images = images.round().astype('uint8')
    return images


def save_checkpoints(model, optimizer, folder, filename):
    os.makedirs(folder, exist_ok=True)
    
    print('=> Saving checkpoints')
    checkpoints = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoints, os.path.join(folder, filename))


def load_checkpoints(model, optimizer, folder, checkpoints_file, lr):
    print('=> Loading checkpoints')
    checkpoints = torch.load(os.path.join(folder, checkpoints_file), map_location=CONFIG.device, weights_only=False)
    
    model.load_state_dict(checkpoints['model'])
    optimizer.load_state_dict(checkpoints['optimizer'])
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_examples(net_G, val_loader, epoch, folder):
    os.makedirs(folder, exist_ok=True)
    
    data = next(iter(val_loader))
    gray, colored = data['gray'], data['colored']
    gray, colored = gray.to(CONFIG.device), colored.to(CONFIG.device)
    
    net_G.eval()
    with torch.no_grad():
        fake_color = net_G(gray)
        fake_color = fake_color.detach()
    
    if CONFIG.output_format == 'ab':
        fake_imgs = postprocess_lab_to_rgb(gray, fake_color)
        real_imgs = postprocess_lab_to_rgb(gray, colored)
        gray = ((gray + 1) * 50.).cpu().squeeze().numpy()  # denormalize L channel
    else:
        fake_imgs = postprocess_rgb(fake_color)
        real_imgs = postprocess_rgb(colored)
        gray = postprocess_rgb(gray)  # denormalize grayscale image
    
    nrows=4
    fig, ax = plt.subplots(nrows, 3, figsize=(12, 12))
    for i in range(nrows):
        ax[i, 0].imshow(gray[i], cmap='gray')
        ax[i, 0].axis('off')
        
        ax[i, 1].imshow(real_imgs[i])
        ax[i, 1].axis('off')
        
        ax[i, 2].imshow(fake_imgs[i])
        ax[i, 2].axis('off')
    
    ax[0, 0].set_title('L channel')
    ax[0, 1].set_title('Real')
    ax[0, 2].set_title('Generated')
    # Adjust the layout and save
    plt.subplots_adjust(wspace=0.05, hspace=0.05)  # Minimal spacing
    plt.savefig(os.path.join(folder, f'pred_batch_{epoch}.png'), bbox_inches='tight', pad_inches=0)
    plt.close()


def log_training(metrics, epoch):
    '''Function to log the training metrics to the .csv file

    Args:
        metrics (dict[str, AverageMeter]): dictionary containing the training metrics
        epoch (int): current epoch
    '''
    if epoch == CONFIG.start_epoch + 1:
        with open('training_logs.csv', 'w') as f:
            f.write('epoch,loss_D_fake,loss_D_real,loss_D,loss_G_GAN,loss_G_L1,loss_G\n')
    
    with open('training_logs.csv', 'a') as f:
        f.write(f"{epoch},{metrics['loss_D_fake'].avg},{metrics['loss_D_real'].avg},{metrics['loss_D'].avg},{metrics['loss_G_GAN'].avg},{metrics['loss_G_L1'].avg},{metrics['loss_G'].avg}\n")
