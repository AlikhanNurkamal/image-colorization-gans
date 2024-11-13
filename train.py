import os
import glob
import cv2
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import albumentations as A

import utils
from config import Config
from custom_dataset import CustomDataset
from models import Generator, Discriminator, GANLoss
from models_with_resnet import GeneratorWithResNet, DiscriminatorWithResNet

CONFIG = Config()

# Define transformations
train_transforms = A.Compose([
    A.Resize(CONFIG.img_size, CONFIG.img_size, interpolation=cv2.INTER_LINEAR),
    A.HorizontalFlip(p=0.3),
    A.VerticalFlip(p=0.3),
])
val_transforms = A.Compose([
    A.Resize(CONFIG.img_size, CONFIG.img_size, interpolation=cv2.INTER_LINEAR),
])

def create_dataloaders():
    train_ds = CustomDataset(CONFIG.train_images_dir, transform=train_transforms)
    val_ds = CustomDataset(CONFIG.val_images_dir, split='val', transform=val_transforms)
    
    train_dl = DataLoader(train_ds, batch_size=CONFIG.batch_size, shuffle=True,
                          pin_memory=True, num_workers=CONFIG.num_workers)
    val_dl = DataLoader(val_ds, batch_size=CONFIG.batch_size, shuffle=False,
                        pin_memory=True, num_workers=CONFIG.num_workers)
    return train_dl, val_dl


def create_models():
    if CONFIG.resnet_backbone:
        net_G = GeneratorWithResNet(out_channels=CONFIG.n_output, img_size=CONFIG.img_size).to(CONFIG.device)
        net_D = DiscriminatorWithResNet().to(CONFIG.device)
    else:
        net_G = Generator(in_channels=CONFIG.n_input, out_channels=CONFIG.n_output).to(CONFIG.device)
        net_D = Discriminator().to(CONFIG.device)
    
    criterion_GAN = GANLoss().to(CONFIG.device)
    criterion_L1 = torch.nn.L1Loss().to(CONFIG.device)
    
    optimizer_G = torch.optim.Adam(net_G.parameters(), lr=CONFIG.generator_lr, betas=CONFIG.betas)
    optimizer_D = torch.optim.Adam(net_D.parameters(), lr=CONFIG.discriminator_lr, betas=CONFIG.betas)
    
    return net_G, net_D, criterion_GAN, criterion_L1, optimizer_G, optimizer_D


def train_rgb(dataloader, net_G, net_D, criterion_GAN, criterion_L1,
              optim_G, optim_D, scaler_G, scaler_D, epoch) -> dict:
    net_G.train()
    net_D.train()
    
    losses = utils.create_loss_meters()  # create the loss meters for the Discriminator and Generator
    loop = tqdm(enumerate(dataloader))
    for idx, data in loop:
        gray, colored = data['gray'].to(CONFIG.device), data['colored'].to(CONFIG.device)
        
        # Train the Discriminator
        for param in net_D.parameters():
            param.requires_grad = True
        
        optim_D.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            fake_colored = net_G(gray)
            fake_preds = net_D(gray, fake_colored.detach())
            real_preds = net_D(gray, colored)
            
            loss_D_fake = criterion_GAN(fake_preds, False)
            loss_D_real = criterion_GAN(real_preds, True)
            loss_D = (loss_D_fake + loss_D_real) / 2
        
        scaler_D.scale(loss_D).backward()
        scaler_D.step(optim_D)
        scaler_D.update()
        losses['loss_D_fake'].update(loss_D_fake.item(), gray.shape[0])
        losses['loss_D_real'].update(loss_D_real.item(), gray.shape[0])
        losses['loss_D'].update(loss_D.item(), gray.shape[0])
        
        # Train the Generator
        for param in net_D.parameters():
            param.requires_grad = False
        
        optim_G.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            fake_preds = net_D(gray, fake_colored)
            
            loss_G_GAN = criterion_GAN(fake_preds, True)
            loss_G_L1 = criterion_L1(fake_colored, colored)
            loss_G = loss_G_GAN + CONFIG.l1_lambda * loss_G_L1
            
        scaler_G.scale(loss_G).backward()
        scaler_G.step(optim_G)
        scaler_G.update()
        losses['loss_G_GAN'].update(loss_G_GAN.item(), gray.shape[0])
        losses['loss_G_L1'].update(loss_G_L1.item(), gray.shape[0])
        losses['loss_G'].update(loss_G.item(), gray.shape[0])
        
        loop.set_description(f'Epoch [{epoch}/{CONFIG.epochs}]')
        loop.set_postfix(loss_D=loss_D.item(), loss_G=loss_G.item())
    return losses


def train_lab(dataloader, net_G, net_D, criterion_GAN, criterion_L1,
          optim_G, optim_D, scaler_G, scaler_D, epoch) -> dict:
    net_G.train()
    net_D.train()
    
    losses = utils.create_loss_meters()  # create the loss meters for the Discriminator and Generator
    loop = tqdm(enumerate(dataloader))
    for idx, data in loop:
        L, AB = data['gray'].to(CONFIG.device), data['colored'].to(CONFIG.device)
        # L_3_channels = torch.cat([L, L, L], dim=1)
        
        # Train the Discriminator
        for param in net_D.parameters():
            param.requires_grad = True
        
        optim_D.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            fake_AB = net_G(L)
            fake_image = torch.cat([L, fake_AB], dim=1)
            fake_preds = net_D(L, fake_image.detach())  # detach the fake image to avoid computing gradients for the Generator
            real_image = torch.cat([L, AB], dim=1)
            real_preds = net_D(L, real_image)
            
            loss_D_fake = criterion_GAN(fake_preds, False)
            loss_D_real = criterion_GAN(real_preds, True)
            loss_D = (loss_D_fake + loss_D_real) / 2
        
        scaler_D.scale(loss_D).backward()
        scaler_D.step(optim_D)
        scaler_D.update()
        losses['loss_D_fake'].update(loss_D_fake.item(), L.shape[0])
        losses['loss_D_real'].update(loss_D_real.item(), L.shape[0])
        losses['loss_D'].update(loss_D.item(), L.shape[0])
        
        # Train the Generator
        for param in net_D.parameters():
            param.requires_grad = False  # freeze the Discriminator parameters
        
        optim_G.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            fake_image = torch.cat([L, fake_AB], dim=1)
            fake_preds = net_D(L, fake_image)
            
            loss_G_GAN = criterion_GAN(fake_preds, True)
            loss_G_L1 = criterion_L1(fake_AB, AB)  # L1 loss between the generated AB channels and the true AB channels
            loss_G = loss_G_GAN + CONFIG.l1_lambda * loss_G_L1
        
        scaler_G.scale(loss_G).backward()
        scaler_G.step(optim_G)
        scaler_G.update()
        losses['loss_G_GAN'].update(loss_G_GAN.item(), L.shape[0])
        losses['loss_G_L1'].update(loss_G_L1.item(), L.shape[0])
        losses['loss_G'].update(loss_G.item(), L.shape[0])
        
        loop.set_description(f'Epoch [{epoch}/{CONFIG.epochs}]')
        loop.set_postfix(loss_D=loss_D.item(), loss_G=loss_G.item())
    return losses


def main():
    parser = argparse.ArgumentParser(description='Train a GAN model for colorizing grayscale images')
    parser.add_argument('--output_format', type=str, default='rgb', choices=['rgb', 'ab'],
                        help='Output format of the model (default: rgb)')
    parser.add_argument('--resnet_backbone', type=bool, default=False,
                        help='Use ResNet-18 as the backbone of both Discriminator and Generator models (default: False)')
    args = parser.parse_args()
    output_format = args.output_format
    resnet_backbone = args.resnet_backbone
    
    if resnet_backbone:
        CONFIG.resnet_backbone = True
        CONFIG.batch_size = 4
    
    if output_format == 'rgb':
        CONFIG.output_format = 'rgb'
        CONFIG.n_output = 3
        train = train_rgb
    elif output_format == 'ab':
        CONFIG.output_format = 'ab'
        CONFIG.n_output = 2
        train = train_lab
    else:
        raise ValueError(f'Output format must be either "rgb" or "ab", got {output_format}')
    
    # Save the model and training configurations to a JSON file
    CONFIG.save_config(filename='config.json')
    print('Run configurations are saved!')
    
    train_dl, val_dl = create_dataloaders()
    net_G, net_D, criterion_GAN, criterion_L1, optimizer_G, optimizer_D = create_models()
    scaler_G = torch.amp.GradScaler('cuda')
    scaler_D = torch.amp.GradScaler('cuda')
    
    # Load the models if the flag is set to True
    if CONFIG.load_models:
        utils.load_checkpoints(net_G, optimizer_G, 'checkpoints_dir...', f'gen_checkpoints_{CONFIG.start_epoch}.pth.tar', CONFIG.generator_lr)
        utils.load_checkpoints(net_D, optimizer_D, 'checkpoints_dir...', f'disc_checkpoints_{CONFIG.start_epoch}.pth.tar', CONFIG.discriminator_lr)
    
    torch.cuda.empty_cache()  # free up some memory before training the GAN
    print('Starting training GAN...')
    for epoch in range(CONFIG.start_epoch + 1, CONFIG.start_epoch + CONFIG.epochs + 1):
        losses = train(train_dl, net_G, net_D, criterion_GAN, criterion_L1,
                       optimizer_G, optimizer_D, scaler_G, scaler_D, epoch)
        utils.log_training(losses, epoch)
        for _, meter in losses.items():
            meter.reset()  # reset the loss meters for the next epoch
        
        if epoch % CONFIG.save_checkpoints_freq == 0:
            if CONFIG.save_models:
                utils.save_checkpoints(net_G, optimizer_G, CONFIG.checkpoints_dir, f'gen_checkpoints_{epoch}.pth.tar')
                utils.save_checkpoints(net_D, optimizer_D, CONFIG.checkpoints_dir, f'disc_checkpoints_{epoch}.pth.tar')
        if epoch % CONFIG.save_images_freq == 0:
            if CONFIG.save_examples:
                utils.save_examples(net_G, val_dl, epoch, CONFIG.examples_dir)
    print('Training is complete!')


if __name__ == '__main__':
    main()
