import os
import argparse

import utils
from models import Generator
from models_with_resnet import GeneratorWithResNet

import torch
import numpy as np
from PIL import Image

def prepare_model(output_format, resnet_backbone, use_checkpoints='best', device='cpu'):
    if output_format == 'rgb':
        out_channels = 3
        if resnet_backbone:
            checkpoints_dir = 'checkpoints/resnet/rgb'  # Checkpoints directory for RGB with ResNet backbone
        else:
            checkpoints_dir = 'checkpoints/default/rgb'  # Checkpoints directory for RGB with default backbone
    else:
        out_channels = 2
        if resnet_backbone:
            checkpoints_dir = 'checkpoints/resnet/ab'
        else:
            checkpoints_dir = 'checkpoints/default/ab'
    
    if resnet_backbone:
        model = GeneratorWithResNet(out_channels=out_channels).to(device)
    else:
        model = Generator(out_channels=out_channels).to(device)
    
    checkpoints = torch.load(f'{checkpoints_dir}/gen_checkpoints_{use_checkpoints}.pth.tar', map_location=device, weights_only=False)
    model.load_state_dict(checkpoints['model'])
    return model


def colorize_image(model, image, output_format):
    with torch.no_grad():
        gen_output = model(image)
        gen_output = gen_output.detach()
    
    if output_format == 'ab':
        output_img = utils.postprocess_lab_to_rgb(image, gen_output)
    else:
        output_img = utils.postprocess_rgb(gen_output)
    output_img = np.squeeze(output_img, axis=0)
    return output_img


def main():
    parser = argparse.ArgumentParser(description='Colorize grayscale images using a pre-trained model')
    parser.add_argument('--images_path', type=str, required=True,
                        help='Path to the folder containing either colored or grayscale images. If the images are colored, they will be converted to grayscale before colorization')
    parser.add_argument('--output_path', type=str, default='./inference_results/colored',
                        help='Path to save the colorized images (default: ./inference_results/colored)')
    parser.add_argument('--use_checkpoints', type=str, default='best', choices=['best', 'last'],
                        help='Use the best or last checkpoints to colorize the images (default: best)')
    parser.add_argument('--output_format', type=str, default='rgb', choices=['rgb', 'ab'],
                        help='Output format of the model (default: rgb)')
    parser.add_argument('--resnet_backbone', type=bool, default=False,
                        help='Use ResNet-18 as the backbone of both Discriminator and Generator models (default: False)')
    args = parser.parse_args()
    images_path = args.images_path
    output_path = args.output_path
    use_checkpoints = args.use_checkpoints
    output_format = args.output_format
    resnet_backbone = args.resnet_backbone
    
    os.makedirs(output_path, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = prepare_model(output_format, resnet_backbone, use_checkpoints, device)
    
    for image_name in os.listdir(images_path):
        if image_name.startswith('.'):
            continue
        image_path = os.path.join(images_path, image_name)
        
        grayscale = Image.open(image_path).resize((256, 256)).convert('L')
        grayscale = np.array(grayscale).astype(np.float32)
        grayscale = grayscale / 127.5 - 1.0  # Normalize the image to [-1, 1]
        grayscale = torch.from_numpy(grayscale).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        
        output_img = colorize_image(model, grayscale, output_format)
        Image.fromarray(output_img).save(f'{output_path}/{image_name}')


if __name__ == '__main__':
    main()
