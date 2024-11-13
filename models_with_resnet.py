import torch
import torch.nn as nn

from torchvision.models.resnet import resnet18
from fastai.vision.models.unet import DynamicUnet

# --------------------
# Discriminator
# --------------------
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride,
                      padding=1, bias=False, padding_mode='reflect'),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.2),
        )
    
    def forward(self, x):
        return self.conv(x)

class DiscriminatorWithResNet(nn.Module):
    def __init__(self, in_channels: int=3, features: list=[64, 128, 256, 512]):
        super().__init__()
        backbone = resnet18(weights='DEFAULT')
        # Modify the first layer to accept the desired input channels
        backbone.conv1 = nn.Conv2d(
            in_channels + 1, features[0], kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        
        # Use the resnet18 as the feature extractor, excluding last 3 layers
        self.backbone = nn.Sequential(*list(backbone.children())[:-3])
        backbone_out_channels = 256  # Last output channels from resnet18’s selected layers
        
        self.additional_layers = nn.Sequential(
            CNNBlock(backbone_out_channels, backbone_out_channels * 2, stride=1),
            CNNBlock(backbone_out_channels * 2, backbone_out_channels * 2, stride=1),
        )
        
        # Add a final convolutional layer to get output size (BATCH, 1, 18, 18)
        self.final_layer = nn.Conv2d(backbone_out_channels * 2, 1, kernel_size=3, stride=1, padding=3, padding_mode='reflect')

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.backbone(x)
        x = self.additional_layers(x)
        return self.final_layer(x)

# --------------------
# Generator
# --------------------
class GeneratorWithResNet(nn.Module):
    def __init__(self, in_channels: int=1, out_channels: int=3, img_size: int=256):
        super().__init__()
        backbone = resnet18(weights='DEFAULT')
        backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        body = nn.Sequential(*list(backbone.children())[:-2])
        self.model = DynamicUnet(body, out_channels, (img_size, img_size))
        
    def forward(self, x):
        return self.model(x)
