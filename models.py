import torch
import torch.nn as nn

# --------------------
# Discriminator
# --------------------
class CNNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride,
                      padding=1, bias=False, padding_mode='reflect'),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.2),
        )
    
    def forward(self, x):
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels: int=3, features: list=[64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(1 + in_channels, features[0], kernel_size=4,
                      stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(negative_slope=0.2),
        )
        
        layers = []
        in_channels = features[0]
        
        for feature in features[1:]:
            layers.append(
                CNNBlock(in_channels, feature, stride=1 if feature == features[-1] else 2)
            )
            in_channels = feature
        
        layers.append(
            nn.Conv2d(in_channels, 1, kernel_size=4, stride=1,
                      padding=1, padding_mode='reflect')
        )
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x, y):
        """Forward pass of the Discriminator

        Args:
            x (torch.Tensor): Batch of grayscale images
            y (torch.Tensor): Batch of colorized images

        Returns:
            _type_: _description_
        """
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        return self.model(x)

# --------------------
# Generator
# --------------------
class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, down: bool=True, act: str="relu", use_dropout: bool=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4,
                      stride=2, padding=1, bias=False, padding_mode="reflect")
            if down else
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4,
                               stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(negative_slope=0.2),
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x

class Generator(nn.Module):
    def __init__(self, in_channels: int=1, out_channels: int=3, features: int=64):
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=4,
                      stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(negative_slope=0.2),
        )
        
        self.down1 = Block(features, features * 2, act="leaky")
        self.down2 = Block(features * 2, features * 4, act="leaky")
        self.down3 = Block(features * 4, features * 8, act="leaky")
        self.down4 = Block(features * 8, features * 8, act="leaky")
        self.down5 = Block(features * 8, features * 8, act="leaky")
        self.down6 = Block(features * 8, features * 8, act="leaky")
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, kernel_size=4,
                      stride=2, padding=1),
            nn.ReLU(),
        )
        self.up1 = Block(features * 8, features * 8, down=False, use_dropout=True)
        self.up2 = Block(features * 8 * 2, features * 8, down=False, use_dropout=True)
        self.up3 = Block(features * 8 * 2, features * 8, down=False, use_dropout=True)
        self.up4 = Block(features * 8 * 2, features * 8, down=False)
        self.up5 = Block(features * 8 * 2, features * 4, down=False)
        self.up6 = Block(features * 4 * 2, features * 2, down=False)
        self.up7 = Block(features * 2 * 2, features, down=False)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottleneck = self.bottleneck(d7)
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d7], dim=1))
        up3 = self.up3(torch.cat([up2, d6], dim=1))
        up4 = self.up4(torch.cat([up3, d5], dim=1))
        up5 = self.up5(torch.cat([up4, d4], dim=1))
        up6 = self.up6(torch.cat([up5, d3], dim=1))
        up7 = self.up7(torch.cat([up6, d2], dim=1))
        return self.final_up(torch.cat([up7, d1], dim=1))

# --------------------
# GAN Loss
# --------------------
class GANLoss(nn.Module):
    def __init__(self, real_label=1.0, fake_label=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))
        self.loss = nn.BCEWithLogitsLoss()
    
    def get_labels(self, preds, target_is_real):
        if target_is_real:
            labels = self.real_label
        else:
            labels = self.fake_label
        return labels.expand_as(preds)
    
    def __call__(self, preds, target_is_real):
        labels = self.get_labels(preds, target_is_real)
        loss = self.loss(preds, labels)
        return loss
