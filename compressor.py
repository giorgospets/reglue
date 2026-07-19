import torch.nn as nn
import torch

class ResidualBlock(nn.Module):
    """
    A simple residual block with two convolutional layers and a skip connection.
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Skip connection
        return self.relu(out)


class Encoder(nn.Module):
    """
    Build an encoder from a build list of channel sizes.
   
    """
    def __init__(self, channels):
        super().__init__()
        layers = []
        for i in range(len(channels)-1):
            in_c = channels[i]
            out_c = channels[i+1]
            layers.append(nn.Conv2d(in_c, out_c, kernel_size=3, padding=1))
            if i < len(channels)-2:
                layers.append(ResidualBlock(out_c))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    """
    Build a decoder from a list of channel sizes.
    Example:
    """
    def __init__(self, channels):
        super().__init__()
        channels = channels[::-1] # Reverse the channel list 
        layers = []
        for i in range(len(channels)-1):
            in_c = channels[i]
            out_c = channels[i+1]
            layers.append(nn.Conv2d(in_c, out_c, kernel_size=3, padding=1))
            if i < len(channels)-2: 
                layers.append(ResidualBlock(out_c))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)



class EncoderDecoder(nn.Module):
    """
    The encoder-decoder model with residual blocks.
    """
    def __init__(self, build_channel_list):
        super(EncoderDecoder, self).__init__()
        self.build_channel_list = build_channel_list
        self.encoder = Encoder(channels = self.build_channel_list)
        self.decoder = Decoder(channels = self.build_channel_list)
        
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent
    
    def encode(self, x):
        latent = self.encoder(x)
        return latent
    
    def decode(self, x):
        reconstructed = self.decoder(x)
        return reconstructed
    