import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False), # in_channels, out_channels, kernel_size, stride, padding
            nn.BatchNorm2d(mid_channels), # num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.model(x)

class Down(nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.MaxPool2d(kernel_size),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.model(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2) # upsample via ConvTranspose2D
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2): # x1 is input which will become convTranspose result, x2 is from skip connection
        x1 = self.model(x1)
        height_diff = x2.shape[2] - x1.shape[2]
        width_diff = x2.shape[3] - x1.shape[3]
        F.pad(x1,[
            width_diff // 2, width_diff - width_diff // 2,
            height_diff // 2, height_diff - height_diff // 2
        ]) # pads last dim by width_diff and second to last dim by height_diff
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
    def forward(self, x):
        return self.conv(x)
    
        
        

        






