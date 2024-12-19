import torch
from torch import nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DownLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownLayer, self).__init__()
        self.pool = nn.MaxPool2d(2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class UpLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpLayer, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_ch + out_ch, out_ch)  # Correct concatenated channels

    def forward(self, x1, x2):
        # Upsample x2
        x2 = self.up(x2)

        # Pad x2 if dimensions differ from x1
        diffY = x1.size(2) - x2.size(2)
        diffX = x1.size(3) - x2.size(3)
        x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        # Debugging: Ensure dimensions match before concatenation
        assert x1.size(2) == x2.size(2) and x1.size(3) == x2.size(3), \
            f"Shape mismatch: x1 {x1.size()} x2 {x2.size()}"

        # Concatenate along the channel dimension
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)



class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super(UNet, self).__init__()
        self.conv1 = DoubleConv(in_channels, 64)
        self.down1 = DownLayer(64, 128)
        self.down2 = DownLayer(128, 256)
        self.down3 = DownLayer(256, 512)
        self.down4 = DownLayer(512, 1024)

        self.up1 = UpLayer(1024, 512)
        self.up2 = UpLayer(512, 256)
        self.up3 = UpLayer(256, 128)
        self.up4 = UpLayer(128, 64)

        self.last_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.conv1(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder
        x4_up = self.up1(x4, x5)
        x3_up = self.up2(x3, x4_up)
        x2_up = self.up3(x2, x3_up)
        x1_up = self.up4(x1, x2_up)

        # Final output
        output = self.last_conv(x1_up)
        
        output = F.sigmoid(output)  # Apply sigmoid for binary segmentation
        return output