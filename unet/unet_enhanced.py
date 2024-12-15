import torch
from torch import nn
import torch.nn.functional as F


# SEBlock for channel attention
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y  # Scale input by attention weights


# DoubleConv with optional SEBlock
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0, use_attention=False):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
        self.attention = SEBlock(out_ch) if use_attention else nn.Identity()  # Optional attention
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.dropout(self.attention(self.conv(x)))


class DownLayer(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0, use_attention=False):
        super(DownLayer, self).__init__()
        self.pool = nn.MaxPool2d(2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch, dropout, use_attention)

    def forward(self, x):
        return self.conv(self.pool(x))


class UpLayer(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0, use_attention=False):
        super(UpLayer, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_ch + out_ch, out_ch, dropout, use_attention)

    def forward(self, x1, x2):
        x2 = self.up(x2)
        diffY, diffX = x1.size(2) - x2.size(2), x1.size(3) - x2.size(3)
        x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)


# UNet with Attention (no dense layers)
class UNet_enhanced(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, dropout=0.0, output_activation="sigmoid", use_attention=False):
        super(UNet, self).__init__()
        self.output_activation = output_activation

        # Encoder
        self.conv1 = DoubleConv(in_channels, 64, dropout, use_attention)
        self.down1 = DownLayer(64, 128, dropout, use_attention)
        self.down2 = DownLayer(128, 256, dropout, use_attention)
        self.down3 = DownLayer(256, 512, dropout, use_attention)
        self.down4 = DownLayer(512, 1024, dropout, use_attention)

        # Decoder
        self.up1 = UpLayer(1024, 512, dropout, use_attention)
        self.up2 = UpLayer(512, 256, dropout, use_attention)
        self.up3 = UpLayer(256, 128, dropout, use_attention)
        self.up4 = UpLayer(128, 64, dropout, use_attention)

        # Final layer
        self.last_conv = nn.Conv2d(64, out_channels, kernel_size=1)

        self._initialize_weights()

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

        if self.output_activation == "sigmoid":
            return torch.sigmoid(output)
        elif self.output_activation == "softmax":
            return F.softmax(output, dim=1)
        else:
            return output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
