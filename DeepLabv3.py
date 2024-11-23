
# there are missing pieces in the code, I lost the website I got it from <- note from Besan to Aya and Thuraya

def init_weights(m):
    classname = m._class.name_
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.constant_(0)


class ResNet_50(nn.Module):
    def _init_(self, in_channels=1, conv1_out=64):
        super(ResNet_50, self)._init_()
        
        self.resnet_50 = models.resnet50(pretrained=True)
        
        # Modify the first convolution layer to accept 1 input channel
        self.resnet_50.conv1 = nn.Conv2d(
            in_channels, conv1_out, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.resnet_50.bn1(self.resnet_50.conv1(x)))
        x = self.resnet_50.maxpool(x)
        x = self.resnet_50.layer1(x)
        x = self.resnet_50.layer2(x)
        x = self.resnet_50.layer3(x)
        return x

class ASSP(nn.Module):
    def _init_(self, in_channels, out_channels=256):
        super(ASSP, self)._init_()
        
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)

        self.adapool = nn.AdaptiveAvgPool2d(1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn5 = nn.BatchNorm2d(out_channels)

        self.convf = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False)
        self.bnf = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.relu(self.bn2(self.conv2(x)))
        x3 = self.relu(self.bn3(self.conv3(x)))
        x4 = self.relu(self.bn4(self.conv4(x)))
        
        x5 = self.adapool(x)
        x5 = self.relu(self.bn5(self.conv5(x5)))
        x5 = F.interpolate(x5, size=x4.shape[-2:], mode='bilinear')

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)  # Concatenate along channels
        x = self.relu(self.bnf(self.convf(x)))
        return x
    
    [11:33, 23/11/2024] Aya Amaireh: class DeepLabv3(nn.Module):
    def _init_(self, nc):
        super(DeepLabv3, self)._init_()
        self.nc = nc

        # Use the modified ResNet-50
        self.resnet = ResNet_50(in_channels=1)  # Single-channel input

        # Atrous Spatial Pyramid Pooling
        self.assp = ASSP(in_channels=1024)

        # Final 1x1 convolution to produce the required number of classes
        self.conv = nn.Conv2d(in_channels=256, out_channels=self.nc, kernel_size=1)

    def forward(self, x):
        _, _, h, w = x.shape  # Get input height and width
        x = self.resnet(x)  # Pass through ResNet-50
        x = self.assp(x)    # Pass through ASSP
        x = self.conv(x)    # Final 1x1 convolution
        x = F.interpolate(x, size=(h, w), mode='bilinear')  # Upsample to original size
        return x
