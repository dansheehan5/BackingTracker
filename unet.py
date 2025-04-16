import torchaudio
import torch
from torch import nn

### Encoder Block
class EncConv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, ker_size):
        super(EncConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels

        ### 2 Convolutional layers w/ batch norm and ReLU
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels,
                      kernel_size=ker_size, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels,
                      kernel_size=ker_size,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        logits = self.conv_block(x)
        return nn.MaxPool2d(2)(logits)


class DecConv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, ker_size):
        super(DecConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        ### Upsample
        self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=2, stride=2)
        ### 2 Convolutional layers w/ batch norm and ReLU
        self.conv_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=mid_channels,
                               kernel_size=ker_size, stride=2, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=mid_channels, out_channels=out_channels,
                               kernel_size=ker_size, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        logits = self.up(x)

        ### Pads this layer to same size as concat layer
        diff_y = skip.shape[2] - logits.shape[2]
        diff_x = skip.shape[3] - logits.shape[3]
        logits = nn.functional.pad(logits, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        logits = torch.concat((logits, skip), dim=1)
        logits = self.conv_block(logits)
        return logits

### Full U-Net architecture
class UNet(nn.Module):
    def __init__(self, n_channels, batch_size):
        super(UNet, self).__init__()
        ### First Convolutional layer to get # of channels to 64
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=64,kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64,kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        ### Double channels at each encoding layer
        self.down1 = EncConv(64, None, 64, 3)
        self.down2 = EncConv(64, None, 128, 3)
        self.down3 = EncConv(128, None, 256, 3)
        self.down4 = EncConv(256, None, 512, 3)
        self.down5 = EncConv(512, None, 1024, 3)
        ### Halve channels at each decoding layer
        self.up1 = DecConv(1024, None, 512, 3)
        self.up2 = DecConv(512, None, 256, 3)
        self.up3 = DecConv(256, None, 128, 3)
        self.up4 = DecConv(128, None, 64, 3)
        ### Output layer to bring back original channels
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=n_channels,kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels,kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
        )



    def forward(self, x):
        ### Channels -> 64
        in_layer = self.in_conv(x)
        ### ENCODE
        s1 = self.down1(in_layer)
        s2 = self.down2(s1)
        s3 = self.down3(s2)
        s4 = self.down4(s3)
        s5 = self.down5(s4)
        ### DECODE
        u1 = self.up1(s5, s4)
        u2 = self.up2(u1, s3)
        u3 = self.up3(u2, s2)
        u4 = self.up4(u3, s1)
        ### Prepare output
        out = self.out_conv(u4)
        ### Ensure output is correct shape
        out = nn.AvgPool2d(2)(out)
        diff_y = x.shape[2] - out.shape[2]
        diff_x = x.shape[3] - out.shape[3]
        out = nn.functional.pad(out, [diff_x // 2, diff_x - diff_x // 2,
                                            diff_y // 2, diff_y - diff_y // 2])
        return out