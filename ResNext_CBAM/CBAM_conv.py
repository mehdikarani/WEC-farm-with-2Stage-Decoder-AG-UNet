import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduc_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // reduc_ratio, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // reduc_ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, gate_channels=None, reduc_ratio=16, kernel_size=7,skip=False):
        self.skip=skip
        self.gate_channels=gate_channels
        super(CBAM, self).__init__()
        self.ChannelAttention = ChannelAttention(self.gate_channels, reduc_ratio)
        self.SpatialAttention = SpatialAttention(kernel_size)

    def forward(self, x):
        x_out = self.ChannelAttention(x) * x
        x_out = self.SpatialAttention(x_out) * x_out
        if self.skip:
            x_out=x_out+x
        return x_out

# Example usage:
# model = CBAM(gate_channels=64, reduc_ratio=16, kernel_size=7)
# input_tensor = torch.randn(1, 64, 32, 32)
# output_tensor = model(input_tensor)
# print(output_tensor.shape)
