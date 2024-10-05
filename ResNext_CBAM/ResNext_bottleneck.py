import torch
import torch.nn as nn

class Resnext_bottleneck_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, shrink=1, cardinality=32,
                 use_cbam=False,reduc_ratio_CBAM=16,CBAM=None,CBAM_skip=False, conv_padd=(1,1),kernel_size=3,max_pool=False):
        self.use_cbam=use_cbam
        super(Resnext_bottleneck_block, self).__init__()
        
        middle_out=int(out_channels*shrink)
        self.conv1 = nn.Conv2d(in_channels, middle_out, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(middle_out)
        self.conv2 = nn.Conv2d(middle_out, middle_out, kernel_size=kernel_size, stride=stride, padding=conv_padd, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(middle_out)
        self.conv3 = nn.Conv2d(middle_out, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        
        if self.use_cbam:
            self.cbam= CBAM(out_channels, reduc_ratio=reduc_ratio_CBAM,skip=CBAM_skip)
        
        self.relu = nn.ReLU(inplace=True)
        

        
        self.shortcut = nn.Sequential()


        if max_pool and (stride != 1 or in_channels != out_channels):
            self.shortcut = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=conv_padd)
            
        elif stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=int(kernel_size-2), stride=stride, padding=(int(conv_padd[0]-1), int(conv_padd[1]-1)), bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        # print(f'Before CBAM: {out.shape}')
        if self.use_cbam:
            out = self.cbam(out)
        # print(f'After CBAM: {out.shape}')
        out += self.shortcut(identity)
        out = self.relu(out)
        return out


# Modified Resnext_bottleneck_transpose_block with CBAM

class Resnext_bottleneck_transpose_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, shrink=2, convt_padd=(1, 1),
                 convt_out_padd=(0, 0), cardinality=32,use_cbam=False,reduc_ratio_CBAM=16,
                 CBAM=None, CBAM_skip=False,kernel_size=3, max_pool=False):  # convt_padd: must be at least 1 

        self.use_cbam=use_cbam
        super(Resnext_bottleneck_transpose_block, self).__init__()
        middle_out=int(out_channels*shrink)

        self.conv1 = nn.Conv2d(in_channels, middle_out, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(middle_out)
        self.conv2 = nn.ConvTranspose2d(middle_out, middle_out, kernel_size=kernel_size, stride=stride, padding=convt_padd, output_padding=convt_out_padd, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(middle_out)
        self.conv3 = nn.Conv2d(middle_out, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        if self.use_cbam:
            self.cbam= CBAM(out_channels, reduc_ratio=reduc_ratio_CBAM,skip=CBAM_skip)
        
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        
        if max_pool and (stride != 1 or in_channels != out_channels):
            self.shortcut = nn.MaxUnpool2d(kernel_size=kernel_size, stride=stride, padding=convt_padd)
            
        elif stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=int(kernel_size-2), stride=stride, padding=(int(convt_padd[0]-1), int(convt_padd[1]-1)), output_padding=convt_out_padd, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        # print(f'Before CBAM: {out.shape}')
        if self.use_cbam:
            out = self.cbam(out)
        # print(f'After CBAM: {out.shape}')
        out += self.shortcut(identity)
        out = self.relu(out)
        return out
    
    

