import torch
from torch import nn

class BasicBlock(nn.Module):
    '''
        inplanes: input channels
        planes: output channels
    '''
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # relu's parameter is shared across two layers
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # whether or not shrink the channel or size of the feature to fit the output dims
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # if downsample is set, then we need to match input dimension to that of out dimension
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)

        return out

class ResNet34(nn.Module):
    def __init__(self, blocks_each_layer, num_classes=13, block=BasicBlock):
        super().__init__()

        self.inplanes = 64
        self.init_channel = 1
        # layers for processing prior to res blocks
        self.conv1 = nn.Conv2d(self.init_channel, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(block, 64, blocks_each_layer[0])
        self.layer2 = self.make_layer(block, 128, blocks_each_layer[1], stride=2)
        self.layer3 = self.make_layer(block, 256, blocks_each_layer[2], stride=2)
        self.layer4 = self.make_layer(block, 512, blocks_each_layer[3], stride=2)
        # avgpool with output size (1,1) shrinks each plane (channel) to a scalar value
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)

    '''
        make_layer: build Residual Block
        inplanes: input feature dim
        planes: intermediate and output feature dim
        n_blocks: # of basic block per residual block
    '''
    def make_layer(self, block, planes, n_blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            # layer to downsample input dim
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes, 1, stride, bias=False), nn.BatchNorm2d(planes),)

        layers = []
        # append the 1st res block
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes

        for _ in range(1, n_blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
