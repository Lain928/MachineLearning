from torch import nn


class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        block = [nn.Conv2d(...)]
        block += [nn.ReLU()]
        block += [nn.BatchNorm2d(...)]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)

class SimpleNetwork(nn.Module):
    def __init__(self, num_resnet_blocks=6):
        super(SimpleNetwork, self).__init__()
        # here we add the individual layers
        layers = [ConvBlock(...)]
        for i in range(num_resnet_blocks):
            layers += [ResBlock(...)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)