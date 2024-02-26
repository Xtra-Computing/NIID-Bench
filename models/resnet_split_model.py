import torch
from torch import nn

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        self.downsample = downsample
        if downsample:
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        # PRINT HERE - cov2_1 conv 4_1 and conv 5_1
        input = nn.ReLU()(self.bn2(self.conv2(input)))
        input = input + shortcut

        return nn.ReLU()(input)


class ResBottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_channels, out_channels//4,
                               kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(
            out_channels//4, out_channels//4, kernel_size=3, stride=2 if downsample else 1, padding=1)
        self.conv3 = nn.Conv2d(out_channels//4, out_channels, kernel_size=1, stride=1)

        if self.downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=2 if self.downsample else 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()

        self.bn1 = nn.BatchNorm2d(out_channels//4)
        self.bn2 = nn.BatchNorm2d(out_channels//4)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = nn.ReLU()(self.bn2(self.conv2(input)))
        input = nn.ReLU()(self.bn3(self.conv3(input)))
        input = input + shortcut
        return nn.ReLU()(input)

class ResNet(nn.Module):
    def __init__(self, in_channels, resblock, repeat, useBottleneck=False, outputs=10, first_cut=-1, last_cut=-1):
        itter = 0
        super().__init__()
        
        self.first_cut = first_cut
        self.last_cut = last_cut
        start = False
        end = False

        if self.last_cut == -1:
            end = True

        if self.first_cut == -1:
            start = True

        if useBottleneck:
            filters = [64, 256, 512, 1024, 2048]
        else:
            filters = [64, 64, 128, 256, 512]

        self.total = 3
        for i in repeat:
            self.total = self.total + i
        
        
        # from the beginning
        if first_cut == -1:
            self.layers = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )
        else:
            self.layers = nn.Sequential()
        
        itter += 1

        if start or ((not start) and (self.first_cut == itter)): #when to start
            # have already started, or just started
            if end or ((not end) and (self.last_cut > itter)):
                self.layers.add_module('conv2_1', resblock(filters[0], filters[1], downsample=False))  

                start = True
            else:
                # reached end
                return 

        
        for i in range(1, repeat[0]):
            itter += 1
            
            if start or ((not start) and (self.first_cut == itter)): #when to start
                # have already started, or just started
                if end or ((not end) and (self.last_cut > itter)):
                    self.layers.add_module('conv2_%d'%(i+1,), resblock(filters[1], filters[1], downsample=False))
    
                    start = True
                else:
                    # reached end
                    return 

        
        itter += 1
            
        if start or ((not start) and (self.first_cut == itter)): #when to start
            # have already started, or just started
            if end or ((not end) and (self.last_cut > itter)):
                self.layers.add_module('conv3_1', resblock(filters[1], filters[2], downsample=True))

                start = True
            else:
                # reached end
                return 

        
        for i in range(1, repeat[1]):
            itter += 1
            if start or ((not start) and (self.first_cut == itter)): #when to start
                # have already started, or just started
                if end or ((not end) and (self.last_cut > itter)):
                    self.layers.add_module('conv3_%d' % (i+1,), resblock(filters[2], filters[2], downsample=False))
    
                    start = True
                else:
                    # reached end
                    return 



        itter += 1
        if start or ((not start) and (self.first_cut == itter)): #when to start
            # have already started, or just started
            if end or ((not end) and (self.last_cut > itter)):
                self.layers.add_module('conv4_1', resblock(filters[2], filters[3], downsample=True))

                start = True
            else:
                # reached end
                return 

        for i in range(1, repeat[2]):
            itter += 1
            if start or ((not start) and (self.first_cut == itter)): #when to start
                # have already started, or just started
                if end or ((not end) and (self.last_cut > itter)):
                    self.layers.add_module('conv4_%d' % (i+1,), resblock(filters[3], filters[3], downsample=False))
    
                    start = True
                else:
                    # reached end
                    return 


        itter += 1
        if start or ((not start) and (self.first_cut == itter)): #when to start
            # have already started, or just started
            if end or ((not end) and (self.last_cut > itter)):
                self.layers.add_module('conv5_1', resblock(filters[3], filters[4], downsample=True))

                start = True
            else:
                # reached end
                return 
        
        for i in range(1, repeat[3]):
            itter += 1
            if start or ((not start) and (self.first_cut == itter)): #when to start
                # have already started, or just started
                if end or ((not end) and (self.last_cut > itter)):
                    self.layers.add_module('conv5_%d'%(i+1,),resblock(filters[4], filters[4], downsample=False))
    
                    start = True
                else:
                    # reached end
                    return 
                
        # last layers
        itter += 1
        if start or ((not start) and (self.first_cut == itter)): #when to start
            # have already started, or just started
            if end or ((not end) and (self.last_cut > itter)):
                self.gap = torch.nn.AdaptiveAvgPool2d(1)
                start = True
            else:
                # reached end
                return
        
        itter += 1
        if start or ((not start) and (self.first_cut == itter)): #when to start
            # have already started, or just started
            if end or ((not end) and (self.last_cut > itter)):
                self.fc = torch.nn.Linear(filters[4], outputs)
                start = True
            else:
                # reached end
                return 

        
    def forward(self, input):
        if ((self.first_cut == -1) or (self.first_cut != -1 and self.first_cut <  self.total - 2)): #not empty
            input = self.layers(input)       
        if ((self.last_cut == -1) or ((self.last_cut != -1) and self.last_cut >= self.total - 1)):
            input = self.gap(input)
            input = torch.flatten(input, start_dim=1) 
        if (self.last_cut == -1):
            input = self.fc(input)

        return input

def get_resnet18(outputs=10, first_cut=-1, last_cut=-1):
    return ResNet(3, ResBlock, [2, 2, 2, 2], False, outputs, first_cut, last_cut)

def get_resnet34(outputs=10, first_cut=-1, last_cut=-1):
    return ResNet(3, ResBlock, [2, 2, 2, 2], False, outputs, first_cut, last_cut)

def get_resnet50(outputs=10, first_cut=-1, last_cut=-1):
    return ResNet(3, ResBottleneckBlock, [3, 4, 6, 3], True, outputs, first_cut, last_cut)

def get_resnet101(outputs=10, first_cut=-1, last_cut=-1):
    return ResNet(3, ResBottleneckBlock, [3, 4, 23, 3], True, outputs, first_cut, last_cut)

def get_resnet152(outputs=10, first_cut=-1, last_cut=-1):
    return ResNet(3, ResBottleneckBlock, [3, 8, 36, 3], True, outputs, first_cut, last_cut)


def get_resnet_split(outputs, first_cut, last_cut, type):
    if type == 'resnet18':
        model_part_a = get_resnet18(outputs, -1, first_cut)
        model_part_b = get_resnet18(outputs, first_cut, last_cut)
        model_part_c = get_resnet18(outputs, last_cut, -1)
    if type == 'resnet34':
        model_part_a = get_resnet34(outputs, -1, first_cut)
        model_part_b = get_resnet34(outputs, first_cut, last_cut)
        model_part_c = get_resnet34(outputs, last_cut, -1)
    if type == 'resnet50':
        model_part_a = get_resnet50(outputs, -1, first_cut)
        model_part_b = get_resnet50(outputs, first_cut, last_cut)
        model_part_c = get_resnet50(outputs, last_cut, -1)
    if type == 'resnet101':
        model_part_a = get_resnet101(outputs, -1, first_cut)
        model_part_b = get_resnet101(outputs, first_cut, last_cut)
        model_part_c = get_resnet101(outputs, last_cut, -1)
    if type == 'resnet152':
        model_part_a = get_resnet152(outputs, -1, first_cut)
        model_part_b = get_resnet152(outputs, first_cut, last_cut)
        model_part_c = get_resnet152(outputs, last_cut, -1)

    return (model_part_a, model_part_b, model_part_c)