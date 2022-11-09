import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F 


__all__ = ['ResNet', 'resnet18','resnet50']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


#---ResNet encoder------------------------------------------------------------------------------------------------------

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # ipdb.set_trace()
        residual = x
        # x 64, 64, 64, 64

        # 64, 64, 64, 64
        out = self.conv1(x)
        # 64, 64, 64, 64
        out = self.bn1(out)
        # 64, 64, 64, 64
        out = self.relu(out)

        # 
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        # 64, 256, 64 ,64
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, image_size, return_fm=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.return_fm = return_fm 
        # define resnet encoder
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)
        self.layer1 = self._make_layer(block, 64, layers[0])  # 64-64
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # 64-128
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # 128-256
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # 256-512
        self.avgpooling = nn.AdaptiveAvgPool2d((1,1))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):  # x:image y:ingredient
        

        x = self.conv1(x)  # ／2

        x = self.bn1(x)

        x = self.relu(x)

        [x, a] = self.maxpool(x)  # ／2

        x = self.layer1(x)
        x = self.layer2(x)  # ／2
        x = self.layer3(x)  # ／2
        x = self.layer4(x)  # ／2
        
        if self.return_fm:
            return x
        else:
            x = self.avgpooling(x)  # (1x1)
            x = x.view(x.shape[0], x.shape[1])
            return x


def resnet50(image_size, pretrained=False, return_fm=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], image_size, return_fm=return_fm, **kwargs)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
        model_dict = model.state_dict()

        shared_dict = {k : v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(shared_dict)
        model.load_state_dict(model_dict)
    return model


def resnet18(image_size, pretrained=False, return_fm=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], image_size, return_fm=return_fm, **kwargs)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet18'])
        model_dict = model.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model
