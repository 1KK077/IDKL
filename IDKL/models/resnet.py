import torch.nn as nn
from torch.nn import functional as F
# from torchvision.models.utils import load_state_dict_from_url   #原来
from torch.hub import load_state_dict_from_url

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
}


class convDiscrimination(nn.Module):
    def __init__(self, dim=512):
        super(convDiscrimination, self).__init__()
        self.conv1 = conv3x3(dim, 512, stride=2)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = conv3x3(512, 128, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = conv3x3(128, 128, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        x = F.dropout(F.relu(self.bn1(self.conv1(x))), training=self.training)
        x = F.dropout(F.relu(self.bn2(self.conv2(x))), training=self.training)
        x = F.dropout(F.relu(self.bn3(self.conv3(x))), training=self.training)
        x = F.avg_pool2d(x, (x.size(2), x.size(3)))
        x = x.view(-1, 128)
        x = self.fc(x)
        return x


class Discrimination(nn.Module):
    def __init__(self, dim=2048):
        super(Discrimination, self).__init__()
        self.fc1 = nn.Linear(dim, 100)
        self.bn1 = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, 100)
        self.bn2 = nn.BatchNorm1d(100)
        self.fc3 = nn.Linear(100, 2)

    def forward(self, x):
        x = F.dropout(F.relu(self.bn1(self.fc1(x))), training=self.training)
        x = F.dropout(F.relu(self.bn2(self.fc2(x))), training=self.training)
        x = self.fc3(x)
        return x


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class MAM(nn.Module):
    def __init__(self, dim, r=16):
        super(MAM, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.Conv2d(dim, dim // r, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // r, dim, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.IN = nn.InstanceNorm2d(dim, track_running_stats=False)

    def forward(self, x):
        pooled = F.avg_pool2d(x, x.size()[2:])
        mask = self.channel_attention(pooled)
        x = x * mask + self.IN(x) * (1 - mask)

        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, zero_init_residual=False, modality_attention=0,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, drop_last_stride=False):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1 if drop_last_stride else 2,
                                       dilate=replace_stride_with_dilation[2])


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

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

        return x


class Shared_module_fr(nn.Module):
    def __init__(self, drop_last_stride, modality_attention):
        super(Shared_module_fr, self).__init__()

        model_sh_fr = resnet50(pretrained=True, drop_last_stride=drop_last_stride,
                               modality_attention=modality_attention)
        # avg pooling to global pooling
        self.model_sh_fr = model_sh_fr

    def forward(self, x):
        x = self.model_sh_fr.conv1(x)
        x = self.model_sh_fr.bn1(x)
        x = self.model_sh_fr.relu(x)
        x = self.model_sh_fr.maxpool(x)
        x = self.model_sh_fr.layer1(x)
        x = self.model_sh_fr.layer2(x)
        # x = self.model_sh_fr.layer3(x)
        return x


class Special_module(nn.Module):
    def __init__(self, drop_last_stride, modality_attention):
        super(Special_module, self).__init__()

        special_module = resnet50(pretrained=True, drop_last_stride=drop_last_stride,
                                )

        self.special_module = special_module

    def forward(self, x):
        # x = self.special_module.layer2(x)
        x = self.special_module.layer3(x)
        x = self.special_module.layer4(x)
        return x


class Shared_module_bh(nn.Module):
    def __init__(self, drop_last_stride, modality_attention):
        super(Shared_module_bh, self).__init__()

        model_sh_bh = resnet50(pretrained=True, drop_last_stride=drop_last_stride,)  # model_sh_fr  model_sh_bh

        self.model_sh_bh = model_sh_bh  # self.model_sh_bh = model_sh_bh  #self.model_sh_fr = model_sh_fr

    def forward(self, x):
        # x = self.model_sh_bh.layer2(x)
        x_sh3 = self.model_sh_bh.layer3(x)  # self.model_sh_fr  self.model_sh_bh
        x_sh4 = self.model_sh_bh.layer4(x_sh3)  # self.model_sh_fr  self.model_sh_bh
        return x_sh3, x_sh4


class Mask(nn.Module):
    def __init__(self, dim, r=16):
        super(Mask, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.Conv2d(dim, dim // r, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // r, dim, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        mask = self.channel_attention(x)
        return mask


class special_att(nn.Module):
    def __init__(self, dim, r=16):
        super(special_att, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.Conv2d(dim, dim // r, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // r, dim, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.IN = nn.InstanceNorm2d(dim, track_running_stats=False) #self.IN = nn.InstanceNorm2d(dim, track_running_stats=True, affine=True)

    def forward(self, x):
        x_IN = self.IN(x)
        x_R = x - x_IN
        pooled = gem(x_R)
        mask = self.channel_attention(pooled)
        x_sp = x_R * mask + x_IN  # x

        return x_sp, x_IN


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)


class embed_net(nn.Module):
    def __init__(self, drop_last_stride,  decompose=False):
        super(embed_net, self).__init__()

        self.shared_module_fr = Shared_module_fr(drop_last_stride=drop_last_stride,
                                                 )
        self.shared_module_bh = Shared_module_bh(drop_last_stride=drop_last_stride,
                                                 )

        self.special = Special_module(drop_last_stride=drop_last_stride)

        self.decompose = decompose
        self.IN = nn.InstanceNorm2d(2048, track_running_stats=True, affine=True)
        if decompose:
            self.special_att = special_att(2048)
            self.mask1 = Mask(2048)
            self.mask2 = Mask(2048)

    def forward(self, x):
        x2 = self.shared_module_fr(x)
        x3, x_sh = self.shared_module_bh(x2)  # bchw

        sh_pl = gem(x_sh).squeeze()  # Gem池化
        sh_pl = sh_pl.view(sh_pl.size(0), -1)  # Gem池化

        if self.decompose:
            ######special structure

            x_sp_f = self.special(x2)
            sp_IN = self.IN(x_sp_f)
            m_IN = self.mask1(sp_IN)
            m_F = self.mask2(x_sp_f)
            sp_IN_p = m_IN * sp_IN
            x_sp_f_p = m_F * x_sp_f
            x_sp = m_IN * x_sp_f_p + m_F * sp_IN_p

            sp_pl = gem(x_sp).squeeze()  # Gem池化
            sp_pl = sp_pl.view(sp_pl.size(0), -1)  # Gem池化


        return x_sh,  sh_pl, sp_pl,sp_IN,sp_IN_p,x_sp_f,x_sp_f_p


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNeXt-50 32x4d model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNeXt-101 32x8d model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)
