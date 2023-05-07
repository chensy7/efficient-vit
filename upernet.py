import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn as nn
import torch.nn.functional as F
from loss import OhemCELoss2D, CrossEntropyLoss
import copy
import torchvision.models as models

import transformers


model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_chan, out_chan, stride=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        self.norm_layer = norm_layer
        self.conv1 = conv3x3(in_chan, out_chan, stride)
        self.bn1 = norm_layer(out_chan, activation="leaky_relu")
        self.conv2 = conv3x3(out_chan, out_chan)
        self.bn2 = norm_layer(out_chan, activation="none")
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=stride, bias=False),
                norm_layer(out_chan, activation="none"),
            )

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(x)

        out_ = shortcut + out
        out_ = self.relu(out_)
        return out_


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_chan, out_chan, stride=1, base_width=64, norm_layer=None):
        super(Bottleneck, self).__init__()
        width = int(out_chan * (base_width / 64.0)) * 1
        self.norm_layer = norm_layer
        self.conv1 = conv1x1(in_chan, width)
        self.bn1 = norm_layer(width, activation="leaky_relu")
        self.conv2 = conv3x3(width, width, stride)
        self.bn2 = norm_layer(width, activation="leaky_relu")
        self.conv3 = conv1x1(width, out_chan * self.expansion)
        self.bn3 = norm_layer(out_chan * self.expansion, activation="none")
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if in_chan != out_chan * self.expansion or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_chan,
                    out_chan * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                norm_layer(out_chan * self.expansion, activation="none"),
            )

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.conv3(out)
        out = self.bn3(out)

        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(x)

        out_ = shortcut + out
        out_ = self.relu(out_)

        return out_


class ResNet(nn.Module):
    def __init__(self, block, layers, strides, norm_layer=None):
        super(ResNet, self).__init__()
        self.norm_layer = norm_layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64, activation="leaky_relu")
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inplanes = 64
        self.layer1 = self.create_layer(
            block, 64, bnum=layers[0], stride=strides[0], norm_layer=norm_layer
        )
        self.layer2 = self.create_layer(
            block, 128, bnum=layers[1], stride=strides[1], norm_layer=norm_layer
        )
        self.layer3 = self.create_layer(
            block, 256, bnum=layers[2], stride=strides[2], norm_layer=norm_layer
        )
        self.layer4 = self.create_layer(
            block, 512, bnum=layers[3], stride=strides[3], norm_layer=norm_layer
        )

    def create_layer(self, block, out_chan, bnum, stride=1, norm_layer=None):
        layers = [block(self.inplanes, out_chan, stride=stride, norm_layer=norm_layer)]
        self.inplanes = out_chan * block.expansion
        for i in range(bnum - 1):
            layers.append(
                block(self.inplanes, out_chan, stride=1, norm_layer=norm_layer)
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)

        feat4 = self.layer1(x)
        feat8 = self.layer2(feat4)  # 1/8
        feat16 = self.layer3(feat8)  # 1/16
        feat32 = self.layer4(feat16)  # 1/32
        return [feat4, feat8, feat16, feat32]

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, (self.norm_layer)):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params

    def init_weight(self, state_dict):
        self_state_dict = self.state_dict()
        for k, v in state_dict.items():
            if "fc" in k:
                continue
            self_state_dict.update({k: v})
        self.load_state_dict(self_state_dict, strict=True)


def Resnet18(pretrained=True, norm_layer=None, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], [2, 2, 2, 2], norm_layer=norm_layer)
    if pretrained:
        model.init_weight(model_zoo.load_url(model_urls["resnet18"]))
    return model


def Resnet34(pretrained=True, norm_layer=None, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], [2, 2, 2, 2], norm_layer=norm_layer)
    if pretrained:
        model.init_weight(model_zoo.load_url(model_urls["resnet34"]))
    return model


def Resnet50(pretrained=True, norm_layer=None, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], [2, 2, 2, 2], norm_layer=norm_layer)
    if pretrained:
        model.init_weight(model_zoo.load_url(model_urls["resnet50"]))
    return model


def Resnet101(pretrained=True, norm_layer=None, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], [2, 2, 2, 2], norm_layer=norm_layer)
    if pretrained:
        model.init_weight(model_zoo.load_url(model_urls["resnet101"]))
    return model


def Resnet152(pretrained=True, norm_layer=None, **kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], [2, 2, 2, 2], norm_layer=norm_layer)
    if pretrained:
        model.init_weight(model_zoo.load_url(model_urls["resnet152"]))
    return model


up_kwargs = {"mode": "bilinear", "align_corners": True}


class BatchNorm2d(nn.BatchNorm2d):
    """(conv => BN => ReLU) * 2"""

    def __init__(self, num_features, activation="none"):
        super(BatchNorm2d, self).__init__(num_features=num_features)
        if activation == "leaky_relu":
            self.activation = nn.LeakyReLU()
        elif activation == "none":
            self.activation = lambda x: x
        else:
            raise Exception("Accepted activation: ['leaky_relu']")

    def forward(self, x):
        return self.activation(super(BatchNorm2d, self).forward(x))

class PSPModule(nn.Module):
    # In the original inmplementation they use precise RoI pooling 
    # Instead of using adaptative average pooling
    def __init__(self, in_channels, bin_sizes=[1, 2, 4, 6]):
        super(PSPModule, self).__init__()
        out_channels = in_channels // len(bin_sizes)
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, b_s) 
                                                        for b_s in bin_sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+(out_channels * len(bin_sizes)), in_channels, 
                                    kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def _make_stages(self, in_channels, out_channels, bin_sz):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)
    
    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear', 
                                        align_corners=True) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, (nn.BatchNorm2d)):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params

def up_and_add(x, y):
    return F.interpolate(x, size=(y.size(2), y.size(3)), mode='bilinear', align_corners=True) + y

class FPN_fuse(nn.Module):
    def __init__(self, feature_channels=[256, 512, 1024, 2048], fpn_out=256):
        super(FPN_fuse, self).__init__()
        assert feature_channels[0] == fpn_out
        self.conv1x1 = nn.ModuleList([nn.Conv2d(ft_size, fpn_out, kernel_size=1)
                                    for ft_size in feature_channels[1:]])
        self.smooth_conv =  nn.ModuleList([nn.Conv2d(fpn_out, fpn_out, kernel_size=3, padding=1)] 
                                    * (len(feature_channels)-1))
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(len(feature_channels)*fpn_out, fpn_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):
        
        features[1:] = [conv1x1(feature) for feature, conv1x1 in zip(features[1:], self.conv1x1)]
        P = [up_and_add(features[i], features[i-1]) for i in reversed(range(1, len(features)))]
        P = [smooth_conv(x) for smooth_conv, x in zip(self.smooth_conv, P)]
        P = list(reversed(P))
        P.append(features[-1]) #P = [P1, P2, P3, P4]
        H, W = P[0].size(2), P[0].size(3)
        P[1:] = [F.interpolate(feature, size=(H, W), mode='bilinear', align_corners=True) for feature in P[1:]]

        x = self.conv_fusion(torch.cat((P), dim=1))
        return x

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, (nn.BatchNorm2d)):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params

def swin_v2_b_forward(model, x):
    model = model.features
    for i in range(len(model)):
        x = model[i](x)
        if i == 1: feat4 = x.transpose(1, 2).transpose(1, 3)
        elif i == 3: feat8 = x.transpose(1, 2).transpose(1, 3)
        elif i == 5: feat16 = x.transpose(1, 2).transpose(1, 3)
        elif i == 7: feat32 = x.transpose(1, 2).transpose(1, 3)
    return [feat4, feat8, feat16, feat32]

def swin_v2_t_forward(model, x):
    model = model.features
    for i in range(len(model)):
        x = model[i](x)
        if i == 1: feat4 = x.transpose(1, 2).transpose(1, 3)
        elif i == 3: feat8 = x.transpose(1, 2).transpose(1, 3)
        elif i == 5: feat16 = x.transpose(1, 2).transpose(1, 3)
        elif i == 7: feat32 = x.transpose(1, 2).transpose(1, 3)
    return [feat4, feat8, feat16, feat32]

def swin_v2_get_params(model):
    wd_params, nowd_params = [], []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            wd_params.append(module.weight)
            # if not module.bias is None:
            #     nowd_params.append(module.bias)
        # elif isinstance(module, (model.norm_layer)):
        #     nowd_params += list(module.parameters())
    return wd_params, nowd_params

def mobilevit_forward(model, x):
    x = model.conv_stem(x)
    for i in range(len(model.encoder.layer)):
        x = model.encoder.layer[i](x)
        if i == 1: feat4 = x
        elif i == 2: feat8 = x
        elif i == 3: feat16 = x
        elif i == 4: feat32 = x
    return [feat4, feat8, feat16, feat32]

def mobilevit_get_params(model):
    wd_params, nowd_params = [], []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            wd_params.append(module.weight)
            # if not module.bias is None:
            #     nowd_params.append(module.bias)
        # elif isinstance(module, (model.norm_layer)):
        #     nowd_params += list(module.parameters())
    return wd_params, nowd_params

class UperNet(nn.Module):
    def __init__(
        self, nclass=21, backbone="resnet18", loss_fn=None, fpn_out=64
    ):
        super(UperNet, self).__init__()

        self.loss_fn = loss_fn
        self.nclass = nclass

        if backbone == "swin_v2_b":
            feature_channels = [128, 256, 512, 1024]
            fpn_out = 128
        elif backbone == "resnet18":
            feature_channels = [64, 128, 256, 512]
            fpn_out = 64
        elif backbone == "mobilevit":
            feature_channels = [64, 96, 128, 160]
            fpn_out = 64
        elif backbone == "resnet50":
            feature_channels = [256, 512, 1024, 2048]
            fpn_out = 256
        elif backbone == "swin_v2_t":
            feature_channels = [96, 192, 384, 768]
            fpn_out = 96

        self.backbone_type = backbone
        if backbone == "swin_v2_b":
            self.backbone = models.swin_v2_b(weights='DEFAULT')
        elif backbone == "swin_v2_t":
            self.backbone = models.swin_v2_t(weights='DEFAULT')
        elif backbone == "resnet18":
            self.backbone = Resnet18(norm_layer=BatchNorm2d)
        elif backbone == "resnet50":
            self.backbone = Resnet18(norm_layer=BatchNorm2d)
        elif backbone == "mobilevit":
            self.backbone = transformers.MobileViTModel.from_pretrained("apple/mobilevit-small", num_attention_heads=2)
        else:
            raise RuntimeError("unknown backbone: {}".format(backbone))

        self.PPN = PSPModule(feature_channels[-1])
        self.FPN = FPN_fuse(feature_channels, fpn_out=fpn_out)
        self.head = nn.Conv2d(fpn_out, nclass, kernel_size=3, padding=1)

    def forward(self, x, lbl=None, teacher_outputs=None):

        input_size = (x.size()[2], x.size()[3])

        if self.backbone_type == "swin_v2_b":
            features = swin_v2_b_forward(self.backbone, x)
        elif self.backbone_type == "swin_v2_t":
            features = swin_v2_t_forward(self.backbone, x)
        elif self.backbone_type == "mobilevit":
            features = mobilevit_forward(self.backbone, x)
        else:
            features = self.backbone(x)
        features[-1] = self.PPN(features[-1])
        x = self.head(self.FPN(features))

        outputs = F.interpolate(x, size=input_size, mode='bilinear')
        if self.training:
            loss = (
                self.loss_fn(outputs, lbl)
            )
            if teacher_outputs is not None:
                # T = 1
                # alpha = 1
                # t_loss = F.kl_div(F.log_softmax(outputs/T, dim=1),
                #                  F.log_softmax(teacher_outputs/T, dim=1), reduction='batchmean') * (alpha * T * T)
                t_loss = F.mse_loss(outputs, teacher_outputs)
                loss += t_loss
            return loss
        else:
            return outputs

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            if isinstance(child, (OhemCELoss2D, CrossEntropyLoss)):
                continue
            # elif isinstance(child, (LAFeatureFusionModule, FPNOutput)):
            #     child_wd_params, child_nowd_params = child.get_params()
            #     lr_mul_wd_params += child_wd_params
            #     lr_mul_nowd_params += child_nowd_params
            else:
                if isinstance(child, models.SwinTransformer):
                    child_wd_params, child_nowd_params = swin_v2_get_params(child)
                elif isinstance(child, transformers.MobileViTModel):
                    child_wd_params, child_nowd_params = mobilevit_get_params(child)
                elif isinstance(child, nn.Conv2d):
                    child_wd_params = child.parameters()
                    child_nowd_params = []
                else:
                    child_wd_params, child_nowd_params = child.get_params()
                wd_params += child_wd_params
                nowd_params += child_nowd_params
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params

def get_model(model_dict, nclass, loss_fn=None):
    name = model_dict["arch"]
    model = _get_model_instance(name)
    param_dict = copy.deepcopy(model_dict)
    param_dict.pop("arch")
    param_dict["loss_fn"] = loss_fn

    model = model(nclass=nclass, **param_dict)
    return model

def _get_model_instance(name):
    try:
        return {"upernet": UperNet}[name]
    except:
        raise ("Model {} not available".format(name))