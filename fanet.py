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
        return feat4, feat8, feat16, feat32

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

def swin_v2_b_forward(model, x):
    model = model.features
    for i in range(len(model)):
        x = model[i](x)
        if i == 1: feat4 = x.transpose(1, 2).transpose(1, 3)
        elif i == 3: feat8 = x.transpose(1, 2).transpose(1, 3)
        elif i == 5: feat16 = x.transpose(1, 2).transpose(1, 3)
        elif i == 7: feat32 = x.transpose(1, 2).transpose(1, 3)
    return feat4, feat8, feat16, feat32

def swin_v2_b_get_params(model):
    wd_params, nowd_params = [], []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            wd_params.append(module.weight)
            # if not module.bias is None:
            #     nowd_params.append(module.bias)
        # elif isinstance(module, (model.norm_layer)):
        #     nowd_params += list(module.parameters())
    return wd_params, nowd_params

class FANet(nn.Module):
    def __init__(
        self, nclass=21, backbone="resnet18", norm_layer=BatchNorm2d, loss_fn=None
    ):
        super(FANet, self).__init__()

        self.loss_fn = loss_fn
        self.norm_layer = norm_layer
        self._up_kwargs = up_kwargs
        self.nclass = nclass
        # copying modules from pretrained models
        self.backbone_type = backbone
        if backbone == "resnet18":
            self.expansion = 1
            self.backbone = Resnet18(norm_layer=norm_layer)
        elif backbone == "resnet50":
            self.expansion = 4
            self.backbone = Resnet50(norm_layer=norm_layer)
        elif backbone == "swin_v2_b":
            self.expansion = 2
            self.backbone = models.swin_v2_b(weights='DEFAULT')
        else:
            raise RuntimeError("unknown backbone: {}".format(backbone))
        # bilinear upsample options

        self.ffm_32 = LAFeatureFusionModule(
            512 * self.expansion, 256, 128, norm_layer=norm_layer
        )
        self.ffm_16 = LAFeatureFusionModule(
            256 * self.expansion, 256, 128, norm_layer=norm_layer
        )
        self.ffm_8 = LAFeatureFusionModule(
            128 * self.expansion, 256, 128, norm_layer=norm_layer
        )
        self.ffm_4 = LAFeatureFusionModule(
            64 * self.expansion, 256, 128, norm_layer=norm_layer
        )

        self.clslayer_32 = FPNOutput(128, 64, nclass, norm_layer=norm_layer)
        self.clslayer_16 = FPNOutput(128, 64, nclass, norm_layer=norm_layer)
        self.clslayer_8 = FPNOutput(256, 256, nclass, norm_layer=norm_layer)

    def forward(self, x, lbl=None):

        _, _, h, w = x.size()

        if self.backbone_type == "swin_v2_b":
            feat4, feat8, feat16, feat32 = swin_v2_b_forward(self.backbone, x)
        elif self.backbone_type == "mobilevit":
            feat4, feat8, feat16, feat32 = mobilevit_forward(self.backbone, x)
        else:
            feat4, feat8, feat16, feat32 = self.backbone(x)

        upfeat_32, smfeat_32 = self.ffm_32(feat32, None, True, True)
        upfeat_16, smfeat_16 = self.ffm_16(feat16, upfeat_32, True, True)
        upfeat_8 = self.ffm_8(feat8, upfeat_16, True, False)
        smfeat_4 = self.ffm_4(feat4, upfeat_8, False, True)

        x = self._upsample_cat(smfeat_16, smfeat_4)

        x = self.clslayer_8(x)
        outputs = F.interpolate(x, (h, w), **self._up_kwargs)

        # Auxiliary layers for training
        if self.training:
            auxout_1 = self.clslayer_32(smfeat_32)
            auxout_2 = self.clslayer_16(smfeat_16)
            auxout_1 = F.interpolate(auxout_1, (h, w), **self._up_kwargs)
            auxout_2 = F.interpolate(auxout_2, (h, w), **self._up_kwargs)
            loss = (
                self.loss_fn(outputs, lbl)
                + 0.5 * self.loss_fn(auxout_1, lbl)
                + 0.5 * self.loss_fn(auxout_2, lbl)
            )
            return loss
        else:
            return outputs

    def _upsample_cat(self, x1, x2):
        """Upsample and concatenate feature maps."""
        _, _, H, W = x2.size()
        x1 = F.interpolate(x1, (H, W), **self._up_kwargs)
        x = torch.cat([x1, x2], dim=1)
        return x

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
            elif isinstance(child, (LAFeatureFusionModule, FPNOutput)):
                child_wd_params, child_nowd_params = child.get_params()
                lr_mul_wd_params += child_wd_params
                lr_mul_nowd_params += child_nowd_params
            else:
                if isinstance(child, models.SwinTransformer):
                    child_wd_params, child_nowd_params = swin_v2_b_get_params(child)
                else:
                    child_wd_params, child_nowd_params = child.get_params()
                wd_params += child_wd_params
                nowd_params += child_nowd_params
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params


class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_chan,
        out_chan,
        ks=3,
        stride=1,
        padding=1,
        norm_layer=None,
        activation="leaky_relu",
        *args,
        **kwargs
    ):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            in_chan,
            out_chan,
            kernel_size=ks,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.norm_layer = norm_layer
        if self.norm_layer is not None:
            self.bn = norm_layer(out_chan, activation=activation)
        else:
            self.bn = lambda x: x

        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)


class FPNOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, norm_layer=None, *args, **kwargs):
        super(FPNOutput, self).__init__()
        self.norm_layer = norm_layer
        self.conv = ConvBNReLU(
            in_chan, mid_chan, ks=3, stride=1, padding=1, norm_layer=norm_layer
        )
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, self.norm_layer):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class LAFeatureFusionModule(nn.Module):
    def __init__(
        self, in_chan, mid_chn=256, out_chan=128, norm_layer=None, *args, **kwargs
    ):
        super(LAFeatureFusionModule, self).__init__()
        self.norm_layer = norm_layer
        self._up_kwargs = up_kwargs
        mid_chn = int(in_chan / 2)
        self.w_qs = ConvBNReLU(
            in_chan,
            32,
            ks=1,
            stride=1,
            padding=0,
            norm_layer=norm_layer,
            activation="none",
        )

        self.w_ks = ConvBNReLU(
            in_chan,
            32,
            ks=1,
            stride=1,
            padding=0,
            norm_layer=norm_layer,
            activation="none",
        )

        self.w_vs = ConvBNReLU(
            in_chan, in_chan, ks=1, stride=1, padding=0, norm_layer=norm_layer
        )

        self.latlayer3 = ConvBNReLU(
            in_chan, in_chan, ks=1, stride=1, padding=0, norm_layer=norm_layer
        )

        self.up = ConvBNReLU(
            in_chan, mid_chn, ks=1, stride=1, padding=1, norm_layer=norm_layer
        )
        self.smooth = ConvBNReLU(
            in_chan, out_chan, ks=3, stride=1, padding=1, norm_layer=norm_layer
        )

        self.init_weight()

    def forward(self, feat, up_fea_in, up_flag, smf_flag):

        query = self.w_qs(feat)
        key = self.w_ks(feat)
        value = self.w_vs(feat)

        N, C, H, W = feat.size()

        query_ = query.view(N, 32, -1).permute(0, 2, 1)
        query = F.normalize(query_, p=2, dim=2, eps=1e-12)

        key_ = key.view(N, 32, -1)
        key = F.normalize(key_, p=2, dim=1, eps=1e-12)

        value = value.view(N, C, -1).permute(0, 2, 1)

        f = torch.matmul(key, value)
        y = torch.matmul(query, f)

        y = y.permute(0, 2, 1).contiguous()

        y = y.view(N, C, H, W)
        W_y = self.latlayer3(y)
        p_feat = W_y + feat

        if up_flag and smf_flag:
            if up_fea_in is not None:
                p_feat = self._upsample_add(up_fea_in, p_feat)
            up_feat = self.up(p_feat)
            smooth_feat = self.smooth(p_feat)
            return up_feat, smooth_feat

        if up_flag and not smf_flag:
            if up_fea_in is not None:
                p_feat = self._upsample_add(up_fea_in, p_feat)
            up_feat = self.up(p_feat)
            return up_feat

        if not up_flag and smf_flag:
            if up_fea_in is not None:
                p_feat = self._upsample_add(up_fea_in, p_feat)
            smooth_feat = self.smooth(p_feat)
            return smooth_feat

    def _upsample_add(self, x, y):
        """Upsample and add two feature maps."""
        _, _, H, W = y.size()
        return F.interpolate(x, (H, W), **self._up_kwargs) + y

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, self.norm_layer):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


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


def get_model(model_dict, nclass, loss_fn=None):
    name = model_dict["arch"]
    model = _get_model_instance(name)
    param_dict = copy.deepcopy(model_dict)
    param_dict.pop("arch")
    param_dict["loss_fn"] = loss_fn
    param_dict["norm_layer"] = BatchNorm2d

    model = model(nclass=nclass, **param_dict)
    return model


def _get_model_instance(name):
    try:
        return {"fanet": FANet}[name]
    except:
        raise ("Model {} not available".format(name))
