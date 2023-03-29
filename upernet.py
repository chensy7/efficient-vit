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

class UperNet(nn.Module):
    def __init__(
        self, nclass=21, backbone="resnet18", loss_fn=None, fpn_out=128
    ):
        super(UperNet, self).__init__()

        self.loss_fn = loss_fn
        self.nclass = nclass

        feature_channels = [128, 256, 512, 1024]

        self.backbone_type = backbone
        if backbone == "swin_v2_b":
            self.backbone = models.swin_v2_b(weights='DEFAULT')
        else:
            raise RuntimeError("unknown backbone: {}".format(backbone))

        self.PPN = PSPModule(feature_channels[-1])
        self.FPN = FPN_fuse(feature_channels, fpn_out=fpn_out)
        self.head = nn.Conv2d(fpn_out, nclass, kernel_size=3, padding=1)

    def forward(self, x, lbl=None):

        input_size = (x.size()[2], x.size()[3])

        features = swin_v2_b_forward(self.backbone, x)
        features[-1] = self.PPN(features[-1])
        x = self.head(self.FPN(features))

        outputs = F.interpolate(x, size=input_size, mode='bilinear')

        if self.training:
            loss = (
                self.loss_fn(outputs, lbl)
            )
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
                    child_wd_params, child_nowd_params = swin_v2_b_get_params(child)
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