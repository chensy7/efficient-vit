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
from transformers import MobileViTForSemanticSegmentation

id2label = {
    '0': 'background',
    '1': 'avalanche',
    '2': 'building_undamaged',
    '3': 'building_damaged',
    '4': 'cracks/fissure/subsidence',
    '5': 'debris/mud//rock flow',
    '6': 'fire/flare',
    '7': 'flood/water/river/sea',
    '8': 'ice_jam_flow',
    '9': 'lava_flow',
    '10': 'person',
    '11': 'pyroclastic_flow',
    '12': 'road/railway/bridge',
    '13': 'vehicle',
}

label2id = {v: k for k, v in id2label.items()}

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

class Deeplabv3(nn.Module):
    def __init__(
        self, nclass=14, backbone="mobilevit_small", norm_layer = None, loss_fn=None
    ):
        super(Deeplabv3,self).__init__()

        self.loss_fn = loss_fn
        self.nclass = nclass
        self.norm_layer = norm_layer
        self.model = MobileViTForSemanticSegmentation.from_pretrained("apple/deeplabv3-mobilevit-xx-small", id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True, num_attention_heads=2)


    def forward(self, x, lbl=None):
        output = self.model(x).logits
        if self.training:
            loss = self.loss_fn(output,lbl)
            return loss
        else:
            return output

    def get_params(self):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            if isinstance(child, (OhemCELoss2D, CrossEntropyLoss)):
                continue
            else:
                child_wd_params, child_nowd_params = mobilevit_get_params(child)
                wd_params += child_wd_params
                nowd_params += child_nowd_params
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params

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
        return {"Deeplabv3": Deeplabv3}[name]
    except:
        raise ("Model {} not available".format(name))
