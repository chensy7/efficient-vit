import math
from typing import Dict, Optional, Set, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

def make_divisible(value: int, divisor: int = 8, min_value: Optional[int] = None) -> int:
    """
    Ensure that all layers have a channel count that is divisible by `divisor`. This function is taken from the
    original TensorFlow repo. It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)

class MobileViTConvLayer(nn.Module):
    def __init__(
        self,
        # config: MobileViTConfig,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        bias: bool = False,
        dilation: int = 1,
        use_normalization: bool = True,
        use_activation: Union[bool, str] = True,
    ) -> None:
        super().__init__()
        padding = int((kernel_size - 1) / 2) * dilation

        if in_channels % groups != 0:
            raise ValueError(f"Input channels ({in_channels}) are not divisible by {groups} groups.")
        if out_channels % groups != 0:
            raise ValueError(f"Output channels ({out_channels}) are not divisible by {groups} groups.")

        self.convolution = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode="zeros",
        )

        if use_normalization:
            self.normalization = nn.BatchNorm2d(
                num_features=out_channels,
                eps=1e-5,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            )
        else:
            self.normalization = None

        if use_activation:
            self.activation = torch.nn.SiLU()
            # if isinstance(use_activation, str):
            #     self.activation = ACT2FN[use_activation]
            # elif isinstance(config.hidden_act, str):
            #     self.activation = ACT2FN[config.hidden_act]
            # else:
            #     self.activation = config.hidden_act
        else:
            self.activation = None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self.convolution(features)
        if self.normalization is not None:
            features = self.normalization(features)
        if self.activation is not None:
            features = self.activation(features)
        return features


class MobileViTInvertedResidual(nn.Module):
    """
    Inverted residual block (MobileNetv2): https://arxiv.org/abs/1801.04381
    """

    def __init__(
        self, 
        # config: MobileViTConfig, 
        in_channels: int, out_channels: int, stride: int, dilation: int = 1
    ) -> None:
        super().__init__()
        expanded_channels = make_divisible(int(round(in_channels * 4)), 8)

        if stride not in [1, 2]:
            raise ValueError(f"Invalid stride {stride}.")

        self.use_residual = (stride == 1) and (in_channels == out_channels)

        self.expand_1x1 = MobileViTConvLayer(
            # config, 
            in_channels=in_channels, out_channels=expanded_channels, kernel_size=1
        )

        self.conv_3x3 = MobileViTConvLayer(
            # config,
            in_channels=expanded_channels,
            out_channels=expanded_channels,
            kernel_size=3,
            stride=stride,
            groups=expanded_channels,
            dilation=dilation,
        )

        self.reduce_1x1 = MobileViTConvLayer(
            # config,
            in_channels=expanded_channels,
            out_channels=out_channels,
            kernel_size=1,
            use_activation=False,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = features

        features = self.expand_1x1(features)
        features = self.conv_3x3(features)
        features = self.reduce_1x1(features)

        return residual + features if self.use_residual else features


class MobileViTMobileNetLayer(nn.Module):
    def __init__(
        self, 
        # config: MobileViTConfig, 
        in_channels: int, out_channels: int, stride: int = 1, num_stages: int = 1
    ) -> None:
        super().__init__()

        self.layer = nn.ModuleList()
        for i in range(num_stages):
            layer = MobileViTInvertedResidual(
                # config,
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride if i == 0 else 1,
            )
            self.layer.append(layer)
            in_channels = out_channels

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        for layer_module in self.layer:
            features = layer_module(features)
        return features


class MobileViTSelfAttention(nn.Module):
    def __init__(self, 
        # config: MobileViTConfig, 
        hidden_size: int) -> None:
        super().__init__()

        # if hidden_size % config.num_attention_heads != 0:
        #     raise ValueError(
        #         f"The hidden size {hidden_size,} is not a multiple of the number of attention "
        #         f"heads {config.num_attention_heads}."
        #     )

        self.num_attention_heads = 4 #config.num_attention_heads
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size, bias=True)
        self.key = nn.Linear(hidden_size, self.all_head_size, bias=True)
        self.value = nn.Linear(hidden_size, self.all_head_size, bias=True)

        self.dropout = nn.Dropout(0)#config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class MobileViTSelfOutput(nn.Module):
    def __init__(self, 
        # config: MobileViTConfig, 
        hidden_size: int) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class MobileViTAttention(nn.Module):
    def __init__(self, 
        # config: MobileViTConfig, 
        hidden_size: int) -> None:
        super().__init__()
        self.attention = MobileViTSelfAttention(hidden_size)
        self.output = MobileViTSelfOutput(hidden_size)
        self.pruned_heads = set()

    def prune_heads(self, heads: Set[int]) -> None:
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self_outputs = self.attention(hidden_states)
        attention_output = self.output(self_outputs)
        return attention_output


class MobileViTIntermediate(nn.Module):
    def __init__(self, 
        # config: MobileViTConfig, 
        hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = torch.nn.SiLU()
        # if isinstance(config.hidden_act, str):
        #     self.intermediate_act_fn = ACT2FN[config.hidden_act]
        # else:
        #     self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class MobileViTOutput(nn.Module):
    def __init__(self, 
        # config: MobileViTConfig, 
        hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states


class MobileViTTransformerLayer(nn.Module):
    def __init__(self, 
        # config: MobileViTConfig, 
        hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.attention = MobileViTAttention(hidden_size)
        self.intermediate = MobileViTIntermediate(hidden_size, intermediate_size)
        self.output = MobileViTOutput(hidden_size, intermediate_size)
        self.layernorm_before = nn.LayerNorm(hidden_size, eps=1e-05)
        self.layernorm_after = nn.LayerNorm(hidden_size, eps=1e-05)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        attention_output = self.attention(self.layernorm_before(hidden_states))
        hidden_states = attention_output + hidden_states

        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)
        layer_output = self.output(layer_output, hidden_states)
        return layer_output


class MobileViTTransformer(nn.Module):
    def __init__(self, 
        # config: MobileViTConfig, 
        hidden_size: int, num_stages: int) -> None:
        super().__init__()

        self.layer = nn.ModuleList()
        for _ in range(num_stages):
            transformer_layer = MobileViTTransformerLayer(
                # config,
                hidden_size=hidden_size,
                intermediate_size=int(hidden_size * 2.0),
            )
            self.layer.append(transformer_layer)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states)
        return hidden_states


class MobileViTLayer(nn.Module):
    """
    MobileViT block: https://arxiv.org/abs/2110.02178
    """

    def __init__(
        self,
        # config: MobileViTConfig,
        in_channels: int,
        out_channels: int,
        stride: int,
        hidden_size: int,
        num_stages: int,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.patch_width = 2#config.patch_size
        self.patch_height = 2#config.patch_size

        if stride == 2:
            self.downsampling_layer = MobileViTInvertedResidual(
                # config,
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride if dilation == 1 else 1,
                dilation=dilation // 2 if dilation > 1 else 1,
            )
            in_channels = out_channels
        else:
            self.downsampling_layer = None

        self.conv_kxk = MobileViTConvLayer(
            # config,
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,#config.conv_kernel_size,
        )

        self.conv_1x1 = MobileViTConvLayer(
            # config,
            in_channels=in_channels,
            out_channels=hidden_size,
            kernel_size=1,
            use_normalization=False,
            use_activation=False,
        )

        self.transformer = MobileViTTransformer(
            # config,
            hidden_size=hidden_size,
            num_stages=num_stages,
        )

        self.layernorm = nn.LayerNorm(hidden_size, eps=1e-05)

        self.conv_projection = MobileViTConvLayer(
            # config, 
            in_channels=hidden_size, out_channels=in_channels, kernel_size=1
        )

        self.fusion = MobileViTConvLayer(
            # config, 
            in_channels=2 * in_channels, out_channels=in_channels, kernel_size=3
        )

    def unfolding(self, features: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        patch_width, patch_height = self.patch_width, self.patch_height
        patch_area = int(patch_width * patch_height)

        batch_size, channels, orig_height, orig_width = features.shape

        new_height = int(math.ceil(orig_height / patch_height) * patch_height)
        new_width = int(math.ceil(orig_width / patch_width) * patch_width)

        interpolate = False
        if new_width != orig_width or new_height != orig_height:
            # Note: Padding can be done, but then it needs to be handled in attention function.
            features = nn.functional.interpolate(
                features, size=(new_height, new_width), mode="bilinear", align_corners=False
            )
            interpolate = True

        # number of patches along width and height
        num_patch_width = new_width // patch_width
        num_patch_height = new_height // patch_height
        num_patches = num_patch_height * num_patch_width

        # convert from shape (batch_size, channels, orig_height, orig_width)
        # to the shape (batch_size * patch_area, num_patches, channels)
        patches = features.reshape(
            batch_size * channels * num_patch_height, patch_height, num_patch_width, patch_width
        )
        patches = patches.transpose(1, 2)
        patches = patches.reshape(batch_size, channels, num_patches, patch_area)
        patches = patches.transpose(1, 3)
        patches = patches.reshape(batch_size * patch_area, num_patches, -1)

        info_dict = {
            "orig_size": (orig_height, orig_width),
            "batch_size": batch_size,
            "channels": channels,
            "interpolate": interpolate,
            "num_patches": num_patches,
            "num_patches_width": num_patch_width,
            "num_patches_height": num_patch_height,
        }
        return patches, info_dict

    def folding(self, patches: torch.Tensor, info_dict: Dict) -> torch.Tensor:
        patch_width, patch_height = self.patch_width, self.patch_height
        patch_area = int(patch_width * patch_height)

        batch_size = info_dict["batch_size"]
        channels = info_dict["channels"]
        num_patches = info_dict["num_patches"]
        num_patch_height = info_dict["num_patches_height"]
        num_patch_width = info_dict["num_patches_width"]

        # convert from shape (batch_size * patch_area, num_patches, channels)
        # back to shape (batch_size, channels, orig_height, orig_width)
        features = patches.contiguous().view(batch_size, patch_area, num_patches, -1)
        features = features.transpose(1, 3)
        features = features.reshape(
            batch_size * channels * num_patch_height, num_patch_width, patch_height, patch_width
        )
        features = features.transpose(1, 2)
        features = features.reshape(
            batch_size, channels, num_patch_height * patch_height, num_patch_width * patch_width
        )

        if info_dict["interpolate"]:
            features = nn.functional.interpolate(
                features, size=info_dict["orig_size"], mode="bilinear", align_corners=False
            )

        return features

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # reduce spatial dimensions if needed
        if self.downsampling_layer:
            features = self.downsampling_layer(features)

        residual = features

        # local representation
        features = self.conv_kxk(features)
        features = self.conv_1x1(features)

        # convert feature map to patches
        patches, info_dict = self.unfolding(features)

        # learn global representations
        patches = self.transformer(patches)
        patches = self.layernorm(patches)

        # convert patches back to feature maps
        features = self.folding(patches, info_dict)

        features = self.conv_projection(features)
        features = self.fusion(torch.cat((residual, features), dim=1))
        return features


class MobileViTEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # self.config = config

        self.layer = nn.ModuleList()
        self.gradient_checkpointing = False

        # segmentation architectures like DeepLab and PSPNet modify the strides
        # of the classification backbones
        dilate_layer_4 = dilate_layer_5 = False
        # if config.output_stride == 8:
        #     dilate_layer_4 = True
        #     dilate_layer_5 = True
        # elif config.output_stride == 16:
        #     dilate_layer_5 = True

        dilation = 1
        neck_hidden_sizes = [
            16,
            32,
            64,
            96,
            128,
            160,
            640
        ]

        layer_1 = MobileViTMobileNetLayer(
            # config,
            in_channels=neck_hidden_sizes[0],
            out_channels=neck_hidden_sizes[1],
            stride=1,
            num_stages=1,
        )
        self.layer.append(layer_1)

        layer_2 = MobileViTMobileNetLayer(
            # config,
            in_channels=neck_hidden_sizes[1],
            out_channels=neck_hidden_sizes[2],
            stride=2,
            num_stages=3,
        )
        self.layer.append(layer_2)

        layer_3 = MobileViTLayer(
            # config,
            in_channels=neck_hidden_sizes[2],
            out_channels=neck_hidden_sizes[3],
            stride=2,
            hidden_size=144,
            num_stages=2,
        )
        self.layer.append(layer_3)

        if dilate_layer_4:
            dilation *= 2

        layer_4 = MobileViTLayer(
            # config,
            in_channels=neck_hidden_sizes[3],
            out_channels=neck_hidden_sizes[4],
            stride=2,
            hidden_size=192,
            num_stages=4,
            dilation=dilation,
        )
        self.layer.append(layer_4)

        if dilate_layer_5:
            dilation *= 2

        layer_5 = MobileViTLayer(
            # config,
            in_channels=neck_hidden_sizes[4],
            out_channels=neck_hidden_sizes[5],
            stride=2,
            hidden_size=240,
            num_stages=3,
            dilation=dilation,
        )
        self.layer.append(layer_5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        all_hidden_states = () if output_hidden_states else None

        for i, layer_module in enumerate(self.layer):
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                )
            else:
                hidden_states = layer_module(hidden_states)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)

        return BaseModelOutputWithNoAttention(last_hidden_state=hidden_states, hidden_states=all_hidden_states)

class MobileViTModel(nn.Module):
    def __init__(self, expand_output: bool = True):
        super().__init__()
        # self.config = config
        self.expand_output = expand_output

        neck_hidden_sizes = [
            16,
            32,
            64,
            96,
            128,
            160,
            640
        ]

        self.conv_stem = MobileViTConvLayer(
            # config,
            in_channels=3,
            out_channels=neck_hidden_sizes[0],
            kernel_size=3,
            stride=2,
        )

        self.encoder = MobileViTEncoder()

        if self.expand_output:
            self.conv_1x1_exp = MobileViTConvLayer(
                # config,
                in_channels=neck_hidden_sizes[5],
                out_channels=neck_hidden_sizes[6],
                kernel_size=1,
            )

        # Initialize weights and apply final processing
        # self.post_init()

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base class PreTrainedModel
        """
        for layer_index, heads in heads_to_prune.items():
            mobilevit_layer = self.encoder.layer[layer_index]
            if isinstance(mobilevit_layer, MobileViTLayer):
                for transformer_layer in mobilevit_layer.transformer.layer:
                    transformer_layer.attention.prune_heads(heads)

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # output_hidden_states = (
        #     output_hidden_states #if output_hidden_states is not None else self.config.output_hidden_states
        # )
        # return_dict = return_dict # if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        embedding_output = self.conv_stem(pixel_values)

        encoder_outputs = self.encoder(
            embedding_output,
            # output_hidden_states=output_hidden_states,
            # return_dict=return_dict,
        )

        if self.expand_output:
            last_hidden_state = self.conv_1x1_exp(encoder_outputs[0])

            # global average pooling: (batch_size, channels, height, width) -> (batch_size, channels)
            pooled_output = torch.mean(last_hidden_state, dim=[-2, -1], keepdim=False)
        else:
            last_hidden_state = encoder_outputs[0]
            pooled_output = None

        if not return_dict:
            output = (last_hidden_state, pooled_output) if pooled_output is not None else (last_hidden_state,)
            return output + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )