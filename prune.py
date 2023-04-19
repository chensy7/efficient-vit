import torch.nn.utils.prune as prune
import torch
import numpy as np

def prune_model(model, arch, percentage):
	if arch == 'resnet18':
		parameters_to_prune = (
			model.conv_stem.convolution,
			model.layer1[0].conv1,
			model.layer1[0].conv2,
			model.layer1[1].conv1,
			model.layer1[1].conv2,
			model.layer2[0].conv1,
			model.layer2[0].conv2,
			model.layer2[0].downsample[0],
			model.layer2[1].conv1,
			model.layer2[1].conv2,
			model.layer3[0].conv1,
			model.layer3[0].conv2,
			model.layer3[0].downsample[0],
			model.layer3[1].conv1,
			model.layer3[1].conv2,
			model.layer4[0].conv1,
			model.layer4[0].conv2,
			model.layer4[0].downsample[0],
			model.layer4[1].conv1,
			model.layer4[1].conv2,
		)
	elif arch == 'mobilevit':
		parameters_to_prune = (
			model.conv_stem.convolution,
			model.encoder.layer[0].layer[0].expand_1x1.convolution,
			model.encoder.layer[0].layer[0].conv_3x3.convolution,
			model.encoder.layer[0].layer[0].reduce_1x1.convolution,
			model.encoder.layer[1].layer[0].expand_1x1.convolution,
			model.encoder.layer[1].layer[0].conv_3x3.convolution,
			model.encoder.layer[1].layer[0].reduce_1x1.convolution,
			model.encoder.layer[1].layer[1].expand_1x1.convolution,
			model.encoder.layer[1].layer[1].conv_3x3.convolution,
			model.encoder.layer[1].layer[1].reduce_1x1.convolution,
			model.encoder.layer[1].layer[2].expand_1x1.convolution,
			model.encoder.layer[1].layer[2].conv_3x3.convolution,
			model.encoder.layer[1].layer[2].reduce_1x1.convolution,
			model.encoder.layer[2].downsampling_layer.expand_1x1.convolution,
			model.encoder.layer[2].downsampling_layer.conv_3x3.convolution,
			model.encoder.layer[2].downsampling_layer.reduce_1x1.convolution,
			model.encoder.layer[2].conv_kxk.convolution,
			model.encoder.layer[2].conv_1x1.convolution,
			model.encoder.layer[2].conv_projection.convolution,
			model.encoder.layer[2].fusion.convolution,
			model.encoder.layer[3].downsampling_layer.expand_1x1.convolution,
			model.encoder.layer[3].downsampling_layer.conv_3x3.convolution,
			model.encoder.layer[3].downsampling_layer.reduce_1x1.convolution,
			model.encoder.layer[3].conv_kxk.convolution,
			model.encoder.layer[3].conv_1x1.convolution,
			model.encoder.layer[3].conv_projection.convolution,
			model.encoder.layer[3].fusion.convolution,
			model.encoder.layer[4].downsampling_layer.expand_1x1.convolution,
			model.encoder.layer[4].downsampling_layer.conv_3x3.convolution,
			model.encoder.layer[4].downsampling_layer.reduce_1x1.convolution,
			model.encoder.layer[4].conv_kxk.convolution,
			model.encoder.layer[4].conv_1x1.convolution,
			model.encoder.layer[4].conv_projection.convolution,
			model.encoder.layer[4].fusion.convolution,
			model.conv_1x1_exp.convolution,
		)
	for idx, i in enumerate(parameters_to_prune):
		prune.ln_structured(
			i, 'weight', amount=percentage, dim=1, n=2)
	print('{0} model pruned...'.format(arch)) 

	sparse = 0
	count = 0
	for i in parameters_to_prune:
		sparse += float(torch.sum(i.weight == 0))
		bias_count = 0
		if i.bias is not None: bias_count = i.bias.nelement()
		count += float(i.weight.nelement()+bias_count)
	print("Size: " + str(count) + "\n")
	print("Sparsity: {:.2f}%".format(100. * sparse/ count))