from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms



import _init_paths
import pruning.channel_selection
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.function import validate
from utils.utils import create_logger

import dataset
import models

import torch.nn.utils.prune as prune

import os
import argparse

import torch
import torch.nn as nn
import numpy as np
# from lib.pruning.channel_selection import ChannelSelection

"""
Snippet 1
Following snippet is from tptools/train.py or tptools/test.py
"""

def parse_args(args___):
    parser = argparse.ArgumentParser(description='Pruning TransPoseR network with channel_selection layers')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')
    parser.add_argument('--percent',
                        help='percentage sparsity in float between 0 and 1',
                        type=float)

    args = parser.parse_args(args___)
    return args

config_file_path = 'experiments/coco/transpose_r/TP_R_256x192_d256_h1024_enc4_mh8_chsel.yaml'
args = parse_args(['--cfg', config_file_path, 'TEST.USE_GT_BBOX', 'True'])
update_config(cfg, args)

logger, final_output_dir, tb_log_dir = create_logger(cfg, args.cfg, 'valid')

logger.info(pprint.pformat(args))
logger.info(cfg)

cudnn.benchmark = cfg.CUDNN.BENCHMARK
torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
    cfg, is_train=False
)
pretrained_weights_chkpt = cfg.MODEL.PRETRAINED_CHSEL
# model.load_state_dict(pretrained_weights_chkpt)

model.init_weights(cfg.TEST.MODEL_FILE)

"""
end of snippet 1
"""


args.cuda = torch.cuda.is_available()

# if not os.path.exists(args.save):
#     os.makedirs(args.save)

# model = resnet(depth=args.depth, dataset=args.dataset) #DONE use transpose_h model from config

if args.cuda:
    model.cuda()

# if args.model: # DONE load saved model
#     if os.path.isfile(args.model):
#         print("=> loading checkpoint '{}'".format(args.model))
#         checkpoint = torch.load(args.model)
#         args.start_epoch = checkpoint['epoch']
#         best_prec1 = checkpoint['best_prec1']
#         model.load_state_dict(checkpoint['state_dict'])
#         print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
#               .format(args.model, checkpoint['epoch'], best_prec1))
#     else:
#         print("=> no checkpoint found at '{}'".format(args.resume))


"""
snippet 2
following code is from the network slimming repository - from resprune.py
"""

total = 0 # calculating number of BN layer weights

old_modules = list(model.named_modules())
bn_layers_sel = []
for layer_id in range(len(old_modules)):
    m = old_modules[layer_id]
    # print(layer_id, m[0])
    if isinstance(m[1], nn.BatchNorm2d):
        if 'downsample' in m[0].lower() or 'bn3' in m[0].lower():
            continue
        if isinstance(old_modules[layer_id-1][1], nn.ConvTranspose2d):
            continue
        if layer_id == 2:
            continue
        total += m[1].weight.data.shape[0]
        bn_layers_sel.append(layer_id)

bn = torch.zeros(total)
index = 0
old_modules = list(model.modules())
for layer_id in range(len(old_modules)):
    m = old_modules[layer_id]
    if isinstance(m, nn.BatchNorm2d) and layer_id in bn_layers_sel:
        size = m.weight.data.shape[0]
        bn[index:(index+size)] = m.weight.data.abs().clone()
        index += size

y, i = torch.sort(bn)
percent_pruning = 0.4
# thre_index = int(total * args.percent)
thre_index = int(total * percent_pruning)
thre = y[thre_index]

pruned = 0
nw_cfg = []
nw_cfg_mask = []
nw_cfg_dict = {}

for k, m in enumerate(model.modules()):
    if isinstance(m, nn.BatchNorm2d) and k in bn_layers_sel:

        weight_copy = m.weight.data.abs().clone()
        mask = weight_copy.gt(thre).float().cuda()
        pruned = pruned + mask.shape[0] - torch.sum(mask)
        m.weight.data.mul_(mask)
        m.bias.data.mul_(mask)
        nw_cfg.append(int(torch.sum(mask)))
        nw_cfg_dict[k] = nw_cfg[-1]
        nw_cfg_mask.append(mask.clone())

        print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
            format(k, mask.shape[0], int(torch.sum(mask))))
    # elif isinstance(m, nn.MaxPool2d):
    #     nw_cfg.append('M')

pruned_ratio = pruned/total

# this is the accuracy of the trained model before pruning
# acc = test(model) #TODO attach testing module

print("nw_Cfg:")
print(nw_cfg)

# newmodel is going to be the pruned model
# newmodel = resnet(depth=args.depth, dataset=args.dataset, nw_cfg=nw_cfg)
newmodel = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
    cfg, is_train=False, nw_cfg=nw_cfg
)

nw_cfg_output_file = os.path.join(final_output_dir, "transpose_r_pruned_iter1_nw_cfg.txt")
with open(nw_cfg_output_file, "w+") as nw_cfg_output_fd:
    nw_cfg_output_fd.write(str(nw_cfg))

if args.cuda:
    newmodel.cuda()

num_parameters = sum([param.nelement() for param in newmodel.parameters()])
# savepath = os.path.join(args.save, "prune.txt")
# with open(savepath, "w") as fp:
#     fp.write("Configuration: \n"+str(nw_cfg)+"\n")
#     fp.write("Number of parameters: \n"+str(num_parameters)+"\n")
#     # fp.write("Test accuracy: \n"+str(acc))

old_modules = list(model.modules())
new_modules = list(newmodel.modules())
layer_id_in_nw_cfg = 0
start_mask = torch.ones(3)
end_mask = nw_cfg_mask[layer_id_in_nw_cfg]
conv_count = 0


for layer_id in range(len(old_modules)):
    m0 = old_modules[layer_id]
    m1 = new_modules[layer_id]
    if isinstance(m0, models.transpose_r_chsel.Bottleneck):
        print("\n\n#\t#\t#\tbottleneck")
    # if isinstance(m0.)
    if isinstance(m0, nn.Conv2d):
        expected_shape = m1.weight.data.shape
        print(f"{layer_id} conv shape old {m0.weight.data.shape} new {m1.weight.data.shape}")
        if layer_id+1 in bn_layers_sel:
            #TODO handle the following cases
            # when input channels do not need to be pruned (for first conv and first BN of the Bottleneck )

            if isinstance(old_modules[layer_id+2], pruning.channel_selection.ChannelSelection):
                m1.weight.data = m0.weight.data.clone()
                # continue
            else:
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()        #here, selecting in_channels (prev layer mask)
                w1 = w1[idx1.tolist(), :, :, :].clone()                    #here, selecting out_channels (curr layer mask)
                m1.weight.data = w1.clone()
        elif layer_id-1 in bn_layers_sel:
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()  # here, selecting in_channels (prev layer mask)
            m1.weight.data = w1.clone()
        else:
            m1.weight.data = m0.weight.data.clone()
        print(f"weight shape {m1.weight.data.shape}")
        if m1.weight.data.shape != expected_shape:
            print(f"^^^MISMATCHED SHAPE OF WEIGHTS exp {expected_shape} real {m1.weight.data.shape}")

    if isinstance(m0, nn.BatchNorm2d):
        print(f"{layer_id} BN shape old {m0.weight.shape} new {m1.weight.shape}")
        expected_shape = m1.weight.data.shape
        if layer_id in bn_layers_sel:
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1,(1,))
            if isinstance(old_modules[layer_id + 1], pruning.channel_selection.ChannelSelection):
                # If the next layer is the channel selection layer, then the current batchnorm 2d layer won't be pruned.
                m1.weight.data = m0.weight.data.clone()  # setting weights from old model layer to new model layer
                m1.bias.data = m0.bias.data.clone()  # setting also bias and other params of BN layer
                m1.running_mean = m0.running_mean.clone()
                m1.running_var = m0.running_var.clone()
                # We need to set the channel selection layer.
                m2 = new_modules[layer_id + 1]
                m2.indexes.data.zero_()
                m2.indexes.data[idx1.tolist()] = 1.0
                print(f"CHSEL - {m2.indexes.data.shape} {torch.sum(m2.indexes.data).cpu().numpy()} \n\n")
            else:
                m1.weight.data = m0.weight.data[idx1.tolist()].clone()  # applying end mask to the BN layer
                m1.bias.data = m0.bias.data[idx1.tolist()].clone()  # if there is no channel selection
                m1.running_mean = m0.running_mean[idx1.tolist()].clone()
                m1.running_var = m0.running_var[idx1.tolist()].clone()
            layer_id_in_nw_cfg += 1  # next mask in mask layers
            start_mask = end_mask.clone()  # so - start mask is the end mask of the prev layer -
            # the selected layers of the prev layer
            if layer_id_in_nw_cfg < len(nw_cfg_mask):
                end_mask = nw_cfg_mask[layer_id_in_nw_cfg]  # mask for next iteration set
                # after setting channel selection layer
                # so basically 3 things need to be pruned
            print(f"weight shape {m1.weight.data.shape}")
            if m1.weight.data.shape != expected_shape:
                print(f"^^^MISMATCHED SHAPE OF WEIGHTS exp {expected_shape} real {m1.weight.data.shape}")
        else:
            m1.weight.data = m0.weight.data.clone()  # setting weights from old model layer to new model layer
            m1.bias.data = m0.bias.data.clone()  # setting also bias and other params of BN layer
            m1.running_mean = m0.running_mean.clone()
            m1.running_var = m0.running_var.clone()

# for layer_id in range(len(old_modules)):
#     m0 = old_modules[layer_id]
#     m1 = new_modules[layer_id]
#     if isinstance(m0, nn.BatchNorm2d) and layer_id in bn_layers_sel:
#         idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
#         if idx1.size == 1:
#             idx1 = np.resize(idx1,(1,))
#
#         if isinstance(old_modules[layer_id + 1], pruning.channel_selection.ChannelSelection):
#             # If the next layer is the channel selection layer, then the current batchnorm 2d layer won't be pruned.
#             m1.weight.data = m0.weight.data.clone()
#             m1.bias.data = m0.bias.data.clone()
#             m1.running_mean = m0.running_mean.clone()
#             m1.running_var = m0.running_var.clone()
#
#             # We need to set the channel selection layer.
#             m2 = new_modules[layer_id + 1]
#             m2.indexes.data.zero_()
#             m2.indexes.data[idx1.tolist()] = 1.0
#
#             layer_id_in_nw_cfg += 1
#             start_mask = end_mask.clone()
#             if layer_id_in_nw_cfg < len(nw_cfg_mask):
#                 end_mask = nw_cfg_mask[layer_id_in_nw_cfg]
#         else:
#             m1.weight.data = m0.weight.data[idx1.tolist()].clone()
#             m1.bias.data = m0.bias.data[idx1.tolist()].clone()
#             m1.running_mean = m0.running_mean[idx1.tolist()].clone()
#             m1.running_var = m0.running_var[idx1.tolist()].clone()
#             layer_id_in_nw_cfg += 1
#             start_mask = end_mask.clone()
#             if layer_id_in_nw_cfg < len(nw_cfg_mask):  # do not change in Final FC
#                 end_mask = nw_cfg_mask[layer_id_in_nw_cfg]
#     elif isinstance(m0, nn.Conv2d) and (layer_id+1) in bn_layers_sel:
#         if isinstance(new_modules[layer_id+2], pruning.channel_selection.ChannelSelection):
#             continue
#         if conv_count == 0:
#             m1.weight.data = m0.weight.data.clone()
#             conv_count += 1
#             continue
#         if isinstance(old_modules[layer_id-1], pruning.channel_selection.ChannelSelection) or isinstance(old_modules[layer_id - 1], nn.BatchNorm2d):
#             # This covers the convolutions in the residual block.
#             # The convolutions are either after the channel selection layer or after the batch normalization layer.
#             conv_count += 1
#             idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
#             idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
#             print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
#             if idx0.size == 1:
#                 idx0 = np.resize(idx0, (1,))
#             if idx1.size == 1:
#                 idx1 = np.resize(idx1, (1,))
#             w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
#
#             # If the current convolution is not the last convolution in the residual block, then we can change the
#             # number of output channels. Currently we use `conv_count` to detect whether it is such convolution.
#             # if conv_count % 3 != 1:
#             w1 = w1[idx1.tolist(), :, :, :].clone()
#             m1.weight.data = w1.clone()
#             continue
#
#         # We need to consider the case where there are downsampling convolutions.
#         # For these convolutions, we just copy the weights.
#         m1.weight.data = m0.weight.data.clone()
#     elif isinstance(m0, nn.Linear):
#         idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
#         if idx0.size == 1:
#             idx0 = np.resize(idx0, (1,))
#
#         m1.weight.data = m0.weight.data[:, idx0].clone()
#         m1.bias.data = m0.bias.data.clone()

torch.save({'nw_cfg': nw_cfg, 'state_dict': newmodel.state_dict()}, os.path.join(final_output_dir, 'transpose_r_modelfile_pruned_iter1.pth'))

"""
end of snippet 2
"""