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

import os
import argparse

import torch
import torch.nn as nn
import numpy as np

"""
Snippet 1
Following snippet is from tptools/train.py or tptools/test.py
"""

def parse_args():
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
                        type=float,
                        default=0.1)
    parser.add_argument('--output_file',
                        help='name of output file',
                        type=str,
                        default='pruned_model.pth')

    args = parser.parse_args()
    return args

# config_file_path = 'experiments/coco/transpose_r/exp2/exp2_step6_prune.yaml'
# args = parse_args(['--cfg', config_file_path, 'TEST.USE_GT_BBOX', 'True'])
args = parse_args()
update_config(cfg, args)

logger, final_output_dir, tb_log_dir = create_logger(cfg, args.cfg, 'prune')

logger.info(pprint.pformat(args))
logger.info(cfg)

cudnn.benchmark = cfg.CUDNN.BENCHMARK
torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

model_checkpt = torch.load(cfg.TEST.MODEL_FILE, map_location='cuda:0')
if 'nw_cfg' in model_checkpt.keys():
    nw_cfg_orig = model_checkpt['nw_cfg']
else:
    nw_cfg_orig = None

model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
    cfg, is_train=False, nw_cfg=nw_cfg_orig
)
model.init_weights(cfg.TEST.MODEL_FILE)
"""
end of snippet 1
"""
args.cuda = torch.cuda.is_available()
if args.cuda:
    model.cuda()

"""
snippet 2
following code is from the network slimming repository - from resprune.py
"""

total = 0 # calculating number of BN layer weights

prune_ignore_layers = cfg.MODEL.EXTRA.PRUNE_IGNORE_LAYERS

old_modules = list(model.named_modules())
bn_layers_sel = []
bn_layers_mask = {}
for layer_id in range(len(old_modules)):
    m = old_modules[layer_id]
    if isinstance(m[1], nn.BatchNorm2d):
        if 'downsample' in m[0].lower() or 'bn3' in m[0].lower():
            continue
        if isinstance(old_modules[layer_id-1][1], nn.ConvTranspose2d):
            continue
        if layer_id == 2:
            continue
        if layer_id in prune_ignore_layers:
            pass
        elif isinstance(old_modules[layer_id + 1][1], pruning.channel_selection.ChannelSelection):
            total += int(torch.sum(old_modules[layer_id + 1][1].indexes).cpu().detach().numpy())
            print(torch.sum(old_modules[layer_id + 1][1].indexes).cpu().detach().numpy(), end=", ")
        else:
            total += m[1].weight.data.shape[0]
            print(m[1].weight.data.shape[0], end=", ")
        bn_layers_sel.append(layer_id)
    if isinstance(m[1], pruning.channel_selection.ChannelSelection):
        bn_layers_mask[layer_id - 1] = m[1].indexes.cpu().detach().numpy()
print()
bn = torch.zeros(total)
index = 0
old_modules = list(model.modules())
for layer_id in range(len(old_modules)):
    m = old_modules[layer_id]
    if isinstance(m, nn.BatchNorm2d) and layer_id in bn_layers_sel and layer_id not in prune_ignore_layers:
        if layer_id in bn_layers_mask.keys():
            size = (int) (np.sum(bn_layers_mask[layer_id]))
            ind = np.squeeze(np.argwhere(bn_layers_mask[layer_id]))
            bn[index:(index+size)] = m.weight.data[ind].abs().clone()
        else:
            size = m.weight.data.shape[0]
            bn[index:(index+size)] = m.weight.data.abs().clone()
        index += size

y, i = torch.sort(bn)
percent_pruning = args.percent
# thre_index = int(total * args.percent)
thre_index = int(total * percent_pruning)
thre = y[thre_index]

pruned = 0
nw_cfg = []
nw_cfg_mask = []
nw_cfg_dict = {}

# for k, m in enumerate(model.modules()):
for layer_id in range(len(old_modules)):
    m = old_modules[layer_id]
    if isinstance(m, nn.BatchNorm2d) and layer_id in bn_layers_sel:
        if layer_id in prune_ignore_layers:
            if layer_id in bn_layers_mask.keys():
                ind = np.squeeze(np.argwhere(bn_layers_mask[layer_id]))
                weight_sel = m.weight.data[ind].abs().clone()
                if ind.size == 1:
                    mask = torch.ones(1)
                else:
                    mask = torch.ones(weight_sel.shape[0])
            else:
                mask = torch.ones(m.weight.data.shape)
        # if layer is followed by channel selection layer
        #
        elif layer_id in bn_layers_mask.keys():
            weight_copy = m.weight.data.abs().clone()
            mask2 = weight_copy.gt(thre).float().cuda()

            ind = np.squeeze(np.argwhere(bn_layers_mask[layer_id]))
            weight_sel = m.weight.data[ind].abs().clone()
            mask = weight_sel.gt(thre).float().cuda()

            pruned = pruned + mask.shape[0] - torch.sum(mask)

            m.weight.data.mul_(mask2)
            m.bias.data.mul_(mask2)

        else:
            weight_copy = m.weight.data.abs().clone()
            mask = weight_copy.gt(thre).float().cuda()
            pruned = pruned + mask.shape[0] - torch.sum(mask)
            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask)
        nw_cfg.append(int(torch.sum(mask)))
        nw_cfg_dict[layer_id] = nw_cfg[-1]
        nw_cfg_mask.append(mask.clone())

        print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
            format(layer_id, mask.shape[0], int(torch.sum(mask))))



pruned_ratio = pruned/total
print("pruned ratio: ", pruned_ratio)

print("nw_Cfg:")
print(nw_cfg)

# newmodel is going to be the pruned model
# newmodel = resnet(depth=args.depth, dataset=args.dataset, nw_cfg=nw_cfg)
newmodel = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
    cfg, is_train=False, nw_cfg=nw_cfg
)

if args.cuda:
    newmodel.cuda()

num_parameters = sum([param.nelement() for param in newmodel.parameters()])

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
    if isinstance(m0, nn.Conv2d):
        expected_shape = m1.weight.data.shape
        print(f"{layer_id} conv shape old {m0.weight.data.shape} new {m1.weight.data.shape}")
        if layer_id+1 in bn_layers_sel:

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

# DONE you are saving nw_cfg in the model checkpoint - so you shall also use it from here
# rather than setting it from the cfg file
torch.save({'nw_cfg': nw_cfg, 'state_dict': newmodel.state_dict()}, os.path.join(final_output_dir, args.output_file))

"""
end of snippet 2
"""