# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
from easydict import EasyDict as edict

C = edict()
config = C
cfg = C

"""please config ROOT_dir and user when u first using"""
C.abs_dir = osp.dirname(osp.realpath(__file__))
C.parent_dir_of_curpath=osp.abspath(osp.realpath(__file__))
C.this_dir = C.abs_dir.split(osp.sep)[-1]
C.project_dir=os.path.abspath(osp.dirname(os.path.abspath(C.abs_dir)))
C.checkpoint=os.path.join(C.project_dir,'checkpoint')

C.motion = edict()
C.motion.input_length =50
C.motion.input_length_dct = 50
C.motion.target_length_train = 10
C.motion.target_length_eval = 25
C.motion.dim = 66

C.data_aug = True
C.use_relative_loss = True
C.incomplete2aug = True


""" Model Config"""
## Network
C.pre_dct = False
C.post_dct = False
## Motion Network mlp
dim_ = 66
C.motion_mlp = edict()
C.motion_mlp.hidden_dim = dim_
C.motion_mlp.seq_len = C.motion.input_length_dct
C.motion_mlp.num_layers = 48
C.motion_mlp.with_normalization = True
C.motion_mlp.spatial_fc_only = False
C.motion_mlp.norm_axis = 'spatial'

C.motion_mlp.distill_layer_num=8
C.node_num=22
"""Train Config"""
C.batch_size =256
C.num_workers = 0

C.cos_lr_max=1e-5
C.cos_lr_min=5e-8
C.cos_lr_total_iters=40000

C.weight_decay = 1e-4
C.model_pth = None

"""Eval Config"""
C.shift_step = 1

"""Display Config"""
C.print_every = 100
C.save_every = 100
# C.print_every = 1
# C.save_every = 1




if __name__ == '__main__':
    print(config.motion_mlp)
