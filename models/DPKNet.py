import copy
import torch
from torch import nn
from einops.layers.torch import Rearrange


class LN(nn.Module):
    def __init__(self, dim, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

        self.alpha = nn.Parameter(torch.ones([1, dim, 1]), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, dim, 1]), requires_grad=True)

    def forward(self, x):
        mean = x.mean(axis=1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        y = y * self.alpha + self.beta
        return y

class LN_v2(nn.Module):
    def __init__(self, dim, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

        self.alpha = nn.Parameter(torch.ones([1, 1, dim]), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, 1, dim]), requires_grad=True)

    def forward(self, x):
        mean = x.mean(axis=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        y = y * self.alpha + self.beta
        return y


class Spatial_FC(nn.Module):
    def __init__(self, dim):
        super(Spatial_FC, self).__init__()
        self.fc = nn.Linear(dim, dim)
        self.arr0 = Rearrange('b n d -> b d n')
        self.arr1 = Rearrange('b d n -> b n d')

    def forward(self, x):
        x = self.arr0(x)
        x = self.fc(x)
        x = self.arr1(x)
        return x

class Temporal_FC(nn.Module):
    def __init__(self, dim):
        super(Temporal_FC, self).__init__()
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        x = self.fc(x)
        return x


class MotionBlock(nn.Module):

    def __init__(self, dim, seq, use_norm=True, use_spatial_fc=False, layernorm_axis='spatial',):
        super().__init__()

        if not use_spatial_fc:
            self.fc0 = Temporal_FC(seq)
        else:
            self.fc0 = Spatial_FC(dim)
        self.add_Temporal_FC=True
        if self.add_Temporal_FC:
            self.fc_temporal=Temporal_FC(seq)
            self.norm_temporal = LN_v2(seq)

        if use_norm:
            if layernorm_axis == 'spatial':
                self.norm0 = LN(dim)
            elif layernorm_axis == 'temporal':
                self.norm0 = LN_v2(seq)
            elif layernorm_axis == 'all':
                self.norm0 = nn.LayerNorm([dim, seq])
            else:
                raise NotImplementedError
        else:
            self.norm0 = nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc0.fc.weight, gain=1e-8)

        nn.init.constant_(self.fc0.fc.bias, 0)

        if self.add_Temporal_FC:
            nn.init.xavier_uniform_(self.fc_temporal.fc.weight, gain=1e-8)
            nn.init.constant_(self.fc_temporal.fc.bias, 0)


    def forward(self, x):

        y_ = self.fc_temporal(x)
        y_ = self.norm_temporal(y_)

        x_ = self.fc0(x + y_)
        x_ = self.norm0(x_)

        x = x + x_

        return x

class KnowledgeDistiller(nn.Module):

    def __init__(self, dim, seq, use_norm=True, use_spatial_fc=False, layernorm_axis='spatial',):
        super().__init__()

        if not use_spatial_fc:
            self.fc0 = Temporal_FC(seq)
        else:
            self.fc0 = Spatial_FC(dim)
        self.add_Temporal_FC=True
        if self.add_Temporal_FC:
            self.fc_temporal=Temporal_FC(seq)
            self.norm_temporal = LN_v2(seq)

        if use_norm:
            if layernorm_axis == 'spatial':
                self.norm0 = LN(dim)
            elif layernorm_axis == 'temporal':
                self.norm0 = LN_v2(seq)
            elif layernorm_axis == 'all':
                self.norm0 = nn.LayerNorm([dim, seq])
            else:
                raise NotImplementedError
        else:
            self.norm0 = nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc0.fc.weight, gain=1e-8)

        nn.init.constant_(self.fc0.fc.bias, 0)

        if self.add_Temporal_FC:
            nn.init.xavier_uniform_(self.fc_temporal.fc.weight, gain=1e-8)
            nn.init.constant_(self.fc_temporal.fc.bias, 0)


    def forward(self, x):

        x_ = self.fc0(x)
        x_ = self.norm0(x_)

        x = x + x_

        return x

class DPKNet(nn.Module):
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        super(DPKNet, self).__init__()

        self.arr0 = Rearrange('b n d -> b d n')
        self.arr1 = Rearrange('b d n -> b n d')
        self.arr_distill_branch = Rearrange('b d n -> b n d')

        self.MLPBlocks = nn.Sequential(*[
            MotionBlock(dim=self.config.motion_mlp.hidden_dim, seq=self.config.motion_mlp.seq_len,
                use_norm=self.config.motion_mlp.with_normalization,
                use_spatial_fc=self.config.motion_mlp.spatial_fc_only,
                layernorm_axis=self.config.motion_mlp.norm_axis)
            for i in range(self.config.motion_mlp.num_layers)])

        # Posterior Knowledge Distillor
        self.PosteriorKnowledgeDistiller = nn.Sequential(*[
            KnowledgeDistiller(dim=self.config.motion_mlp.hidden_dim, seq=self.config.motion_mlp.seq_len,
                                 use_norm=self.config.motion_mlp.with_normalization,
                                 use_spatial_fc=self.config.motion_mlp.spatial_fc_only,
                                 layernorm_axis=self.config.motion_mlp.norm_axis)
            for i in range(self.config.motion_mlp.distill_layer_num)])

        self.motion_fc_in = nn.Linear(self.config.motion.dim, self.config.motion.dim)
        self.motion_fc_out = nn.Linear(self.config.motion.dim, self.config.motion.dim)
        self.motion_fc_for_distill = nn.Linear(self.config.motion.dim, self.config.motion.dim)

        self.reset_parameters()


    def reset_parameters(self):
        nn.init.xavier_uniform_(self.motion_fc_in.weight, gain=1e-8)
        nn.init.constant_(self.motion_fc_in.bias, 0)
        nn.init.xavier_uniform_(self.motion_fc_out.weight, gain=1e-8)
        nn.init.constant_(self.motion_fc_out.bias, 0)
        nn.init.xavier_uniform_(self.motion_fc_for_distill.weight, gain=1e-8)
        nn.init.constant_(self.motion_fc_for_distill.bias, 0)


    def forward(self, motion_input):

        # Normal Prediction
        motion_input_feats = self.motion_fc_in(motion_input)
        motion_input_feats = self.arr0(motion_input_feats)
        motion_feats = self.MLPBlocks(motion_input_feats)
        motion_feats = self.arr1(motion_feats)
        human_motion_feats = self.motion_fc_out(motion_feats)

        # Distilling Posterior Knowledge
        DistilledPosteriorKnowledge= self.PosteriorKnowledgeDistiller(motion_input_feats)
        DistilledPosteriorKnowledge = self.arr_distill_branch(DistilledPosteriorKnowledge)
        DistilledPosteriorKnowledge = self.motion_fc_for_distill(DistilledPosteriorKnowledge)

        return human_motion_feats,DistilledPosteriorKnowledge

