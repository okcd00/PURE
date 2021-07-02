# coding: utf-8
# ==========================================================================
#   Copyright (C) 2016-2021 All rights reserved.
#
#   filename : feature_fusion.py
#   author   : chendian / okcd00@qq.com
#   date     : 2021-07-02
#   desc     :
# ==========================================================================
import torch
from torch import nn
import torch.nn.functional as F
from allennlp.modules import FeedForward


class FeatureFusion(object):
    def __init__(self, method='none',
                 input_dim=768, head_hidden_dim=150):
        super().__init__()
        self.method = self.method_mapping(method)

        if method in ['none', 'concat']:
            self.fusion = self.concat_fusion
        elif 'weighted' in method:
            self.mlp = nn.Sequential(
                FeedForward(input_dim=input_dim,
                            num_layers=2,
                            hidden_dims=head_hidden_dim,  # 150
                            activations=F.relu,
                            dropout=0.2),
                nn.Linear(head_hidden_dim, 1)  # to scalar
            )
            self.softmax = nn.Softmax(dim=-1)
            self.fusion = self.gated_fusion
        elif method in ['biaffine', 'bi-affine']:
            from modules.bi_affine import Biaffine
            self.bi_affine = Biaffine(n_in=input_dim, n_out=1)
            self.sigmoid = nn.Sigmoid()
            self.fusion = self.biaffine_fusion
        else:
            raise ValueError(
                "Invalid feature fusion method type: {}".format(method))

    def method_mapping(self, method):
        method = method.lower()
        method = {
            'gated': 'weighted-sum'
        }.get(method, method)
        return method

    def concat_fusion(self, feature_case):
        return torch.cat(feature_case, dim=-1)

    def biaffine_fusion(self, feature_case):
        assert len(feature_case) == 2
        feat_1, feat_2 = feature_case
        # [batch_size, input_dim] => [batch]
        coeff = self.sigmoid(self.bi_affine(feat_1, feat_2))
        # [batch_size, n_feature=2, input_dim]
        features = torch.stack(feature_case, dim=-2)
        # [batch, n_feature=2]
        weights = torch.stack([coeff, 1. - coeff], dim=-1)

        # [batch_size, input_dim]
        return torch.bmm(weights, features).squeeze(1)

    def gated_fusion(self, feature_case):
        # [batch_size, n_feature, input_dim]
        features = torch.stack(feature_case, dim=-2)
        # [batch_size, 1, n_feature]
        weights = self.softmax(self.mlp(features).transpose(-2, -1))

        if self.method in ['weighted-sum']:
            # [batch_size, input_dim]
            return torch.bmm(weights, features).squeeze(1)
        else:  # ['weighted-concat]
            # [batch_size, n_feature, input_dim]
            return weights.transpose(-2, -1) * features

    def __call__(self, feature_case):
        # a list of [batch_size, input_dim]
        return self.fusion(feature_case)


if __name__ == "__main__":
    pass
