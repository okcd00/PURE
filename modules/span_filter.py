# coding: utf-8
# ==========================================================================
#   Copyright (C) 2016-2021 All rights reserved.
#
#   filename : span_filter.py
#   author   : chendian / okcd00@qq.com
#   date     : 2021-07-02
#   desc     :
# ==========================================================================
import torch
from torch import nn
import torch.nn.functional as F
from allennlp.modules import FeedForward


class SpanFilter(object):
    def __init__(self, method='none'):
        super().__init__()
        self.method = self.method_mapping(method)

    def method_mapping(self, method):
        method = method.lower()
        return method

    def __call__(self):
        # a list of [batch_size, input_dim]
        return None


if __name__ == "__main__":
    pass
