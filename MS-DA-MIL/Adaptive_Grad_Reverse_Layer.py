# -*- coding: utf-8 -*-

# Gradient reversal layer for domain adversarial network

import torch
import torch.nn as nn
from torch.autograd import Function

'''
lambda: learning rate
Propagate with flat feature map in forward propagation
Propagate with negating gradients of weight in backpropagation
'''

class AdaptiveGradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambda, attention):
        ctx.lambda = lambda
        ctx.attention = attention
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        attention = ctx.attention.squeeze(0) # delete 1st dim [1,100] -> [100]
        max_attention = torch.max(attention)
        adaptive_attention = max_attention-attention
        adaptive_attention = adaptive_attention.unsqueeze(1)
        output = (grad_output.neg() * ctx.lambda)
        adaptive_output = adaptive_attention * output
        return adaptive_output, None, None
