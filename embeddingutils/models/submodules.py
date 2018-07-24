from inferno.extensions.layers.convolutional import ConvELU3D

import torch.nn as nn
import torch.nn.functional as F


def make_separable(conv_type):

    class SeparableConv(nn.Module):
        def __init__(self, in_channels, out_channels, **kwargs):
            #TODO handle channels
            kernel_size = kwargs.pop('kernel_size')
            super(SeparableConv, self).__init__()
            self.depthwise = conv_type(in_channels, in_channels, kernel_size=kernel_size, depthwise=True)
            self.pointwise = conv_type(in_channels, out_channels, kernel_size=1, **kwargs)

        def forward(self, x):
            x = self.depthwise(x)
            x = self.pointwise(x)
            return x

    return SeparableConv


class DepthToChannel(nn.Module):
    def forward(self, input_):
        assert len(input_.shape) == 5, \
            f'input must be 5D tensor of shape (B, C, D, H, W), but got shape {input_.shape}.'
        input_ = input_.permute((0, 2, 1, 3, 4))
        return input_.contiguous().view((-1, ) + input_.shape[-3:])


class Normalize(nn.Module):
    def __init__(self, dim=1):
        super(Normalize, self).__init__()
        self.dim=dim

    def forward(self, input_):
        return F.normalize(input_, dim=self.dim)


class ResBlock(nn.Module):
    def __init__(self, inner, pre=None, post=None):
        super(ResBlock, self).__init__()
        self.inner = inner
        self.pre = pre
        self.post = post

    def forward(self, x):
        if self.pre is not None:
            x = self.pre(x)
        x = x + self.inner(x)
        if self.post is not None:
            x = self.post(x)
        print(x.shape)
        return x


class SuperhumanSNEMIBlock(ResBlock):
    def __init__(self, f_in, f_main=None, f_out=None,
                 pre_kernel_size=(1, 3, 3), inner_kernel_size=(3, 3, 3),
                 conv_type=ConvELU3D):
        if f_main is None:
            f_main = f_in
        if f_out is None:
            f_out = f_main
        pre = conv_type(f_in, f_out, kernel_size=pre_kernel_size)
        inner = nn.Sequential(conv_type(f_out, f_main, kernel_size=inner_kernel_size),
                              conv_type(f_main, f_out, kernel_size=inner_kernel_size))
        super(SuperhumanSNEMIBlock, self).__init__(pre=pre, inner=inner)
