from collections import OrderedDict
import math
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from einops import rearrange

from siren.siren import FiLMLayer, frequency_init


def kaiming_leaky_init(m):
    """
    Init the mapping network of StyleGAN.
    fc -> leaky_relu -> fc -> ...
    Note the outputs of each fc, especially when the number of layers increases.

    :param m:
    :return:
    """
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')


class SkipLayer(nn.Module):
    def __init__(self, ):
        super(SkipLayer, self).__init__()

    def forward(self, x0, x1):
        out = (x0 + x1)
        return out


class LinearBlock(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 name_prefix,
                 ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.name_prefix = name_prefix

        self.net = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=out_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=out_dim, out_features=out_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.net.apply(kaiming_leaky_init)

        pass

    def forward(self,
                x,
                *args,
                **kwargs):
        out = self.net(x)
        return out


class SinAct(nn.Module):
    def __init__(self, ):
        super(SinAct, self).__init__()

    def forward(self, x):
        return torch.sin(x)


class SinStyleMod(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size=1,
            style_dim=None,
            use_style_fc=False,
            demodulate=True,
            use_group_conv=False,
            eps=1e-8,
        ):
        """

        :param in_channel:
        :param out_channel:
        :param kernel_size:
        :param style_dim: =in_channel
        :param use_style_fc:
        :param demodulate:
        """
        super().__init__()

        self.eps = eps
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.style_dim = style_dim
        self.use_style_fc = use_style_fc
        self.demodulate = demodulate
        self.use_group_conv = use_group_conv

        self.padding = kernel_size // 2

        if use_group_conv:
            self.weight = nn.Parameter(torch.randn(1, out_channel, in_channel, kernel_size, kernel_size))
            torch.nn.init.kaiming_normal_(self.weight[0], a=0.2, mode='fan_in', nonlinearity='leaky_relu')
        else:
            assert kernel_size == 1
            self.weight = nn.Parameter(torch.randn(1, in_channel, out_channel))
            torch.nn.init.kaiming_normal_(self.weight[0], a=0.2, mode='fan_in', nonlinearity='leaky_relu')

        if use_style_fc:
            self.modulation = nn.Linear(style_dim, in_channel)
            self.modulation.apply(kaiming_leaky_init)
        else:
            self.style_dim = in_channel

        self.sin = SinAct()
        self.norm = nn.LayerNorm(in_channel)

        return

    def forward_bmm(self,
                    x,
                    style,
                    weight
                    ):
        """

        :param x: (b, in_c, h, w), (b, in_c), (b, n, in_c)
        :param style: (b, in_c)
        :return:
        """
        assert x.shape[0] == style.shape[0]
        if x.dim() == 2:
            input = rearrange(x, "b c -> b 1 c")
        elif x.dim() == 3:
            input = x
        else:
            assert 0

        batch, N, in_channel = input.shape

        if self.use_style_fc:
            style = self.modulation(style)
            style = style.view(-1, in_channel, 1)
        else:
            style = rearrange(style, 'b c -> b c 1')

        # (1, in, out) * (b in 1) -> (b, in, out)
        weight = weight * (style + 1)

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([1, ]) + self.eps)  # (b, out)
            weight = weight * demod.view(batch, 1, self.out_channel)  # (b, in, out) * (b, 1, out) -> (b, in, out)
        # (b, n, in) * (b, in, out) -> (b, n, out)
        out = torch.bmm(input, weight)

        if x.dim() == 2:
            out = rearrange(out, "b 1 c -> b c")
        elif x.dim() == 3:
            pass
        return out

    def forward_group_conv(self,
                           x,
                           style):
        """

        :param x: (b, in_c, h, w), (b, in_c), (b, n, in_c)
        :param style: (b, in_c)
        :return:
        """
        assert x.shape[0] == style.shape[0]
        if x.dim() == 2:
            input = rearrange(x, "b c -> b c 1 1")
        elif x.dim() == 3:
            input = rearrange(x, "b n c -> b c n 1")
        elif x.dim() == 4:
            input = x
        else:
            assert 0

        batch, in_channel, height, width = input.shape

        if self.use_style_fc:
            style = self.modulation(style).view(-1, 1, in_channel, 1, 1)
            style = style + 1.
        else:
            style = rearrange(style, 'b c -> b 1 c 1 1')
        # (1, out, in, ks, ks) * (b, 1, in, 1, 1) -> (b, out, in, ks, ks)
        weight = self.weight * style
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + self.eps)  # (b, out)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)  # (b, out, in, ks, ks) * (b, out, 1, 1, 1)
        # (b*out, in, ks, ks)
        weight = weight.view(batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size)
        # (1, b*in, h, w)
        input = input.reshape(1, batch * in_channel, height, width)
        out = F.conv2d(input, weight, padding=self.padding, groups=batch)
        _, _, height, width = out.shape
        out = out.view(batch, self.out_channel, height, width)

        if x.dim() == 2:
            out = rearrange(out, "b c 1 1 -> b c")
        elif x.dim() == 3:
            out = rearrange(out, "b c n 1 -> b n c")

        return out

    def forward(self,
                x,
                style,
                force_bmm=False):
        """

        :param input: (b, in_c, h, w), (b, in_c), (b, n, in_c)
        :param style: (b, in_c)
        :return:
        """
        if self.use_group_conv:
            if force_bmm:
                weight = rearrange(self.weight, "1 out in 1 1 -> 1 in out")
                out = self.forward_bmm(x=x, style=style, weight=weight)
            else:
                out = self.forward_group_conv(x=x, style=style)
        else:
            out = self.forward_bmm(x=x, style=style, weight=self.weight)
        return out


class SinBlock(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 style_dim,
                 name_prefix,
                 ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.style_dim = style_dim
        self.name_prefix = name_prefix

        self.style_dim_dict = {}

        self.mod1 = SinStyleMod(in_channel=in_dim,
                                out_channel=out_dim,
                                style_dim=style_dim,
                                use_style_fc=True,
                                )
        self.style_dim_dict[f'{name_prefix}_0'] = self.mod1.style_dim
        self.act1 = nn.LeakyReLU(0.2, inplace=True)

        self.mod2 = SinStyleMod(in_channel=out_dim,
                                out_channel=out_dim,
                                style_dim=style_dim,
                                use_style_fc=True,
                                )
        self.style_dim_dict[f'{name_prefix}_1'] = self.mod2.style_dim
        self.act2 = nn.LeakyReLU(0.2, inplace=True)

        self.skip = SkipLayer()
        pass

    def forward(self,
                x,
                style_dict,
                skip=False):
        x_orig = x

        style = style_dict[f'{self.name_prefix}_0']
        x = self.mod1(x, style)
        x = self.act1(x)

        style = style_dict[f'{self.name_prefix}_1']
        x = self.mod2(x, style)
        out = self.act2(x)

        if skip and out.shape[-1] == x_orig.shape[-1]:
            out = self.skip(out, x_orig)
        return out


class INRNetwork(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        hidden_layers,
        rgb_dim=3,
        device=None,
        name_prefix="inr",
    ):
        super(INRNetwork, self).__init__()

        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.rgb_dim = rgb_dim
        self.name_prefix = name_prefix

        self.network = nn.ModuleList(
            [
                FiLMLayer(input_dim, hidden_dim),
            ]
            + [FiLMLayer(hidden_dim, hidden_dim) for _ in range(hidden_layers - 1)]
        )
        self.network.apply(frequency_init(25))

        self.to_rgb = nn.Sequential(
            nn.Linear(hidden_dim, rgb_dim),
            nn.Tanh(),
        )
        self.to_rgb.apply(frequency_init(25))

        return

    def forward(self, input, style_dict):
        """

        :param input: points xyz, (b, num_points, 3)
        :param style_dict:
        :param ray_directions: (b, num_points, 3)
        :return:
        - out: (b, num_points, 4), rgb(3) + sigma(1)
        """

        x = input

        frequencies, phase_shifts = self.get_freq_phase(
            style_dict=style_dict, name=f"{self.name_prefix}_network"
        )
        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index + 1) * self.hidden_dim
            x = layer(x, frequencies[start:end], phase_shifts[start:end])

        out = self.to_rgb(x)
        return out

    def get_freq_phase(self, style_dict, name):
        styles = style_dict[name]
        styles = rearrange(styles, "b (n d) -> b d n", n=2)
        frequencies, phase_shifts = styles.unbind(-1)
        frequencies = frequencies * 15 + 30
        return frequencies, phase_shifts


class EqualLinear(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 bias=True,
                 bias_init=0,
                 lr_mul=1.,
                 scale=None,
                 norm_weight=False,
                 ):
        """

        :param in_dim:
        :param out_dim:
        :param bias:
        :param bias_init:
        :param lr_mul: 0.01
        """
        super().__init__()

        self.lr_mul = lr_mul
        self.norm_weight = norm_weight

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        else:
            self.bias = None

        if scale is not None:
            self.scale = scale
        else:
            self.scale = (1 / math.sqrt(in_dim)) * lr_mul

        return

    def forward(self,
                input):
        """

        :param input: (b c), (b, n, c)
        :return:
        """
        if self.norm_weight:
            demod = torch.rsqrt(self.weight.pow(2).sum([1, ], keepdim=True) + 1e-8)
            weight = self.weight * demod
        else:
            weight = self.weight
        out = F.linear(input, weight * self.scale, bias=self.bias * self.lr_mul)
        return out


class ToRGB(nn.Module):
    def __init__(self,
                 in_dim,
                 dim_rgb=3,
                 use_equal_fc=False):
        super().__init__()
        self.in_dim = in_dim
        self.dim_rgb = dim_rgb

        if use_equal_fc:
            self.linear = EqualLinear(in_dim, dim_rgb, scale=1.)
        else:
            self.linear = nn.Linear(in_dim, dim_rgb)
        pass

    def forward(self,
                input,
                skip=None):

        out = self.linear(input)

        if skip is not None:
            out = out + skip
        return out


class CIPSNet(nn.Module):
    def __init__(self,
                 input_dim,
                 style_dim,
                 hidden_dim=256,
                 pre_rgb_dim=32,
                 device=None,
                 name_prefix='inr',
                 ):
        """

        :param input_dim:
        :param style_dim:
        :param hidden_dim:
        :param pre_rgb_dim:
        :param device:
        :param name_prefix:
        """
        super(CIPSNet, self).__init__()

        self.device = device
        self.pre_rgb_dim = pre_rgb_dim
        self.name_prefix = name_prefix

        self.channels = {
            "4": hidden_dim,
            "8": hidden_dim,
            "16": hidden_dim,
            "32": hidden_dim,
            "64": hidden_dim,
            "128": hidden_dim,
            "256": hidden_dim,
            "512": hidden_dim,
            "1024": hidden_dim,
        }

        self.style_dim_dict = {}

        _out_dim = input_dim
        network = OrderedDict()
        to_rbgs = OrderedDict()

        for i, (name, channel) in enumerate(self.channels.items()):
            _in_dim = _out_dim
            _out_dim = channel

            if name.startswith(('none', )):
                _linear_block = LinearBlock(
                    in_dim=_in_dim,
                    out_dim=_out_dim,
                    name_prefix=f"{self.name_prefix}_linear_block_{name}",
                )
                network[name] = _linear_block
            else:
                _film_block = SinBlock(
                    in_dim=_in_dim,
                    out_dim=_out_dim,
                    style_dim=style_dim,
                    name_prefix=f"{self.name_prefix}_sin_block_{name}",
                )
                self.style_dim_dict.update(_film_block.style_dim_dict)
                network[name] = _film_block

            _to_rgb = ToRGB(
                in_dim=_out_dim,
                dim_rgb=self.pre_rgb_dim,
                use_equal_fc=False,
            )
            to_rbgs[name] = _to_rgb

        self.network = nn.ModuleDict(network)
        self.to_rbgs = nn.ModuleDict(to_rbgs)
        self.to_rgbs.apply(frequency_init(100))

        out_layers = []
        if pre_rgb_dim > 3:
            out_layers.append(nn.Linear(pre_rgb_dim, 3))
        out_layers.append(nn.Tanh())
        self.tanh = nn.Sequential(*out_layers)
        self.tanh.apply(frequency_init(100))

    def forward(self,
                x,
                style_dict,
                img_size=1024,
                ):
        """
        :return:
        - out: (b, num_points, 4), rgb(3) + sigma(1)
        """

        img_size = str(2 ** int(np.log2(img_size)))

        rgb = 0
        for idx, (name, block) in enumerate(self.network.items()):
            skip = idx >= 4

            x = block(x, style_dict, skip=skip)

            if idx >= 3:
                rgb = self.to_rbgs[name](x, skip=rgb)

            if name == img_size:
                break

        rgb = self.tanh(rgb)
        return rgb


