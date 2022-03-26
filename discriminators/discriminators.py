"""Discrimators used in pi-GAN"""
import collections
import math
import torch
import torch.nn as nn
import curriculums
import torch.nn.functional as F

from upfirdn2d.upfirdn2d import upfirdn2d
from fused_act.fused_act import FusedLeakyReLU, fused_leaky_relu

from .sgdiscriminators import *
from .diffaug import DiffAugment


class GlobalAveragePooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.mean([2, 3])


class AdapterBlock(nn.Module):
    def __init__(self, output_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, output_channels, 1, padding=0), nn.LeakyReLU(0.2)
        )

    def forward(self, input):
        return self.model(input)


def kaiming_leaky_init(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        torch.nn.init.kaiming_normal_(
            m.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu"
        )


class AddCoords(nn.Module):
    """
    Source: https://github.com/mkocabas/CoordConv-pytorch/blob/master/CoordConv.py
    """

    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat(
            [
                input_tensor,
                xx_channel.type_as(input_tensor),
                yy_channel.type_as(input_tensor),
            ],
            dim=1,
        )

        if self.with_r:
            rr = torch.sqrt(
                torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2)
                + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2)
            )
            ret = torch.cat([ret, rr], dim=1)

        return ret


class CoordConv(nn.Module):
    """
    Source: https://github.com/mkocabas/CoordConv-pytorch/blob/master/CoordConv.py
    """

    def __init__(self, in_channels, out_channels, with_r=False, **kwargs):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_size = in_channels + 2
        if with_r:
            in_size += 1
        self.conv = nn.Conv2d(in_size, out_channels, **kwargs)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)
        return ret


class ResidualCoordConvBlock(nn.Module):
    def __init__(
        self, inplanes, planes, kernel_size=3, stride=1, downsample=False, groups=1
    ):
        super().__init__()
        p = kernel_size // 2
        self.network = nn.Sequential(
            CoordConv(
                inplanes, planes, kernel_size=kernel_size, stride=stride, padding=p
            ),
            nn.LeakyReLU(0.2, inplace=True),
            CoordConv(planes, planes, kernel_size=kernel_size, padding=p),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.network.apply(kaiming_leaky_init)

        self.proj = nn.Conv2d(inplanes, planes, 1) if inplanes != planes else None
        self.downsample = downsample

    def forward(self, identity):
        y = self.network(identity)

        if self.downsample:
            y = nn.functional.avg_pool2d(y, 2)
        if self.downsample:
            identity = nn.functional.avg_pool2d(identity, 2)
        identity = identity if self.proj is None else self.proj(identity)

        y = (y + identity) / math.sqrt(2)
        return y


class ProgressiveDiscriminator(nn.Module):
    """Implement of a progressive growing discriminator with ResidualCoordConv Blocks"""

    def __init__(self, **kwargs):
        super().__init__()
        self.epoch = 0
        self.step = 0
        self.layers = nn.ModuleList(
            [
                ResidualCoordConvBlock(16, 32, downsample=True),  # 512x512 -> 256x256
                ResidualCoordConvBlock(32, 64, downsample=True),  # 256x256 -> 128x128
                ResidualCoordConvBlock(64, 128, downsample=True),  # 128x128 -> 64x64
                ResidualCoordConvBlock(128, 256, downsample=True),  # 64x64   -> 32x32
                ResidualCoordConvBlock(256, 400, downsample=True),  # 32x32   -> 16x16
                ResidualCoordConvBlock(400, 400, downsample=True),  # 16x16   -> 8x8
                ResidualCoordConvBlock(400, 400, downsample=True),  # 8x8     -> 4x4
                ResidualCoordConvBlock(400, 400, downsample=True),  # 4x4     -> 2x2
            ]
        )

        self.fromRGB = nn.ModuleList(
            [
                AdapterBlock(16),
                AdapterBlock(32),
                AdapterBlock(64),
                AdapterBlock(128),
                AdapterBlock(256),
                AdapterBlock(400),
                AdapterBlock(400),
                AdapterBlock(400),
                AdapterBlock(400),
            ]
        )
        self.final_layer = nn.Conv2d(400, 1, 2)
        self.img_size_to_layer = {
            2: 8,
            4: 7,
            8: 6,
            16: 5,
            32: 4,
            64: 3,
            128: 2,
            256: 1,
            512: 0,
        }

    def forward(self, input, alpha, instance_noise=0, **kwargs):
        start = self.img_size_to_layer[input.shape[-1]]

        x = self.fromRGB[start](input)
        for i, layer in enumerate(self.layers[start:]):
            if i == 1:
                x = alpha * x + (1 - alpha) * self.fromRGB[start + 1](
                    F.interpolate(input, scale_factor=0.5, mode="nearest")
                )
            x = layer(x)

        x = self.final_layer(x).reshape(x.shape[0], 1)

        return x


class ProgressiveEncoderDiscriminator(nn.Module):
    """
    Implement of a progressive growing discriminator with ResidualCoordConv Blocks.
    Identical to ProgressiveDiscriminator except it also predicts camera angles and latent codes.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.epoch = 0
        self.step = 0
        self.layers = nn.ModuleList(
            [
                ResidualCoordConvBlock(16, 32, downsample=True),  # 512x512 -> 256x256
                ResidualCoordConvBlock(32, 64, downsample=True),  # 256x256 -> 128x128
                ResidualCoordConvBlock(64, 128, downsample=True),  # 128x128 -> 64x64
                ResidualCoordConvBlock(128, 256, downsample=True),  # 64x64   -> 32x32
                ResidualCoordConvBlock(256, 400, downsample=True),  # 32x32   -> 16x16
                ResidualCoordConvBlock(400, 400, downsample=True),  # 16x16   -> 8x8
                ResidualCoordConvBlock(400, 400, downsample=True),  # 8x8     -> 4x4
                ResidualCoordConvBlock(400, 400, downsample=True),  # 4x4     -> 2x2
            ]
        )

        self.fromRGB = nn.ModuleList(
            [
                AdapterBlock(16),
                AdapterBlock(32),
                AdapterBlock(64),
                AdapterBlock(128),
                AdapterBlock(256),
                AdapterBlock(400),
                AdapterBlock(400),
                AdapterBlock(400),
                AdapterBlock(400),
            ]
        )
        self.final_layer = nn.Conv2d(400, 1 + 256 + 2, 2)
        self.img_size_to_layer = {
            2: 8,
            4: 7,
            8: 6,
            16: 5,
            32: 4,
            64: 3,
            128: 2,
            256: 1,
            512: 0,
        }

    def forward(self, input, alpha, instance_noise=0, **kwargs):
        if instance_noise > 0:
            input = input + torch.randn_like(input) * instance_noise

        start = self.img_size_to_layer[input.shape[-1]]
        x = self.fromRGB[start](input)
        for i, layer in enumerate(self.layers[start:]):
            if i == 1:
                x = alpha * x + (1 - alpha) * self.fromRGB[start + 1](
                    F.interpolate(input, scale_factor=0.5, mode="nearest")
                )
            x = layer(x)

        x = self.final_layer(x).reshape(x.shape[0], -1)

        prediction = x[..., 0:1]
        latent = x[..., 1:257]
        position = x[..., 257:259]

        return prediction, latent, position


class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, x):
        out = F.leaky_relu(x, negative_slope=self.negative_slope)
        return out * math.sqrt(2)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor**2)

        self.register_buffer("kernel", kernel)

        self.pad = pad

    def forward(self, x):
        out = upfirdn2d(x, self.kernel, pad=self.pad)
        return out


class EqualConv2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size**2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, x):
        out = F.conv2d(
            x,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )
        return out


class EqualConvTranspose2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(in_channel, out_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size**2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, x):
        out = F.conv_transpose2d(
            x,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out


class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        down_sample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
        upsample=False,
        padding="zero",
    ):
        layers = collections.OrderedDict()

        self.padding = 0
        stride = 2 if down_sample else 1

        if down_sample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers["down_blur"] = Blur(blur_kernel, pad=(pad0, pad1))

        if upsample:
            up_conv = EqualConvTranspose2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=0,
                stride=2,
                bias=bias and not activate,
            )
            layers["up_conv"] = up_conv

            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1
            layers["up_blur"] = Blur(blur_kernel, pad=(pad0, pad1))
        else:
            if not down_sample:
                if padding == "zero":
                    self.padding = (kernel_size - 1) // 2
                elif padding == "reflect":
                    padding = (kernel_size - 1) // 2
                    if padding > 0:
                        layers["pad"] = nn.ReflectionPad2d(padding)
                    self.padding = 0
                elif padding != "valid":
                    raise ValueError("padding must be 'zero', 'reflect' or 'valid'")

            equal_conv = EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
            layers["equal_conv"] = equal_conv

        if activate:
            if bias:
                layers["flrelu"] = FusedLeakyReLU(out_channel)
            else:
                layers["slrelu"] = ScaledLeakyReLU(0.2)

        super().__init__(layers)
        return


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        blur_kernel=[1, 3, 3, 1],
        kernel_size=3,
        down_sample=True,
        first_downsample=False,
    ):
        super().__init__()

        if first_downsample:
            self.conv1 = ConvLayer(
                in_channel, in_channel, kernel_size, down_sample=down_sample
            )
            self.conv2 = ConvLayer(in_channel, out_channel, kernel_size)
        else:
            self.conv1 = ConvLayer(in_channel, in_channel, kernel_size)
            self.conv2 = ConvLayer(
                in_channel, out_channel, kernel_size, down_sample=down_sample
            )

        self.skip = ConvLayer(
            in_channel,
            out_channel,
            1,
            down_sample=down_sample,
            activate=False,
            bias=False,
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out


class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out


class MultiScaleDiscriminator(nn.Module):
    def __init__(
        self,
        diffaug,
        max_size,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        input_size=3,
        first_downsample=False,
        channels=None,
        stddev_group=4,
    ):
        super().__init__()
        self.epoch = 0
        self.step = 0

        self.diffaug = diffaug
        self.max_size = max_size
        self.input_size = input_size
        self.stddev_group = stddev_group

        if channels is None:
            channels = {
                4: 512,
                8: 512,
                16: 512,
                32: 512,
                64: 256 * channel_multiplier,
                128: 128 * channel_multiplier,
                256: 64 * channel_multiplier,
                512: 32 * channel_multiplier,
                1024: 16 * channel_multiplier,
            }

        self.conv_in = nn.ModuleDict()
        for name, ch in channels.items():
            self.conv_in[f"{name}"] = nn.Conv2d(input_size, ch, 3, padding=1)

        self.convs = nn.ModuleDict()
        log_size = int(math.log(max_size, 2))
        in_channel = channels[max_size]
        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            self.convs[f"{2 ** i}"] = ResBlock(
                in_channel, out_channel, blur_kernel, first_downsample=first_downsample
            )
            in_channel = out_channel

        self.stddev_feat = 1

        if self.stddev_group > 1:
            self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        else:
            self.final_conv = ConvLayer(in_channel, channels[4], 3)

        self.space_linear = EqualLinear(
            channels[4] * 4 * 4, channels[4], activation="fused_lrelu"
        )

        self.out_linear = EqualLinear(channels[4], 1)

        return

    def diff_aug_img(self, img):
        img = DiffAugment(img, policy="color,translation,cutout")
        return img

    def forward(
        self,
        input,
        alpha,
        summary_ddict=None,
    ):
        if self.diffaug:
            input = self.diff_aug_img(input)

        size = input.shape[-1]
        log_size = int(math.log(size, 2))

        cur_size_out = self.conv_in[f"{2 ** log_size}"](input)
        cur_size_out = self.convs[f"{2 ** log_size}"](cur_size_out)

        if alpha < 1:
            down_input = F.interpolate(input, scale_factor=0.5, mode="bilinear")
            down_size_out = self.conv_in[f"{2 ** (log_size - 1)}"](down_input)
            out = alpha * cur_size_out + (1 - alpha) * down_size_out
        else:
            out = cur_size_out

        for i in range(log_size - 1, 2, -1):
            out = self.convs[f"{2 ** i}"](out)

        batch, channel, height, width = out.shape

        if self.stddev_group > 0:
            group = min(batch, self.stddev_group)
            # (4, 2, 1, 512//1, 4, 4)
            stddev = out.view(
                group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
            )
            # (2, 1, 512//1, 4, 4)
            stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
            # (2, 1, 1, 1)
            stddev = stddev.mean([2, 3, 4], keepdim=True).squeeze(2)
            # (8, 1, 4, 4)
            stddev = stddev.repeat(group, 1, height, width)
            # (8, 513, 4, 4)
            out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)
        out = out.view(batch, -1)

        out = self.space_linear(out)

        if summary_ddict is not None:
            with torch.no_grad():
                logits_norm = out.norm(dim=1).mean().item()
                w_norm = self.out_linear.weight.norm(dim=1).mean().item()
                summary_ddict["logits_norm"]["logits_norm"] = logits_norm
                summary_ddict["w_norm"]["w_norm"] = w_norm

        out = self.out_linear(out)

        latent, position = None, None
        return out, latent, position


class MultiScaleAuxDiscriminator(nn.Module):
    def __init__(
        self,
        diffaug,
        max_size,
        channel_multiplier=2,
        first_downsample=False,
        stddev_group=0,
    ):
        super().__init__()
        self.epoch = 0
        self.step = 0

        self.main_disc = MultiScaleDiscriminator(
            diffaug=diffaug,
            max_size=max_size,
            channel_multiplier=channel_multiplier,
            first_downsample=first_downsample,
            stddev_group=stddev_group,
        )

        # Auxiliary Discriminator
        channel_multiplier = 2
        channels = {
            4: 128 * channel_multiplier,
            8: 128 * channel_multiplier,
            16: 128 * channel_multiplier,
            32: 128 * channel_multiplier,
            64: 128 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }
        self.aux_disc = MultiScaleDiscriminator(
            diffaug=diffaug,
            max_size=max_size,
            channel_multiplier=channel_multiplier,
            first_downsample=True,
            channels=channels,
            stddev_group=stddev_group,
        )

        return

    def forward(
        self, input, use_aux_disc=False, summary_ddict=None, alpha=1.0, **kwargs
    ):
        if use_aux_disc:
            b = input.shape[0] // 2
            main_input = input[:b]
            aux_input = input[b:]
            main_out, latent, position = self.main_disc(
                main_input, alpha, summary_ddict=summary_ddict
            )
            aux_out, _, _ = self.aux_disc(aux_input, alpha)
            out = torch.cat([main_out, aux_out], dim=0)
        else:
            out, latent, position = self.main_disc(
                input, alpha, summary_ddict=summary_ddict
            )

        return out, latent, position
