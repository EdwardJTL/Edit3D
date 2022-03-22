import numpy as np
import math

import torch.nn as nn
import torch
import torch.nn.functional as F

from einops import rearrange

from PosEncoding.PosEncoding import PosEmbedding


class Sine(nn.Module):
    """Sine Activation Function."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(30.0 * x)


def sine_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)


def film_sine_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_film_sine_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)


def kaiming_leaky_init(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        torch.nn.init.kaiming_normal_(
            m.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu"
        )


class CustomMappingNetwork(nn.Module):
    def __init__(self, z_dim, map_hidden_dim, map_output_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(z_dim, map_hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(map_hidden_dim, map_hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(map_hidden_dim, map_hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(map_hidden_dim, map_output_dim),
        )

        self.network.apply(kaiming_leaky_init)
        with torch.no_grad():
            self.network[-1].weight *= 0.25

    def forward(self, z):
        frequencies_offsets = self.network(z)
        frequencies = frequencies_offsets[..., : frequencies_offsets.shape[-1] // 2]
        phase_shifts = frequencies_offsets[..., frequencies_offsets.shape[-1] // 2 :]

        return frequencies, phase_shifts


def frequency_init(freq):
    def init(m):
        with torch.no_grad():
            if isinstance(m, nn.Linear):
                num_input = m.weight.size(-1)
                m.weight.uniform_(
                    -np.sqrt(6 / num_input) / freq, np.sqrt(6 / num_input) / freq
                )

    return init


class LinearScale(nn.Module):
    def __init__(self, scale, bias):
        super(LinearScale, self).__init__()
        self.scale_v = scale
        self.bias_v = bias
        return

    def forward(self, x):
        out = x * self.scale_v + self.bias_v
        return out


class FiLMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim)

    def forward(self, x, freq, phase_shift):
        x = self.layer(x)
        freq = freq.unsqueeze(1).expand_as(x)
        phase_shift = phase_shift.unsqueeze(1).expand_as(x)
        return torch.sin(freq * x + phase_shift)


class StyleFiLMLayer(nn.Module):
    def __init__(
        self, in_dim, out_dim, style_dim, use_style_fc=True, which_linear=nn.Linear
    ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.style_dim = style_dim
        self.use_style_fc = use_style_fc

        self.linear = which_linear(in_dim, out_dim)
        self.linear.apply(frequency_init(25))

        self.gain_scale = LinearScale(scale=15, bias=30)

        if use_style_fc:
            self.gain_fc = which_linear(style_dim, out_dim)
            self.bias_fc = which_linear(style_dim, out_dim)
            self.gain_fc.weight.data.mul_(0.25)
            self.bias_fc.weight.data.mul_(0.25)
        else:
            self.style_dim = out_dim * 2

        return

    def forward(self, x, style):
        """

        :param x: (b, c) or (b, n, c)
        :param style: (b, c)
        :return:
        """
        if self.use_style_fc:
            gain = self.gain_scale(self.gain_fc(style))
            bias = self.bias_fc(style)
        else:
            style = rearrange(style, "b (n c) -> b n c", n=2)
            gain, bias = style.unbind(dim=1)
            gain = self.gain_scale(gain)

        if x.dim() == 3:
            gain = rearrange(gain, "b c -> b 1 c")
            bias = rearrange(bias, "b c -> b 1 c")
        elif x.dim() == 2:
            return
        else:
            raise NotImplementedError

        x = self.linear(x)
        out = torch.sin(gain * x + bias)
        return out


class TALLSIREN(nn.Module):
    """Primary SIREN  architecture used in pi-GAN generators."""

    def __init__(
        self, input_dim=2, z_dim=100, hidden_dim=256, output_dim=1, device=None
    ):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.network = nn.ModuleList(
            [
                FiLMLayer(input_dim, hidden_dim),
                FiLMLayer(hidden_dim, hidden_dim),
                FiLMLayer(hidden_dim, hidden_dim),
                FiLMLayer(hidden_dim, hidden_dim),
                FiLMLayer(hidden_dim, hidden_dim),
                FiLMLayer(hidden_dim, hidden_dim),
                FiLMLayer(hidden_dim, hidden_dim),
                FiLMLayer(hidden_dim, hidden_dim),
            ]
        )
        self.final_layer = nn.Linear(hidden_dim, 1)

        self.color_layer_sine = FiLMLayer(hidden_dim + 3, hidden_dim)
        self.color_layer_linear = nn.Sequential(nn.Linear(hidden_dim, 3), nn.Sigmoid())

        self.mapping_network = CustomMappingNetwork(
            z_dim, 256, (len(self.network) + 1) * hidden_dim * 2
        )

        self.network.apply(frequency_init(25))
        self.final_layer.apply(frequency_init(25))
        self.color_layer_sine.apply(frequency_init(25))
        self.color_layer_linear.apply(frequency_init(25))
        self.network[0].apply(first_layer_film_sine_init)

    def forward(self, input, z, ray_directions, **kwargs):
        frequencies, phase_shifts = self.mapping_network(z)
        return self.forward_with_frequencies_phase_shifts(
            input, frequencies, phase_shifts, ray_directions, **kwargs
        )

    def forward_with_frequencies_phase_shifts(
        self, input, frequencies, phase_shifts, ray_directions, **kwargs
    ):
        frequencies = frequencies * 15 + 30

        x = input

        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index + 1) * self.hidden_dim
            x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])

        sigma = self.final_layer(x)
        rbg = self.color_layer_sine(
            torch.cat([ray_directions, x], dim=-1),
            frequencies[..., -self.hidden_dim :],
            phase_shifts[..., -self.hidden_dim :],
        )
        rbg = self.color_layer_linear(rbg)

        return torch.cat([rbg, sigma], dim=-1)


class UniformBoxWarp(nn.Module):
    def __init__(self, sidelength):
        super().__init__()
        self.scale_factor = 2 / sidelength

    def forward(self, coordinates):
        return coordinates * self.scale_factor


class SPATIALSIRENBASELINE(nn.Module):
    """Same architecture as TALLSIREN but adds a UniformBoxWarp to map input points to -1, 1"""

    def __init__(
        self, input_dim=2, z_dim=100, hidden_dim=256, output_dim=1, device=None
    ):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.network = nn.ModuleList(
            [
                FiLMLayer(3, hidden_dim),
                FiLMLayer(hidden_dim, hidden_dim),
                FiLMLayer(hidden_dim, hidden_dim),
                FiLMLayer(hidden_dim, hidden_dim),
                FiLMLayer(hidden_dim, hidden_dim),
                FiLMLayer(hidden_dim, hidden_dim),
                FiLMLayer(hidden_dim, hidden_dim),
                FiLMLayer(hidden_dim, hidden_dim),
            ]
        )
        self.final_layer = nn.Linear(hidden_dim, 1)

        self.color_layer_sine = FiLMLayer(hidden_dim + 3, hidden_dim)
        self.color_layer_linear = nn.Sequential(nn.Linear(hidden_dim, 3))

        self.mapping_network = CustomMappingNetwork(
            z_dim, 256, (len(self.network) + 1) * hidden_dim * 2
        )

        self.network.apply(frequency_init(25))
        self.final_layer.apply(frequency_init(25))
        self.color_layer_sine.apply(frequency_init(25))
        self.color_layer_linear.apply(frequency_init(25))
        self.network[0].apply(first_layer_film_sine_init)

        self.gridwarper = UniformBoxWarp(
            0.24
        )  # Don't worry about this, it was added to ensure compatibility with another model. Shouldn't affect performance.

    def forward(self, input, z, ray_directions, **kwargs):
        frequencies, phase_shifts = self.mapping_network(z)
        return self.forward_with_frequencies_phase_shifts(
            input, frequencies, phase_shifts, ray_directions, **kwargs
        )

    def forward_with_frequencies_phase_shifts(
        self, input, frequencies, phase_shifts, ray_directions, **kwargs
    ):
        frequencies = frequencies * 15 + 30

        input = self.gridwarper(input)
        x = input

        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index + 1) * self.hidden_dim
            x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])

        sigma = self.final_layer(x)
        rbg = self.color_layer_sine(
            torch.cat([ray_directions, x], dim=-1),
            frequencies[..., -self.hidden_dim :],
            phase_shifts[..., -self.hidden_dim :],
        )
        rbg = torch.sigmoid(self.color_layer_linear(rbg))

        return torch.cat([rbg, sigma], dim=-1)


class ShallowSIRENWithPosEmb(nn.Module):
    def __init__(
        self,
        z_dim=128,
        hidden_dim=256,
        rgb_dim=3,
        device=None,
        name_prefix="nerf",
    ):
        super(ShallowSIRENWithPosEmb, self).__init__()

        self.device = device
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.rgb_dim = rgb_dim
        self.name_prefix = name_prefix

        self.pos_emb = PosEmbedding(max_logscale=9, N_freqs=10)
        self.dir_emb = PosEmbedding(max_logscale=3, N_freqs=4)
        dim_pos_emb = self.pos_emb.get_out_dim()
        dim_dir_emb = self.dir_emb.get_out_dim()

        self.network = nn.ModuleList(
            [
                FiLMLayer(dim_pos_emb, hidden_dim),
                FiLMLayer(hidden_dim, hidden_dim),
            ]
        )
        self.network.apply(frequency_init(25))
        self.network[0].apply(first_layer_film_sine_init)

        self.final_layer = nn.Linear(hidden_dim, 1)
        self.final_layer.apply(frequency_init(25))

        self.color_layer_sine = FiLMLayer(hidden_dim + dim_dir_emb, hidden_dim)
        self.color_layer_sine.apply(frequency_init(25))

        self.color_layer_linear = nn.Sequential(nn.Linear(hidden_dim, rgb_dim))
        self.color_layer_linear.apply(frequency_init(25))

        self.grid_warper = UniformBoxWarp(0.24)

        return

    def forward(self, input, style_dict, ray_directions, **kwargs):
        """

        :param input: points xyz, (b, num_points, 3)
        :param style_dict:
        :param ray_directions: (b, num_points, 3)
        :param kwargs:
        :return:
        - out: (b, num_points, 4), rgb(3) + sigma(1)
        """
        out = self.forward_with_frequencies_phase_shifts(
            input=input, style_dict=style_dict, ray_directions=ray_directions, **kwargs
        )

        return out

    def get_freq_phase(self, style_dict, name):
        styles = style_dict[name]
        styles = rearrange(styles, "b (n d) -> b d n", n=2)
        frequencies, phase_shifts = styles.unbind(-1)
        frequencies = frequencies * 15 + 30
        return frequencies, phase_shifts

    def forward_with_frequencies_phase_shifts(
        self, input, style_dict, ray_directions, **kwargs
    ):
        """

        :param input: (b, n, 3)
        :param style_dict:
        :param ray_directions:
        :param kwargs:
        :return:
        """

        input = self.grid_warper(input)
        pos_emb = self.pos_emb(input)

        x = pos_emb
        frequencies, phase_shifts = self.get_freq_phase(
            style_dict=style_dict, name=f"{self.name_prefix}_network"
        )
        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index + 1) * self.hidden_dim

            x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])

        sigma = self.final_layer(x)

        # rgb branch
        dir_emb = self.dir_emb(ray_directions)
        frequencies, phase_shifts = self.get_freq_phase(
            style_dict=style_dict, name=f"{self.name_prefix}_color_layer_sine"
        )
        rbg_sine = self.color_layer_sine(
            torch.cat([dir_emb, x], dim=-1), frequencies, phase_shifts
        )
        rbg = self.color_layer_linear(rbg_sine)

        out = torch.cat([rbg, sigma], dim=-1)
        return out

    def staged_forward(
        self,
        transformed_points,
        transformed_ray_directions_expanded,
        style_dict,
        max_points,
        num_steps,
    ):
        batch_size, num_points, _ = transformed_points.shape

        rgb_sigma_output = torch.zeros(
            (batch_size, num_steps, self.rgb_dim + 1), device=self.device
        )

        for b in range(batch_size):
            head = 0
            while head < num_points:
                tail = head + max_points
                rgb_sigma_output[b : b + 1, head:tail] = self(
                    input=transformed_points[b : b + 1, head:tail],  # (b, h x w x s, 3)
                    style_dict={
                        name: style[b : b + 1] for name, style in style_dict.items()
                    },
                    ray_directions=transformed_ray_directions_expanded[
                        b : b + 1, head:tail
                    ],
                )
                head = tail

        rgb_sigma_output = rearrange(
            rgb_sigma_output, "b (hw s) rgb_sigma -> b hw s rgb_sigma", s=num_steps
        )
        return rgb_sigma_output


class ShallownSIREN(nn.Module):
    def __init__(
        self,
        in_dim=3,
        hidden_dim=256,
        hidden_layers=2,
        style_dim=512,
        rgb_dim=3,
        device=None,
        name_prefix="nerf",
    ):
        super(ShallownSIREN, self).__init__()

        self.device = device
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.rgb_dim = rgb_dim
        self.style_dim = style_dim
        self.hidden_layers = hidden_layers
        self.name_prefix = name_prefix

        self.style_dim_dict = {}

        self.network = nn.ModuleList()

        _out_dim = in_dim
        for idx in range(hidden_layers):
            _in_dim = _out_dim
            _out_dim = hidden_dim

            _layer = StyleFiLMLayer(
                in_dim=_in_dim, out_dim=_out_dim, style_dim=style_dim, use_style_fc=True
            )

            self.network.append(_layer)
            self.style_dim_dict[f"{self.name_prefix}_network_{idx}"] = _layer.style_dim

        self.final_layer = nn.Linear(hidden_dim, 1)

        _in_dim = hidden_dim
        _out_dim = hidden_dim // 2
        self.color_layer_sine = StyleFiLMLayer(
            in_dim=_in_dim, out_dim=_out_dim, style_dim=style_dim, use_style_fc=True
        )
        self.style_dim_dict[f"{self.name_prefix}_rgb"] = self.color_layer_sine.style_dim

        self.color_layer_linear = nn.Sequential(
            nn.Linear(_out_dim, rgb_dim),
        )
        self.color_layer_linear.apply(kaiming_leaky_init)

        self.dim_styles = sum(self.style_dim_dict.values())

        # Don't worry about this, it was added to ensure compatibility with another model.
        # Shouldn't affect performance.
        self.grid_warper = UniformBoxWarp(0.24)

    def forward(self, x, style_dict, ray_directions):
        out = self.forward_with_frequencies_phase_shifts(
            input=input,
            style_dict=style_dict,
            ray_directions=ray_directions,
        )

        return out

    def forward_with_frequencies_phase_shifts(
        self,
        x,
        style_dict,
        ray_directions,
    ):
        """

        :param input: (b, n, 3)
        :param style_dict:
        :param ray_directions:
        :param kwargs:
        :return:
        """

        x = self.grid_warper(x)

        for index, layer in enumerate(self.network):
            style = style_dict[f"{self.name_prefix}_network_{index}"]
            x = layer(x, style)

        sigma = self.final_layer(x)

        # rgb branch
        style = style_dict[f"{self.name_prefix}_rgb"]
        x = self.color_layer_sine(x, style)

        rbg = self.color_layer_linear(x)

        out = torch.cat([rbg, sigma], dim=-1)
        return out

    def get_freq_phase(self, style_dict, name):
        style = style_dict[name]
        style = rearrange(style, "b (n d) -> b d n", n=2)
        frequencies, phase_shifts = style.unbind(-1)
        frequencies = frequencies * 15 + 30
        return frequencies, phase_shifts

    def staged_forward(
        self,
        transformed_points,
        transformed_ray_directions_expanded,
        style_dict,
        max_points,
        num_steps,
    ):
        batch_size, num_points, _ = transformed_points.shape

        rgb_sigma_output = torch.zeros(
            (batch_size, num_points, self.rgb_dim + 1),
            device=self.device,
        )

        for b in range(batch_size):
            head = 0
            while head < num_points:
                tail = head + max_points
                rgb_sigma_output[b : b + 1, head:tail] = self(
                    input=transformed_points[b : b + 1, head:tail],
                    style_dict={
                        name: style[b : b + 1] for name, style in style_dict.items()
                    },
                    ray_directions=transformed_ray_directions_expanded[
                        b : b + 1, head:tail
                    ],
                )
                head += max_points

        rgb_sigma_output = rearrange(
            rgb_sigma_output, "b (hw s) rgb_sigma -> b hw s rgb_sigma", s=num_steps
        )
        return rgb_sigma_output


def sample_from_3dgrid(coordinates, grid):
    """
    Expects coordinates in shape (batch_size, num_points_per_batch, 3)
    Expects grid in shape (1, channels, H, W, D)
    (Also works if grid has batch size)
    Returns sampled features of shape (batch_size, num_points_per_batch, feature_channels)
    """
    coordinates = coordinates.float()
    grid = grid.float()

    batch_size, n_coords, n_dims = coordinates.shape
    sampled_features = torch.nn.functional.grid_sample(
        grid.expand(batch_size, -1, -1, -1, -1),
        coordinates.reshape(batch_size, 1, 1, -1, n_dims),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    N, C, H, W, D = sampled_features.shape
    sampled_features = sampled_features.permute(0, 4, 3, 2, 1).reshape(N, H * W * D, C)
    return sampled_features


def modified_first_sine_init(m):
    with torch.no_grad():
        # if hasattr(m, 'weight'):
        if isinstance(m, nn.Linear):
            num_input = 3
            m.weight.uniform_(-1 / num_input, 1 / num_input)


class EmbeddingPiGAN128(nn.Module):
    """Smaller architecture that has an additional cube of embeddings. Often gives better fine details."""

    def __init__(
        self, input_dim=2, z_dim=100, hidden_dim=128, output_dim=1, device=None
    ):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.network = nn.ModuleList(
            [
                FiLMLayer(32 + 3, hidden_dim),
                FiLMLayer(hidden_dim, hidden_dim),
                FiLMLayer(hidden_dim, hidden_dim),
                FiLMLayer(hidden_dim, hidden_dim),
                FiLMLayer(hidden_dim, hidden_dim),
                FiLMLayer(hidden_dim, hidden_dim),
                FiLMLayer(hidden_dim, hidden_dim),
                FiLMLayer(hidden_dim, hidden_dim),
            ]
        )
        print(self.network)

        self.final_layer = nn.Linear(hidden_dim, 1)

        self.color_layer_sine = FiLMLayer(hidden_dim + 3, hidden_dim)
        self.color_layer_linear = nn.Sequential(nn.Linear(hidden_dim, 3))

        self.mapping_network = CustomMappingNetwork(
            z_dim, 256, (len(self.network) + 1) * hidden_dim * 2
        )

        self.network.apply(frequency_init(25))
        self.final_layer.apply(frequency_init(25))
        self.color_layer_sine.apply(frequency_init(25))
        self.color_layer_linear.apply(frequency_init(25))
        self.network[0].apply(modified_first_sine_init)

        self.spatial_embeddings = nn.Parameter(torch.randn(1, 32, 96, 96, 96) * 0.01)

        # !! Important !! Set this value to the expected side-length of your scene. e.g. for for faces, heads usually fit in
        # a box of side-length 0.24, since the camera has such a narrow FOV. For other scenes, with higher FOV, probably needs to be bigger.
        self.gridwarper = UniformBoxWarp(0.24)

    def forward(self, input, z, ray_directions, **kwargs):
        frequencies, phase_shifts = self.mapping_network(z)
        return self.forward_with_frequencies_phase_shifts(
            input, frequencies, phase_shifts, ray_directions, **kwargs
        )

    def forward_with_frequencies_phase_shifts(
        self, input, frequencies, phase_shifts, ray_directions, **kwargs
    ):
        frequencies = frequencies * 15 + 30

        input = self.gridwarper(input)
        shared_features = sample_from_3dgrid(input, self.spatial_embeddings)
        x = torch.cat([shared_features, input], -1)

        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index + 1) * self.hidden_dim
            x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])

        sigma = self.final_layer(x)
        rbg = self.color_layer_sine(
            torch.cat([ray_directions, x], dim=-1),
            frequencies[..., -self.hidden_dim :],
            phase_shifts[..., -self.hidden_dim :],
        )
        rbg = torch.sigmoid(self.color_layer_linear(rbg))

        return torch.cat([rbg, sigma], dim=-1)


class EmbeddingPiGAN256(EmbeddingPiGAN128):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, hidden_dim=256)
        self.spatial_embeddings = nn.Parameter(torch.randn(1, 32, 64, 64, 64) * 0.1)
