import torch.nn as nn

from einops import rearrange

from siren.siren import FiLMLayer, frequency_init


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
