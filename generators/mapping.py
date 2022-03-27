import torch
import torch.nn as nn

from siren.siren import kaiming_leaky_init


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        assert input.dim() == 2
        return input * torch.rsqrt(torch.mean(input**2, dim=1, keepdim=True) + 1e-8)


class MultiHeadMappingNetwork(nn.Module):
    def __init__(
        self,
        z_dim,
        hidden_dim,
        base_layers,
        head_layers,
        head_dim_dict,
        add_norm=False,
        norm_out=False,
    ):
        super(MultiHeadMappingNetwork, self).__init__()

        self.z_dim = z_dim
        self.head_dim_dict = head_dim_dict

        out_dim = z_dim

        self.norm = PixelNorm()

        # base network
        base_net = []
        for i in range(base_layers):
            in_dim = out_dim
            out_dim = hidden_dim

            _base_layer = nn.Linear(in_features=in_dim, out_features=out_dim)
            _base_layer.apply(kaiming_leaky_init)
            base_net.append(_base_layer)

            if head_layers > 0 or i != base_layers - 1:
                if add_norm:
                    _norm_layer = nn.LayerNorm(out_dim)
                    base_net.append(_norm_layer)
                _act_layer = nn.LeakyReLU(0.2, inplace=True)
                base_net.append(_act_layer)

        if len(base_net) > 0:
            if norm_out and head_layers <= 0:
                _norm_layer = nn.LayerNorm(out_dim)
                base_net.append(_norm_layer)
            self.base_net = nn.Sequential(*base_net)
            self.num_z = 1
        else:
            self.base_net = None
            self.num_z = len(head_dim_dict)

        # head networks
        head_in_dim = out_dim
        for name, head_dim in head_dim_dict.items():
            if head_layers > 0:
                head_net = []
                out_dim = head_in_dim
                for i in range(head_layers):
                    in_dim = out_dim
                    if i == head_layers - 1:
                        out_dim = head_dim
                    else:
                        out_dim = hidden_dim

                    _head_layer = nn.Linear(in_features=in_dim, out_features=out_dim)
                    _head_layer.apply(kaiming_leaky_init)
                    head_net.append(_head_layer)

                    if i == head_layers - 1:
                        if norm_out:
                            _norm_layer = nn.LayerNorm(out_dim)
                            head_net.append(_norm_layer)
                    else:
                        _act_layer = nn.LeakyReLU(0.2, inplace=True)
                        head_net.append(_act_layer)
                head_net = nn.Sequential(*head_net)
            else:
                head_net = nn.Identity()
                
            self.add_module(name, head_net)

        return

    def forward(self, z):
        if self.base_net is not None:
            z = self.norm(z)
            base_feature = self.base_net(z)
            head_inputs = {name: base_feature for name in self.head_dim_dict.keys()}
        else:
            head_inputs = {}
            for idx, name in enumerate(self.head_dim_dict.keys()):
                head_inputs[name] = self.norm(z[idx])

        out_dict = {}
        for name, head_dim in self.head_dim_dict.items():
            head_net = getattr(self, name)
            head_input = head_inputs[name]
            out = head_net(head_input)
            out_dict[name] = out

        return out_dict
