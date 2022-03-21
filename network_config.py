import abc

from generators.generators import CIPSGeneratorNerfINR
from INR.INR import CIPSNet
from siren.siren import ShallownSIREN


class NetworkConfig:
    def __init__(self, config, constructor):
        self.config = config
        self.constructor = constructor

    @abc.abstractmethod
    def build_model(self):
        """Method documentation"""
        return


class CIPSNetINRConfig(NetworkConfig):
    def __init__(self):
        config = {
            'input_dim': 32,
            'style_dim': 512,
            'hidden_dim': 512,
            'pre_rgb_dim': 3
        }
        super().__init__(config, CIPSNet)

    def build_model(self):
        return self.constructor(
            input_dim=self.config['input_dim'],
            style_dim=self.config['style_dim'],
            hidden_dim=self.config['hidden_dim'],
            pre_rgb_dim=self.config['pre_rgb_dim']
        )


class ShallowSIRENConfig(NetworkConfig):
    def __init__(self):
        config = {
            'in_dim': 3,
            'hidden_dim': 128,
            'hidden_layers': 2,
            'rgb_dim': 32,
            'style_dim': 128,
        }
        super().__init__(config, ShallownSIREN)

    def build_model(self):
        return self.constructor(
            in_dim=self.config['in_dim'],
            hidden_dim=self.config['hidden_dim'],
            hidden_layers=self.config['hidden_layers'],
            style_dim=self.config['style_dim'],
            rgb_dim=self.config['rgb_dim']
        )