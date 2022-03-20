import abc

from generators.generators import CIPSGeneratorNerfINR
from INR.INR import CIPSNet


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
    