import abc

from generators.generators import CIPSGeneratorNerfINR
from generators.mapping import MultiHeadMappingNetwork
from discriminators.discriminators import MultiScaleAuxDiscriminator
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
            "input_dim": 32,
            "style_dim": 512,
            "hidden_dim": 512,
            "pre_rgb_dim": 3,
        }
        super().__init__(config, CIPSNet)

    def build_model(self):
        return self.constructor(
            input_dim=self.config["input_dim"],
            style_dim=self.config["style_dim"],
            hidden_dim=self.config["hidden_dim"],
            pre_rgb_dim=self.config["pre_rgb_dim"],
        )


class ShallowSIRENConfig(NetworkConfig):
    def __init__(self):
        config = {
            "in_dim": 3,
            "hidden_dim": 128,
            "hidden_layers": 2,
            "rgb_dim": 32,
            "style_dim": 128,
        }
        super().__init__(config, ShallownSIREN)

    def build_model(self):
        return self.constructor(
            in_dim=self.config["in_dim"],
            hidden_dim=self.config["hidden_dim"],
            hidden_layers=self.config["hidden_layers"],
            style_dim=self.config["style_dim"],
            rgb_dim=self.config["rgb_dim"],
        )


class INRMultiHeadMappingConfig(NetworkConfig):
    def __init__(self):
        config = {
            "z_dim": 512,
            "hidden_dim": 512,
            "base_layers": 8,
            "head_layers": 0,
            "add_norm": True,
            "norm_out": True,
        }

        super().__init__(config, MultiHeadMappingNetwork)

    def build_model(self, inr_model):
        self.config["head_dim_dict"] = inr_model.style_dim_dict
        return self.constructor(**self.config)


class SirenMultiHeadMappingConfig(NetworkConfig):
    def __init__(self):
        config = {
            "z_dim": 256,
            "hidden_dim": 128,
            "base_layers": 2,
            "head_layers": 0,
            "add_norm": False,
            "norm_out": False,
        }

        super().__init__(config, MultiHeadMappingNetwork)

    def build_model(self, siren):
        self.config["head_dim_dict"] = siren.style_dim_dict
        return self.constructor(**self.config)


class MultiScaleAuxDiscriminatorConfig(NetworkConfig):
    def __init__(self):
        config = {
            "diffaug": False,
            "max_size": 1024,
            "channel_multiplier": 2,
            "first_downsample": False,
            "stddev_group": 0,
        }
        super().__init__(config, MultiScaleAuxDiscriminator)

    def build_model(self, inr_model):
        return self.constructor(**self.config)
