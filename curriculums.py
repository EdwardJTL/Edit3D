"""
To easily reproduce experiments, and avoid passing several command line arguments, we implemented
a curriculum utility. Parameters can be set in a curriculum dictionary.

Curriculum Schema:

    Numerical keys in the curriculum specify an upsample step. When the current step matches the upsample step,
    the values in the corresponding dict be updated in the curriculum. Common curriculum values specified at upsamples:
        batch_size: Batch Size.
        num_steps: Number of samples along ray.
        img_size: Generated image resolution.
        batch_split: Integer number over which to divide batches and aggregate sequentially. (Used due to memory constraints)
        gen_lr: Generator learnig rate.
        disc_lr: Discriminator learning rate.

    fov: Camera field of view
    ray_start: Near clipping for camera rays.
    ray_end: Far clipping for camera rays.
    fade_steps: Number of steps to fade in new layer on discriminator after upsample.
    h_stddev: Stddev of camera yaw in radians.
    v_stddev: Stddev of camera pitch in radians.
    h_mean:  Mean of camera yaw in radians.
    v_mean: Mean of camera pitch in radians.
    sample_dist: Type of camera pose distribution. (gaussian | spherical_uniform | uniform)
    topk_interval: Interval over which to fade the top k ratio.
    topk_v: Minimum fraction of a batch to keep during top k training.
    betas: Beta parameters for Adam.
    unique_lr: Whether to use reduced LRs for mapping network.
    weight_decay: Weight decay parameter.
    r1_lambda: R1 regularization parameter.
    latent_dim: Latent dim for Siren network  in generator.
    grad_clip: Grad clipping parameter.
    model: Siren architecture used in generator. (SPATIALSIRENBASELINE | TALLSIREN)
    generator: Generator class. (ImplicitGenerator3d)
    discriminator: Discriminator class. (ProgressiveEncoderDiscriminator | ProgressiveDiscriminator)
    dataset: Training dataset. (CelebA | Carla | Cats)
    clamp_mode: Clamping function for Siren density output. (relu | softplus)
    z_dist: Latent vector distributiion. (gaussian | uniform)
    hierarchical_sample: Flag to enable hierarchical_sampling from NeRF algorithm. (Doubles the number of sampled points)
    z_labmda: Weight for experimental latent code positional consistency loss.
    pos_lambda: Weight parameter for experimental positional consistency loss.
    last_back: Flag to fill in background color with last sampled color on ray.
"""

import math


def next_upsample_step(curriculum, current_step):
    # Return the epoch when it will next upsample
    current_metadata = extract_metadata(curriculum, current_step)
    current_size = current_metadata["img_size"]
    for curriculum_step in sorted([cs for cs in curriculum.keys() if type(cs) == int]):
        if (
            curriculum_step > current_step
            and curriculum[curriculum_step].get("img_size", 512) > current_size
        ):
            return curriculum_step
    return float("Inf")


def last_upsample_step(curriculum, current_step):
    # Returns the start epoch of the current stage, i.e. the epoch
    # it last upsampled
    current_metadata = extract_metadata(curriculum, current_step)
    current_size = current_metadata["img_size"]
    for curriculum_step in sorted([cs for cs in curriculum.keys() if type(cs) == int]):
        if (
            curriculum_step <= current_step
            and curriculum[curriculum_step]["img_size"] == current_size
        ):
            return curriculum_step
    return 0


def get_current_step(curriculum, epoch):
    step = 0
    for update_epoch in curriculum["update_epochs"]:
        if epoch >= update_epoch:
            step += 1
    return step


def extract_metadata(curriculum, current_step):
    return_dict = {}
    for curriculum_step in sorted(
        [cs for cs in curriculum.keys() if type(cs) == int], reverse=True
    ):
        if curriculum_step <= current_step:
            for key, value in curriculum[curriculum_step].items():
                return_dict[key] = value
            break
    for key in [k for k in curriculum.keys() if type(k) != int]:
        return_dict[key] = curriculum[key]
    return return_dict


LSUN = {
    0: {
        "batch_size": 20,
        "num_steps": 12,
        "img_size": 32,
        "batch_split": 1,
        "gen_lr": 0.0002,
        "disc_lr": 0.002,
    },
    int(150e3): {
        "batch_size": 10,
        "num_steps": 12,
        "img_size": 64,
        "batch_split": 1,
        "gen_lr": 5e-5,
        "disc_lr": 5e-4,
    },
    int(400e3): {
        "batch_size": 1,
        "num_steps": 48,
        "img_size": 128,
        "batch_split": 1,
        "gen_lr": 10e-6,
        "disc_lr": 10e-5,
    },
    int(600e3): {},
    # int(55e3): {'batch_size': 1, 'num_steps': 48, 'img_size': 128, 'batch_split': 5, 'gen_lr': 10e-6, 'disc_lr': 10e-5},
    # int(200e3): {},
    "dataset_path": "/h/edwardl/datasets/LSUN/cars/combined/*.webp",
    # "dataset_path": "/h/edwardl/datasets/carla/images/*.png",
    "fov": 30,
    "ray_start": 0.7,
    "ray_end": 1.3,
    "fade_steps": 10000,
    "sample_dist": "spherical_uniform",
    "h_stddev": math.pi,
    "v_stddev": math.pi / 4 * 85 / 90,
    "h_mean": math.pi * 0.5,
    "v_mean": math.pi / 4 * 85 / 90,
    "topk_interval": 1000,
    "topk_v": 1,
    "betas": (0, 0.9),
    "unique_lr": False,
    "weight_decay": 0,
    "r1_lambda": 10,
    "latent_dim": 256,
    "grad_clip": 10,
    "generator": "CIPSGeneratorNerfINR",
    "discriminator": "MultiScaleAuxDiscriminatorConfig",
    "INR": "CIPSNetINRConfig",
    "siren": "ShallowSIRENConfig",
    "inr_mapping": "INRMultiHeadMappingConfig",
    "siren_mapping": "SirenMultiHeadMappingConfig",
    "dataset": "LSUNCars",
    # "dataset": "Carla",
    "white_back": True,
    "clamp_mode": "relu",
    "z_dist": "gaussian",
    "hierarchical_sample": True,
    "z_lambda": 0,
    "pos_lambda": 0,
    "learnable_dist": False,
    "use_amp_G": False,
    "use_amp_D": False,
    "forward_points": 256,
    "train_aux_img": True,
    "aux_img_interval": 1,
    "d_reg_interval": 1,
    "batch_size_eval": 16,
    "grad_points": 256,
}

CelebA = {
    0: {
        "batch_size": 4,
        "num_steps": 12,
        "img_size": 32,
        "batch_split": 1,
        "gen_lr": 0.0002,
        "disc_lr": 0.002,
    },
    int(50e3): {
        "batch_size": 10,
        "num_steps": 12,
        "img_size": 64,
        "batch_split": 1,
        "gen_lr": 5e-5,
        "disc_lr": 5e-4,
    },
    int(200e3): {},
    "dataset_path": "/h/edwardl/datasets/LSUN/cars/combined/*.webp",
    "fov": 12,
    "ray_start": 0.88,
    "ray_end": 1.12,
    "fade_steps": 10000,
    "sample_dist": "gaussian",
    "h_stddev": 0.3,
    "v_stddev": 0.155,
    "h_mean": math.pi * 0.5,
    "v_mean": math.pi / 4 * 85 / 90,
    "topk_interval": 1000,
    "topk_v": 1,
    "betas": (0, 0.9),
    "unique_lr": False,
    "weight_decay": 0,
    "r1_lambda": 10,
    "latent_dim": 256,
    "grad_clip": 10,
    "generator": "CIPSGeneratorNerfINR",
    "discriminator": "MultiScaleAuxDiscriminatorConfig",
    "INR": "CIPSNetINRConfig",
    "siren": "ShallowSIRENConfig",
    "inr_mapping": "INRMultiHeadMappingConfig",
    "siren_mapping": "SirenMultiHeadMappingConfig",
    "dataset": "CelebA",
    "white_back": True,
    "clamp_mode": "relu",
    "z_dist": "gaussian",
    "hierarchical_sample": True,
    "z_lambda": 0,
    "pos_lambda": 0,
    "learnable_dist": False,
    "use_amp_G": False,
    "use_amp_D": False,
    "forward_points": 256,
    "train_aux_img": True,
    "aux_img_interval": 1,
    "d_reg_interval": 1,
    "batch_size_eval": 16,
    "grad_points": 256,
}
