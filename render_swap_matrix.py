import math
import numpy as np
import os

import torch
from torch_ema import ExponentialMovingAverage
from torchvision.utils import save_image

from PIL import Image, ImageDraw, ImageFont

import curriculums
import network_config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_image(generator, z, return_aux_img=True, **kwargs):
    def transform(img):
        img_min = img.min()
        img_max = img.max()
        img = (img - img_min) / (img_max - img_min) * 256
        img = img.permute(0, 2, 3, 1).squeeze().cpu().numpy()
        return img

    with torch.no_grad():
        imgs = generator(z, forward_points=256**2, return_aux_img=return_aux_img, **kwargs)[0]

        img = imgs[0, :, :, :].unsqueeze(dim=0)
        aux_img = imgs[1, :, :, :].unsqueeze(dim=0)

        img = transform(img)
        aux_img = transform(aux_img)
    return img, aux_img


def make_curriculum(curriculum):
    curriculum = getattr(curriculums, curriculum, None)
    if curriculum is None:
        raise ValueError(f"{curriculum} is not a valid curriculum")
    curriculum["num_steps"] = curriculum[0]["num_steps"]
    curriculum["psi"] = 0.7
    curriculum["v_stddev"] = 0
    curriculum["h_stddev"] = 0
    curriculum["nerf_noise"] = 0
    curriculum = {key: value for key, value in curriculum.items() if type(key) is str}
    return curriculum


def make_gen_args(curriculum):
    gen_args = {
        "img_size": curriculum["img_size"],
        "fov": curriculum["fov"],
        "ray_start": curriculum["ray_start"],
        "ray_end": curriculum["ray_end"],
        "num_steps": curriculum["num_steps"],
        "h_mean": curriculum["h_mean"],
        "v_mean": curriculum["v_mean"],
        "h_stddev": 0,
        "v_stddev": 0,
        "hierarchical_sample": curriculum["hierarchical_sample"],
        "psi": 0.7,
        "sample_dist": curriculum["sample_dist"],
        "nerf_noise": 0
    }
    return gen_args


def load_generator(model_path):
    generator = torch.load(
        os.path.join(model_path, "generator.pth"), map_location=torch.device(device)
    )
    ema_dict = torch.load(
        os.path.join(model_path, "ema.pth"), map_location=torch.device(device)
    )
    ema = ExponentialMovingAverage(generator.parameters(), decay=0.999)
    ema.load_state_dict(ema_dict)
    ema.copy_to(generator.parameters())
    generator.set_device(device)
    generator.eval()
    return generator


def main():
    img_size = 64
    model_path = "/checkpoint/edwardl/6774980/DELAYEDPURGE/"
    curriculum = "CelebA"
    curriculum = make_curriculum(curriculum)
    curriculum["img_size"] = img_size
    yaw = curriculum["h_mean"]
    pitch = curriculum["v_mean"]
    gen_args = make_gen_args(curriculum)
    # make z's
    generator = load_generator(model_path)
    seeds = [0, 30, 37, 44, 58]
    z_s = []
    for seed in seeds:
        torch.manual_seed(seed)
        z_s.append(generator.get_zs(b=1, dist=curriculum['z_dist']))

    imgs = []
    aux_imgs = []

    for i, z_a in enumerate(z_s):
        for j, z_b in enumerate(z_s):
            print("{} {}".format(i, j))
            z = {
                "z_nerf": z_a["z_nerf"],
                "z_inr": z_b["z_inr"],
            }
            img, aux_img = generate_image(generator, z, **gen_args)
            imgs.append(img)
            aux_imgs.append(aux_img)
    save_image(torch.cat(imgs, dim=0), "imgs.png", nrows=len(seeds))
    save_image(torch.cat(aux_imgs, dim=0), "aux_imgs.png", nrows=len(seeds))
    return


if __name__ == "__main__":
    main()
