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
    curriculum["psi"] = 1
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
        "psi": curriculum["psi"],
        "sample_dist": curriculum["sample_dist"],
        "nerf_noise": curriculum["nerf_noise"]
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
    img_size = 128
    # model_path = "/checkpoint/edwardl/6774980/DELAYEDPURGE/"
    # curriculum = "CelebA"
    model_path = "/h/edwardl/scratch/edit3d/output/6754083/DELAYEDPURGE/"
    curriculum = "LSUN"
    curriculum = make_curriculum(curriculum)
    curriculum["img_size"] = img_size
    curriculum["h_mean"] = math.pi * 0.5 + 0.7
    curriculum["v_mean"] = math.pi / 2 - 0.5
    gen_args = make_gen_args(curriculum)
    # make z's
    generator = load_generator(model_path)
    # seeds = [0, 30, 37, 44, 58]
    seeds = [51, 68, 285, 4363, 1996, 314233, 314418, 314344, 314381]
    z_s = []
    for seed in seeds:
        torch.manual_seed(seed)
        z_s.append(generator.get_zs(b=1, dist=curriculum['z_dist']))

    canvas = Image.new(
        # channels
        "RGBA",
        (
            # width
            img_size * len(seeds),
            # height
            img_size * len(seeds)
        ),
        # fill color
        (255, 255, 255, 255),
    )
    canvas_aux = Image.new(
        # channels
        "RGBA",
        (
            # width
            img_size * len(seeds),
            # height
            img_size * len(seeds)
        ),
        # fill color
        (255, 255, 255, 255),
    )

    for i, z_a in enumerate(z_s):
        for j, z_b in enumerate(z_s):
            print("i {} {}; j {} {}".format(i, np.linalg.norm(z_a["z_nerf"].cpu()), j, np.linalg.norm(z_b["z_inr"].cpu())))
            z = {
                "z_nerf": z_a["z_nerf"],
                # "z_inr": torch.zeros(z_b["z_inr"].shape, device=device),
                "z_inr": z_b["z_inr"]
            }
            img, aux_img = generate_image(generator, z, **gen_args)

            PIL_image = Image.fromarray(np.uint8(img)).convert("RGB")
            canvas.paste(
                PIL_image, (img_size * i, img_size * j)
            )
            PIL_image_aux = Image.fromarray(np.uint8(aux_img)).convert("RGB")
            canvas_aux.paste(
                PIL_image_aux, (img_size * i, img_size * j)
            )
    canvas.save("./test.png")
    canvas_aux.save("./test_aux.png")
    return


if __name__ == "__main__":
    main()
