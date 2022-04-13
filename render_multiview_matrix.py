import math
import numpy as np
import os

import torch
from torch_ema import ExponentialMovingAverage

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


def write_labels(canvas, yaw, pitch, img_size, text_height=20, left_margin=0):
    draw = ImageDraw.Draw(canvas)
    for iy, y in enumerate(yaw):
        draw.text((img_size * iy + left_margin, 0), f"{y:.3f}", fill=(0, 0, 0))
    for ip, p in enumerate(pitch):
        draw.text((0, img_size * ip + text_height), f"{p:.3f}", fill=(0, 0, 0))
    return canvas

def make_matrices(
    gen, curriculum, seed, yaw, pitch, img_size, text_height=20, left_margin=0
):
    curriculum = make_curriculum(curriculum)
    curriculum["img_size"] = img_size
    torch.manual_seed(seed)
    z = gen.get_zs(
        b=1,
        dist=curriculum['z_dist'],
    )
    print("z nerf {}".format(z["z_nerf"].cpu()))
    print("z inr {}".format(z["z_inr"].cpu()))
    canvas = Image.new(
        # channels
        "RGBA",
        (
            # width
            img_size * len(yaw) + left_margin,
            # height
            img_size * len(pitch) + text_height,
        ),
        # fill color
        (255, 255, 255, 255),
    )
    canvas_aux = Image.new(
        # channels
        "RGBA",
        (
            # width
            img_size * len(yaw) + left_margin,
            # height
            img_size * len(pitch) + text_height,
        ),
        # fill color
        (255, 255, 255, 255),
    )
    canvas_w, canvas_h = canvas.size
    for iy, y in enumerate(yaw):
        for ip, p in enumerate(pitch):
            print("Making Image yaw {} pitch {} at ({}, {})".format(y, p, iy, ip))
            curriculum["h_mean"] = y
            curriculum["v_mean"] = p
            gen_args = make_gen_args(curriculum)

            img, aux_img = generate_image(generator=gen, z=z, return_aux_img=True, **gen_args)
            
            PIL_image = Image.fromarray(np.uint8(img)).convert("RGB")
            canvas.paste(
                PIL_image, (img_size * iy + left_margin, img_size * ip + text_height)
            )
            PIL_image_aux = Image.fromarray(np.uint8(aux_img)).convert("RGB")
            canvas_aux.paste(
                PIL_image_aux, (img_size * iy + left_margin, img_size * ip + text_height)
            )
    canvas = write_labels(canvas, yaw, pitch, img_size, text_height, left_margin)
    canvas_aux = write_labels(canvas_aux, yaw, pitch, img_size, text_height, left_margin)
    return canvas, canvas_aux

def main():
    model_path = "/checkpoint/edwardl/6774980/DELAYEDPURGE/"
    curriculum = "CelebA"
    yaw = np.linspace(math.pi * 0.5 - 0.3, math.pi * 0.5 + 0.3, 5, endpoint=False)
    pitch_offset = math.pi / 4
    pitch_range = math.pi / 4
    pitch = np.linspace(
        math.pi / 4 * 85 / 90 - pitch_range, 
        math.pi / 4 * 85 / 90 + pitch_range, 
        5, 
        endpoint=False)
    pitch += pitch_offset
    img_size = 64
    text_height = 20
    left_margin = 50
    # seed = 23
    for seed in [0, 30, 37, 44, 58]:
        print("Starting Generation {}".format(seed))
        image, aux_image = make_matrices(
            load_generator(model_path),
            curriculum,
            seed,
            yaw,
            pitch,
            img_size,
            text_height,
            left_margin,
        )
        print("Saving Image")
        image.save("./test{}.png".format(seed))
        aux_image.save("./test_aux{}.png".format(seed))
    return


if __name__ == "__main__":
    main()
