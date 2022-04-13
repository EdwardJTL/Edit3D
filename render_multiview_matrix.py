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
    with torch.no_grad():
        imgs = generator(z, forward_points=256**2, return_aux_img=return_aux_img, **kwargs)[0]

        img = imgs[0:3, :, :, :]
        aux_img = imgs[3:, :, :, :]

        img_min = img.min()
        img_max = img.max()
        img = (img - img_min) / (img_max - img_min) * 256
        img = img.permute(0, 2, 3, 1).squeeze().cpu().numpy()
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

def make_matrix(
    gen, curriculum, seed, yaw, pitch, img_size, text_height=20, left_margin=0
):
    curriculum = make_curriculum(curriculum)
    curriculum["img_size"] = img_size
    torch.manual_seed(seed)
    z = gen.get_zs(
        b=1,
        dist=curriculum['z_dist'],
    )
    print("z {}".format(z.cpu()))
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
    canvas_w, canvas_h = canvas.size
    for iy, y in enumerate(yaw):
        for ip, p in enumerate(pitch):
            print("Making Image yaw {} pitch {} at ({}, {})".format(y, p, iy, ip))
            curriculum["h_mean"] = y
            curriculum["v_mean"] = p
            img, aux_img = generate_image(gen, z, **curriculum)
            PIL_image = Image.fromarray(np.uint8(img)).convert("RGB")
            # PIL_image.save("{}_{}.png".format(iy, ip))
            canvas.paste(
                PIL_image, (img_size * iy + left_margin, img_size * ip + text_height)
            )
    canvas = write_labels(canvas, yaw, pitch, img_size, text_height, left_margin)
    return canvas

def main():
    model_path = "/h/edwardl/pigan/output/5320339/DELAYEDPURGE/"
    curriculum = "CelebA"
    yaw = np.linspace(math.pi * 0.5 - 0.3, math.pi * 0.5 + 0.3, 5, endpoint=False)
    pitch = np.linspace(math.pi / 4 * 85 / 90 - 0.15, math.pi / 4 * 85 / 90 + 0.15, 5, endpoint=False)
    img_size = 64
    text_height = 20
    left_margin = 50
    seed = 0
    print("Starting Generation")
    image = make_matrix(
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
    image.save("./test.png")
    return


if __name__ == "__main__":
    main()
