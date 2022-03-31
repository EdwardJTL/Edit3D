"""
Contains code for logging approximate FID scores during training.
If you want to output ground-truth images from the training dataset, you can
run this file as a script.
"""

import os
import shutil
import torch
import copy
import argparse

from torchvision.utils import save_image
from pytorch_fid import fid_score
from tqdm import tqdm

import datasets
import curriculums
from ddp_utils import synchronize


def output_real_images(dataloader, num_imgs, real_dir):
    img_counter = 0
    batch_size = dataloader.batch_size
    dataloader = iter(dataloader)
    for i in range(num_imgs // batch_size):
        real_imgs, _ = next(dataloader)

        for img in real_imgs:
            save_image(
                img,
                os.path.join(real_dir, f"{img_counter:0>5}.jpg"),
                normalize=True,
                range=(-1, 1),
            )
            img_counter += 1


def setup_evaluation(
    rank,
    dataset_name,
    generated_dir,
    real_dir,
    dataset_path,
    target_size=128,
    num_imgs=8000,
):
    if rank == 0:
        os.makedirs(real_dir, exist_ok=True)
    synchronize()

    # if a directory exists but the images are not right, remove them
    real_dir_contents = [os.path.join(real_dir, name) for name in os.listdir(real_dir)]
    existing_files = [name for name in real_dir_contents if os.path.isfile(name)]
    if len(existing_files) != num_imgs:
        print("Removing {} files".format(len(existing_files)))
        for file in existing_files:
            os.remove(file)

        # then create the images
        dataloader, CHANNELS = datasets.get_dataset(
            dataset_name, img_size=target_size, dataset_path=dataset_path
        )
        print("outputting real images...")
        output_real_images(dataloader, num_imgs, real_dir)
        print("...done")

    if generated_dir is not None:
        os.makedirs(generated_dir, exist_ok=True)
    return real_dir


def output_images(
    rank,
    world_size,
    generator,
    metadata,
    generated_dir,
    img_size=128,
    num_imgs=2048,
):
    if rank == 0:
        os.makedirs(generated_dir, exist_ok=True)
    synchronize()

    metadata = copy.deepcopy(metadata)

    batch_size = metadata.get("batch_size_eval", metadata["batch_size"])

    batch_gpu = metadata["batch_size"] // world_size

    metadata["img_size"] = img_size
    metadata["batch_size"] = batch_gpu

    metadata["h_stddev"] = metadata.get("h_stddev_eval", metadata["h_stddev"])
    metadata["v_stddev"] = metadata.get("v_stddev_eval", metadata["v_stddev"])
    metadata["sample_dist"] = metadata.get("sample_dist_eval", metadata["sample_dist"])
    metadata["psi"] = 1.0

    generator.eval()

    if rank == 0:
        pbar = tqdm(desc=f"Generating images at {img_size}x{img_size}", total=num_imgs)
    with torch.no_grad():
        for idx_b in range((num_imgs + batch_size - 1) // batch_size):
            if rank == 0:
                pbar.update(batch_size)

            zs = generator.module.get_zs(metadata["batch_size"])
            generated_imgs = generator.module.forward(
                zs, 
                img_size=metadata["img_size"],
                nerf_noise=metadata["nerf_noise"],
                return_aux_img=False,
                grad_points=None,
                forward_points=256**2,
                fov=metadata["fov"],
                ray_start=metadata["ray_start"],
                ray_end=metadata["ray_end"],
                num_steps=metadata["num_steps"],
                h_stddev=metadata["h_stddev"],
                v_stddev=metadata["v_stddev"],
                hierarchical_sample=metadata["hierarchical_sample"],
                psi=metadata["psi"],
                sample_dist=metadata["z_dist"],
            )[0]

            for idx_i, img in enumerate(generated_imgs):
                saved_path = f"{generated_dir}/{idx_b * batch_size + idx_i * world_size + rank:0>5}.jpg"
                save_image(img, saved_path, normalize=True, value_range=(-1, 1))

    if rank == 0:
        pbar.close()
    synchronize()
    return


def calculate_fid(generated_dir, real_dir):
    fid = fid_score.calculate_fid_given_paths(
        [real_dir, generated_dir], 128, "cuda", 2048
    )
    print("calculate_fid torch empty cache")
    torch.cuda.empty_cache()

    return fid
