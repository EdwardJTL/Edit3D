import argparse
import copy
import math
import numpy as np
import os

import collections
from collections import deque

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image

from generators import generators
from discriminators import discriminators
import network_config

import datasets
import curriculums
from tqdm import tqdm
from datetime import datetime
from ddp_utils import synchronize
import fid_evaluation

from torch_ema import ExponentialMovingAverage


def setup_ddp(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    return


def cleanup():
    dist.destroy_process_group()


def sampler(
        z,
        generator,
        metadata,
        output_dir,
        img_name
):
    generator.eval()
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            generated_imgs = generator(z, forward_points=256**2, **metadata)[0]
            save_image(
                generated_imgs,
                os.path.join(output_dir, f"{img_name}.png"),
                nrow=5,
                normalize=True,
            )
    return


def build_optimizer(generator_ddp, discriminator_ddp, metadata):
    if metadata["unique_lr"] == False:
        optimizer_G = torch.optim.Adam(
            generator_ddp.parameters(),
            lr=metadata["gen_lr"],
            betas=metadata["betas"],
            weight_decay=metadata["weight_decay"],
        )
    else:
        raise NotImplementedError

    optimizer_D = torch.optim.Adam(
        discriminator_ddp.parameters(),
        lr=metadata["disc_lr"],
        betas=metadata["betas"],
        weight_decay=metadata["weight_decay"],
    )

    return optimizer_G, optimizer_D


def train(rank, world_size, opt):
    torch.manual_seed(0)

    setup_ddp(rank, world_size, opt)
    device = torch.device(rank)

    curriculum = getattr(curriculums, opt.curriculum)
    metadata = curriculums.extract_metadata(curriculum, 0)

    scaler_G = torch.cuda.amp.GradScaler(enabled=metadata["use_amp_G"])
    scaler_D = torch.cuda.amp.GradScaler(enabled=metadata["use_amp_D"])

    # Initialize the model
    inr_model = getattr(network_config, metadata["inr_model"]).build_model().to(device)
    siren_model = (
        getattr(network_config, metadata["siren_model"]).build_model().to(device)
    )
    inr_mapping = (
        getattr(network_config, metadata["inr_mapping"])
        .build_model(inr_model.style_dim_dict)
        .to(device)
    )
    siren_mapping = (
        getattr(network_config, metadata["siren_mapping"])
        .build_model(siren_model.style_dim_dict)
        .to(device)
    )
    generator = getattr(generators, metadata["generator"])(
        z_dim=metadata["latent_dim"],
        siren_model=siren_model,
        inr=inr_model,
        mapping_network_nerf=siren_mapping,
        mapping_network_inr=inr_mapping,
    ).to(device)
    discriminator = (
        getattr(discriminators, metadata["discriminator"]).build_model().to(device)
    )

    # ema
    ema = ExponentialMovingAverage(generator, decay=0.999)
    ema2 = ExponentialMovingAverage(generator, decay=0.9999)

    # ddp
    generator_ddp = DDP(generator, device_ids=[rank], find_unused_parameters=True)
    discriminator_ddp = DDP(
        discriminator,
        device_ids=[rank],
        find_unused_parameters=True,
        broadcast_buffers=False,
    )
    generator = generator_ddp.module
    discriminator = discriminator_ddp.module

    # optimizer
    optimizer_G, optimizer_D = build_optimizer(
        generator_ddp, discriminator_ddp, metadata
    )

    state_dict = {
        "cur_fid": np.inf,
        "best_fid": np.inf,
        "worst_fid": 0,
        "step": 0,
    }

    model_dict = {
        "generator": generator_ddp.module,
        "discriminator": discriminator_ddp.module,
        "state_dict": state_dict,
    }

    # load checkpoint
    # todo: load checkpoint

    # training
    torch.manual_seed(rank)
    dataloader = None
    total_progress_bar = tqdm(
        total=opt.n_epochs, desc="Total progress", dynamic_ncols=True
    )
    total_progress_bar.update(discriminator.epoch)
    interior_step_bar = tqdm(dynamic_ncols=True)

    summary_ddict = collections.defaultdict(dict)

    generator_losses = []
    discriminator_losses = []

    fixed_zs = generator.get_zs(b=25)

    for _ in range(opt.n_epochs):
        total_progress_bar.update(1)

        metadata = curriculums.extract_metadata(curriculum, discriminator.step)

        # Set learning rates
        for param_group in optimizer_G.param_groups:
            if param_group.get("name", None) == "mapping_network":
                param_group["lr"] = metadata["gen_lr"] * 5e-2
            else:
                param_group["lr"] = metadata["gen_lr"]
            param_group["betas"] = metadata["betas"]
            param_group["weight_decay"] = metadata["weight_decay"]
        for param_group in optimizer_D.param_groups:
            param_group["lr"] = metadata["disc_lr"]
            param_group["betas"] = metadata["betas"]
            param_group["weight_decay"] = metadata["weight_decay"]

        if not dataloader or dataloader.batch_size != metadata["batch_size"]:
            dataloader, CHANNELS = datasets.get_dataset_distributed(
                metadata["dataset"], world_size, rank, **metadata
            )

            step_next_upsample = curriculums.next_upsample_step(
                curriculum, discriminator.step
            )
            step_last_upsample = curriculums.last_upsample_step(
                curriculum, discriminator.step
            )

            interior_step_bar.reset(total=(step_next_upsample - step_last_upsample))
            interior_step_bar.set_description(f"Progress to next stage")
            interior_step_bar.update((discriminator.step - step_last_upsample))

        for i, (imgs, _) in enumerate(dataloader):
            if discriminator.step % opt.model_save_interval == 0 and rank == 0:
                now = datetime.now()
                now = now.strftime("%d--%H:%M--")
                torch.save(
                    ema.state_dict(), os.path.join(opt.output_dir, now + "ema.pth")
                )
                torch.save(
                    ema2.state_dict(), os.path.join(opt.output_dir, now + "ema2.pth")
                )
                torch.save(
                    generator_ddp.module,
                    os.path.join(opt.output_dir, now + "generator.pth"),
                )
                torch.save(
                    discriminator_ddp.module,
                    os.path.join(opt.output_dir, now + "discriminator.pth"),
                )
                torch.save(
                    optimizer_G.state_dict(),
                    os.path.join(opt.output_dir, now + "optimizer_G.pth"),
                )
                torch.save(
                    optimizer_D.state_dict(),
                    os.path.join(opt.output_dir, now + "optimizer_D.pth"),
                )
            metadata = curriculums.extract_metadata(curriculum, discriminator.step)

            if dataloader.batch_size != metadata["batch_size"]:
                break

            if scaler_G.get_scale() < 1:
                scaler_G.update(1.0)
            if scaler_D.get_scale() < 1:
                scaler_D.update(1.0)

            generator_ddp.train()
            discriminator_ddp.train()

            alpha = min(
                1, (discriminator.step - step_last_upsample) / (metadata["fade_steps"])
            )

            real_imgs = imgs.to(device, non_blocking=True)

            metadata["nerf_noise"] = max(0, 1.0 - discriminator.step / 5000.0)

            aux_reg = metadata["train_aux_img"] and (
                discriminator.step % metadata["aux_img_interval"] == 0
            )

            # TRAIN DISCRIMINATOR
            with torch.cuda.amp.autocast():
                # Generate images for discriminator training
                with torch.no_grad():
                    zs_list = generator.get_zs(
                        real_imgs.shape[0], batch_split=metadata["batch_split"]
                    )
                    if metadata["batch_split"] == 1:
                        zs_list = [zs_list]
                    gen_imgs = []
                    gen_imgs_aux = []
                    gen_positions = []
                    gen_positions_aux = []
                    if (
                        metadata["img_size"] >= 256
                        and metadata["forward_points"] is not None
                    ):
                        forward_points = metadata["forward_points"] ** 2
                    else:
                        forward_points = None
                    for subset_z in zs_list:
                        g_imgs, g_pos = generator_ddp(
                            subset_z,
                            img_size=metadata["img_size"],
                            nerf_noise=metadata["nerf_noise"],
                            return_aux_img=aux_reg,
                            forward_points=forward_points,
                            grad_points=None,
                            fov=metadata["fov"],
                            ray_start=metadata["ray_start"],
                            ray_end=metadata["ray_end"],
                            num_steps=metadata["num_steps"],
                            h_stddev=metadata["h_stddev"],
                            v_stddev=metadata["v_stddev"],
                            hierarchical_sample=metadata["hierarchical_sample"],
                            psi=1,
                            sample_distance=metadata["z_dist"],
                        )
                        if metadata["batch_split"] > 1:
                            Gz, Gz_aux = g_imgs.chunk(metadata["batch_split"])
                            gen_imgs.append(Gz)
                            gen_imgs_aux.append(Gz_aux)
                            Gpos, Gpos_aux = g_pos.chunk(metadata["batch_split"])
                            gen_positions.append(Gpos)
                            gen_positions_aux.append(Gpos_aux)
                        else:
                            gen_imgs.append(g_imgs)
                            gen_positions.append(g_pos)
                # end of no grad
                if aux_reg:
                    real_imgs = torch.cat([real_imgs, real_imgs], dim=0)
                real_imgs.requires_grad_()
                r_preds, _, _ = discriminator_ddp(
                    real_imgs,
                    alpha=alpha,
                    use_aux_disc=aux_reg,
                    summary_ddict=summary_ddict,
                )

            d_regularize = discriminator.step & metadata["d_reg_interval"] == 0
            if metadata["r1_lambda"] > 0 and d_regularize:
                # Gradient Penalty
                grad_real = torch.autograd.grad(
                    outputs=scaler_D.scale(r_preds.sum()),
                    inputs=real_imgs,
                    create_graph=True,
                )
                inv_scale = 1.0 / scaler_D.get_scale()
                grad_real = [p * inv_scale for p in grad_real][0]

            with torch.cuda.amp.autocast(metadata["use_amp_D"]):
                if metadata["r1_lambda"] > 0 and d_regularize:
                    grad_penalty = (
                        grad_real.flatten(start_dim=1).square().sum(dim=1, keepdim=True)
                    )
                    grad_penalty = (
                        0.5
                        * metadata["r1_lambda"]
                        * grad_penalty
                        * metadata["d_reg_interval"]
                        + 0.0 * r_preds
                    )
                else:
                    grad_penalty = 0

                g_preds, _, _ = discriminator_ddp(
                    gen_imgs, alpha=alpha, use_aux_disc=aux_reg
                )

                d_loss = (
                    torch.nn.functional.softplus(g_preds)
                    + torch.nn.functional.softplus(-r_preds)
                    + grad_penalty
                ).mean()

                if rank == 0:
                    with torch.no_grad():
                        summary_ddict["D_logits"][
                            "D_logits_real"
                        ] = r_preds.mean().item()
                        summary_ddict["D_logits"][
                            "D_logits_fake"
                        ] = g_preds.mean().item()
                        summary_ddict["grad_penalty"][
                            "grad_penalty"
                        ] = grad_penalty.mean().item()

                discriminator_losses.append(d_loss.item())

            optimizer_D.zero_grad()
            scaler_D.scale(d_loss).backward()
            scaler_D.unscale_(optimizer_D)

            try:
                D_total_norm = torch.nn.utils.clip_grad_norm_(
                    discriminator_ddp.parameters(),
                    metadata["grad_clip"],
                )
                summary_ddict["D_total_norm"]["D_total_norm"] = D_total_norm.item()
            except:
                summary_ddict["D_total_norm"]["D_total_norm"] = np.nan
                optimizer_D.zero_grad()

            scaler_D.step(optimizer_D)
            scaler_D.update()

            # TRAIN GENERATOR
            zs_list = generator.get_zs(
                imgs.shape[0], batch_split=metadata["batch_split"]
            )
            if metadata["batch_split"] == 1:
                zs_list = [zs_list]

            if metadata["grad_points"] is not None:
                grad_points = metadata["grad_points"] ** 2
            else:
                grad_points = None

            for subset_z in zs_list:
                with torch.cuda.amp.autocast(metadata["use_amp_G"]):
                    gen_imgs, gen_positions = generator_ddp(
                        subset_z,
                        img_size=metadata["img_size"],
                        nerf_noise=metadata["nerf_noise"],
                        return_aux_img=aux_reg,
                        grad_points=grad_points,
                        forward_points=None,
                        fov=metadata["fov"],
                        ray_start=metadata["ray_start"],
                        ray_end=metadata["ray_end"],
                        num_steps=metadata["num_steps"],
                        h_stddev=metadata["h_stddev"],
                        v_stddev=metadata["v_stddev"],
                        hierarchical_sample=metadata["hierarchical_sample"],
                        psi=1,
                        sample_distance=metadata["z_dist"],
                    )
                    with torch.cuda.amp.autocast(metadata["use_amp_D"]):
                        g_preds, _, _ = discriminator_ddp(
                            gen_imgs.to(torch.float32),
                            alpha=alpha,
                            use_aux_disc=aux_reg,
                        )
                    g_loss = torch.nn.functional.softplus(-g_preds).mean()
                    generator_losses.append(g_loss.item())
                scaler_G.scale(g_loss).backward()
            # end accumulate gradients
            scaler_G.unscale_(optimizer_G)
            try:
                G_total_norm = torch.nn.utils.clip_grad_norm_(
                    generator_ddp.parameters(),
                    metadata["grad_clip"],
                )
                summary_ddict["G_total_norm"]["G_total_norm"] = G_total_norm.item()
            except:
                summary_ddict["G_total_norm"]["G_total_norm"] = np.nan
                optimizer_G.zero_grad()
            scaler_G.step(optimizer_G)
            scaler_G.update()
            optimizer_G.zero_grad()

            # update ema
            ema.update(generator_ddp.parameters())
            ema2.update(generator_ddp.parameters())

            if rank == 0:
                interior_step_bar.update(1)
                if i % 10 == 0:
                    tqdm.write(
                        f"[Experiment: {opt.output_dir}] [GPU: {os.environ['CUDA_VISIBLE_DEVICES']}] [Epoch: {discriminator.epoch}/{opt.n_epochs}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}] [Step: {discriminator.step}] [Alpha: {alpha:.2f}] [Img Size: {metadata['img_size']}] [Batch Size: {metadata['batch_size']}] [Scale: {scaler_G.get_scale()}, {scaler_D.get_scale()}]"
                    )

                # todo: sample image
                if discriminator.step % opt.sample_interval == 0:
                    with torch.no_grad():
                        copied_metadata = copy.deepcopy(metadata)
                        copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                        copied_metadata['img_size'] = 128
                    sampler(
                        z=fixed_zs,
                        generator=generator_ddp,
                        metadata={
                            "img_size": copied_metadata["img_size"],
                            "fov": copied_metadata["fov"],
                            "ray_start": copied_metadata["ray_start"],
                            "ray_end": copied_metadata["ray_end"],
                            "num_steps": copied_metadata["num_steps"],
                            "h_stddev": copied_metadata["h_stddev"],
                            "v_stddev": copied_metadata["v_stddev"],
                            "hierarchical_sample": copied_metadata["hierarchical_sample"],
                            "psi": 1,
                            "sample_distance": copied_metadata["z_dist"],
                        },
                        output_dir=opt.output_dir,
                        img_name=f"{discriminator.step}",
                    )

                if discriminator.step % opt.sample_interval == 0:
                    torch.save(
                        ema.state_dict(), os.path.join(opt.output_dir, "ema.pth")
                    )
                    torch.save(
                        ema2.state_dict(), os.path.join(opt.output_dir, "ema2.pth")
                    )
                    torch.save(
                        generator_ddp.module,
                        os.path.join(opt.output_dir, "generator.pth"),
                    )
                    torch.save(
                        discriminator_ddp.module,
                        os.path.join(opt.output_dir, "discriminator.pth"),
                    )
                    torch.save(
                        optimizer_G.state_dict(),
                        os.path.join(opt.output_dir, "optimizer_G.pth"),
                    )
                    torch.save(
                        optimizer_D.state_dict(),
                        os.path.join(opt.output_dir, "optimizer_D.pth"),
                    )
                    torch.save(
                        scaler_G.state_dict(),
                        os.path.join(opt.output_dir, "scaler_G.pth"),
                    )
                    torch.save(
                        scaler_D.state_dict(),
                        os.path.join(opt.output_dir, "scaler_D.pth"),
                    )
                    torch.save(
                        generator_losses,
                        os.path.join(opt.output_dir, "generator.losses"),
                    )
                    torch.save(
                        discriminator_losses,
                        os.path.join(opt.output_dir, "discriminator.losses"),
                    )

            if opt.eval_freq > 0 and (discriminator.step + 1) % opt.eval_freq == 0:
                generated_dir = os.path.join(opt.output_dir, "evaluation/generated")
                real_dir = os.path.join(opt.output_dir, "evaluation/real")

                if rank == 0:
                    fid_evaluation.setup_evaluation(
                        rank=rank,
                        dataset_name=metadata["dataset"],
                        generated_dir=generated_dir,
                        real_dir=real_dir,
                        dataset_path=metadata["dataset_path"],
                        num_imgs=8000,
                        target_size=128,
                    )
                dist.barrier()
                ema.store(generator_ddp.parameters())
                ema.copy_to(generator_ddp.parameters())
                generator_ddp.eval()
                fid_evaluation.output_images(
                    rank=rank,
                    world_size=world_size,
                    generator=generator_ddp,
                    metadata=metadata,
                    generated_dir=generated_dir,
                )
                ema.restore(generator_ddp.parameters())
                dist.barrier()
                if rank == 0:
                    fid = fid_evaluation.calculate_fid(generated_dir, real_dir)
                    with open(os.path.join(opt.output_dir, f"fid.txt"), "a") as f:
                        f.write(f"\n{discriminator.step}:{fid}")

                torch.cuda.empty_cache()

            discriminator.step += 1
            generator.step += 1

            synchronize()
        discriminator.epoch += 1
        generator.epoch += 1

    cleanup()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_epochs", type=int, default=3000, help="number of epochs of training"
    )
    parser.add_argument(
        "--sample_interval",
        type=int,
        default=200,
        help="interval between image sampling",
    )
    parser.add_argument("--output_dir", type=str, default="debug")
    parser.add_argument("--load_dir", type=str, default="")
    parser.add_argument("--curriculum", type=str, required=True)
    parser.add_argument("--eval_freq", type=int, default=5000)
    parser.add_argument("--port", type=str, default="12355")
    parser.add_argument("--set_step", type=int, default=None)
    parser.add_argument("--model_save_interval", type=int, default=5000)

    opt = parser.parse_args()
    print(opt)

    os.makedirs(opt.output_dir, exist_ok=True)

    num_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    if num_gpus > 1:
        mp.spawn(train, args=(num_gpus, opt), nprocs=num_gpus, join=True)
    else:
        train(rank=0, world_size=num_gpus, opt=opt)
