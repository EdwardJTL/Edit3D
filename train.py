import argparse
import os
import numpy as np
import math

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
from siren import siren
from generators import mapping
import network_config

import curriculums
from tqdm import tqdm
from datetime import datetime
import copy

from torch_ema import ExponentialMovingAverage


def setup_ddp(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    return


def cleanup():
    dist.destroy_process_group()


def train(rank,
          world_size,
          opt):
    torch.manual_seed(0)

    setup_ddp(rank, world_size, opt)
    device = torch.device(rank)

    curriculum = getattr(curriculums, opt.curriculum)
    metadata = curriculums.extract_metadata(curriculum, 0)

    scaler_G = torch.cuda.amp.GradScaler(enabled=metadata['use_amp_G'])
    scaler_D = torch.cuda.amp.GradScaler(enabled=metadata['use_amp_D'])

    # Initialize the model
    inr_model = getattr(network_config, metadata['inr_model']).build_model().to(device)
    siren_model = getattr(network_config, metadata['siren_model']).build_model().to(device)
    inr_mapping = getattr(network_config, metadata['inr_mapping']).build_model(inr_model.style_dim_dict).to(device)
    siren_mapping = getattr(network_config, metadata['siren_mapping']).build_model(siren_model.style_dim_dict).to(device)
    generator = getattr(generators, metadata['generator'])(
        z_dim=metadata['latent_dim'],
        siren_model=siren_model,
        inr=inr_model,
        mapping_network_nerf=siren_mapping,
        mapping_network_inr=inr_mapping,
    ).to(device)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=3000, help="number of epochs of training")
    parser.add_argument("--sample_interval", type=int, default=200, help="interval between image sampling")
    parser.add_argument('--output_dir', type=str, default='debug')
    parser.add_argument('--load_dir', type=str, default='')
    parser.add_argument('--curriculum', type=str, required=True)
    parser.add_argument('--eval_freq', type=int, default=5000)
    parser.add_argument('--port', type=str, default='12355')
    parser.add_argument('--set_step', type=int, default=None)
    parser.add_argument('--model_save_interval', type=int, default=5000)

    opt = parser.parse_args()
    print(opt)

    os.makedirs(opt.output_dir, exist_ok=True)

    num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if num_gpus > 1:
        mp.spawn(train, args=(num_gpus, opt), nprocs=num_gpus, join=True)
    else:
        train(rank=0, world_size=num_gpus, opt=opt)
