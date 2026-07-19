import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os

import torch.nn as nn
import torch.optim as optim
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import numpy as np
from collections import OrderedDict
from PIL import Image
import argparse
import logging
from dataset import CustomDataset
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize
from tqdm.auto import tqdm, trange
from compressor import EncoderDecoder
from torchvision import transforms as pth_transforms

import sys
import pdb

# Functionality for multiprocessing debugging
class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child
    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()

@torch.no_grad()
def sample_posterior(moments, latents_scale=1., latents_bias=0.):
    device = moments.device
    mean, std = torch.chunk(moments, 2, dim=1)
    z = mean + std * torch.randn_like(mean)
    z = (z * latents_scale + latents_bias) 
    return z 

def preprocess_raw_image(x):
    resolution = x.shape[-1]
    x = x / 255.
    x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    scale = resolution // 256
    x = torch.nn.functional.interpolate(x, 224 * scale, mode='bicubic')
    return x

@torch.no_grad()
def load_encoders(device, resolution=256, rank=0):
    assert resolution in (256, 512)
    repo, tag = 'facebookresearch/dinov2', 'dinov2_vitb14_reg'

    if rank == 0:
        # Only rank-0 does the download / extraction
        encoder = torch.hub.load(repo, tag, trust_repo=True)
    # Everyone waits until rank-0 finishes
    dist.barrier()
    if rank != 0:
        # Now the repo is safely in the cache; just load it
        encoder = torch.hub.load(repo, tag, trust_repo=True)

    del encoder.head
    encoder.head = torch.nn.Identity()
    encoder = encoder.to(device)
    encoder.eval()
    return encoder

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    local_batch_size = int(args.global_batch_size // dist.get_world_size())
    INPUT_CHANNELS = 768  # DINOv2 feature channels

    if rank == 0:
        os.makedirs(args.results_path, exist_ok=True)
        experiment_index = len(glob(f"{args.results_path}/*"))
        output_channels = int(args.encoder_channel_list.split(',')[-1])
        experiment_dir = os.path.join(args.results_path, f"{experiment_index:03d}-image_size-{args.image_size}-out_channels-{output_channels}-layers-{args.interm_layers}")
        os.makedirs(experiment_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info("==== Experiment Arguments ====")
        for k, v in sorted(vars(args).items()):
            logger.info(f"{k}: {v}")
        logger.info("=============================")
        logger.info(f"Experiment directory created at {experiment_dir}")
        logger.info(f"Using output channels: {output_channels}")
        logger.info(f"Using global batch size: {args.global_batch_size}")
        logger.info(f"Using learning rate: {args.learning_rate}")
    else:
        logger = create_logger(None)


    encoder_channel_list = [int(x) for x in args.encoder_channel_list.split(',')]
    model = EncoderDecoder(build_channel_list = encoder_channel_list)
    model = model.to(device)
    model = DDP(model.to(device), device_ids=[rank])
    args.interm_layers = [int(x) for x in args.interm_layers.split(',')]
    # Define dataset    
    dataset_train = CustomDataset(args.train_data_dir)
  
    # Define samplers and dataloaders
    train_sampler = DistributedSampler(
        dataset_train,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )

    train_loader = DataLoader(
        dataset_train,
        batch_size=local_batch_size,
        shuffle=False,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    norm_stats = torch.load(args.norm_stats, weights_only=False)
    mean_dino = torch.tensor(norm_stats["mean"]).to(device)
    std_dino = torch.tensor(norm_stats["std"]).to(device)

    valdir = args.val_data_dir
    
    val_transform = pth_transforms.Compose([
        pth_transforms.Resize(args.image_size, interpolation=3),
        pth_transforms.CenterCrop(args.image_size),
        pth_transforms.Lambda(lambda img: torch.from_numpy(np.array(img)).permute(2, 0, 1).float()),
    ])

    dataset_val = ImageFolder(valdir, transform=val_transform)
    val_sampler = DistributedSampler(
        dataset_val,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=False,
        seed=args.global_seed
    )

    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=local_batch_size,
        num_workers=args.num_workers,
        sampler=val_sampler,
        pin_memory=True,
        drop_last=True
    )
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    start_time = time()

    dino_encoder = load_encoders(device=device, resolution=args.image_size, rank=rank)
    dino_encoder.eval()

    criterion = nn.MSELoss()    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0.0001)
    model.train()

    for epoch in trange(args.num_epochs, desc="Epoch", position=0, disable=(rank != 0)):
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for raw_image, _ in tqdm(val_loader, desc="Val", disable=(rank != 0), total=len(val_loader), position=1, leave=False):
                raw_image = raw_image.to(device)

                with torch.no_grad():
                    raw_image_ = preprocess_raw_image(raw_image)
                    z = dino_encoder.get_intermediate_layers(raw_image_, n=args.interm_layers, reshape=True, norm=True)
                    z = torch.cat(z,dim = 1)
                    z = z.reshape([z.shape[0],z.shape[1],-1])
                    z = torch.permute(z,(0,2,1))
                    z = (z - mean_dino) / std_dino
                    B, N, C = z.shape
                    H = W = int(N ** 0.5)
                    z = z.transpose(1, 2).contiguous().view(B, C, H, W)
                reconstructed_output, _ = model(z)
                
                loss = criterion(reconstructed_output, z)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        if rank == 0:
            model_save_path = f"{experiment_dir}/compression_model.pth"
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path} (, validation loss: {avg_val_loss:.4f})")
        train_loss = 0
        model.train()
        for raw_image, _, _ in tqdm(train_loader, desc=f"Train", disable=(rank != 0), total=len(train_loader), position=1, leave=False):
            raw_image = raw_image.to(device)
            
            with torch.no_grad():
                raw_image_ = preprocess_raw_image(raw_image)
                z = dino_encoder.get_intermediate_layers(raw_image_, n=args.interm_layers, reshape=True, norm=True)
                z = torch.cat(z,dim = 1)
                z = z.reshape([z.shape[0],z.shape[1],-1])
                z = torch.permute(z,(0,2,1))
                z = (z - mean_dino) / std_dino
                B, N, C = z.shape
                H = W = int(N ** 0.5)
                z = z.transpose(1, 2).contiguous().view(B, C, H, W)            
            optimizer.zero_grad()
            reconstructed_output, _ = model(z)

            loss = criterion(reconstructed_output, z)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        scheduler.step()

        torch.cuda.synchronize()
        end_time = time()

        avg_train_loss = train_loss / len(train_loader)
        avg_train_loss_tr = torch.tensor(avg_train_loss, device=device)
        dist.all_reduce(avg_train_loss_tr, op=dist.ReduceOp.SUM)
        avg_train_loss_tr = avg_train_loss_tr.item() / dist.get_world_size()

        avg_val_loss_tr = torch.tensor(avg_val_loss, device=device)
        dist.all_reduce(avg_val_loss_tr, op=dist.ReduceOp.SUM)
        avg_val_loss_tr = avg_val_loss_tr.item() / dist.get_world_size()

        logger.info(f"(Epoch={epoch + 1}) Train Loss: {avg_train_loss_tr:.4f}, Val Loss: {avg_val_loss_tr:.4f}, Epoch Time: {end_time - start_time:.2f}s")
        start_time = time()
    cleanup()       
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_dir", type=str)
    parser.add_argument("--val_data_dir", type=str)
    parser.add_argument("--results_path", type=str, default="./compression_model")
    parser.add_argument("--norm_stats", type=str, help='Path to the DINO normalization stats (mean and std).')
    parser.add_argument("--image_size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num_epochs", type=int, default=25)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--global_batch_size", type=int, default=4096)
    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--interm_layers", type = str, default = "8,9,10,11")
    parser.add_argument("--encoder_channel_list", type = str, default = "3072,256,16")
    args = parser.parse_args()
    main(args)
