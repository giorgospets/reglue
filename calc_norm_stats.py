import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.utils.data import DataLoader
from PIL import Image
import argparse
from torchvision.transforms import Normalize
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from accelerate import Accelerator
from dataset import CustomDataset
from tqdm import tqdm
import numpy as np
import os 


def preprocess_raw_image(x):
    resolution = x.shape[-1]
    x = x / 255.
    x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')

    return x
@torch.no_grad()
def load_encoders(device, resolution=256):
    assert (resolution == 256) or (resolution == 512)
    encoder = torch.hub.load('facebookresearch/dinov2', f'dinov2_vitb14_reg')
    del encoder.head
    encoder.head = torch.nn.Identity()
    encoder = encoder.to(device)
    encoder.eval()
    
    return encoder

def main(args):

    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup accelerator:
    accelerator = Accelerator()
    device = accelerator.device

    # Setup data:
    dataset = CustomDataset(args.data_dir)
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // accelerator.num_processes),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    encoder = load_encoders(device=device)
    encoder.eval()
    
    f_list = []
    interm_layers = [int(x) for x in args.interm_layers.split(",")]
    for c, (raw_image, _, _) in tqdm(enumerate(loader)):
        raw_image = raw_image.to(device)

        with torch.no_grad():
            with accelerator.autocast():
                raw_image_ = preprocess_raw_image(raw_image)
                z = encoder.get_intermediate_layers(raw_image_, n=interm_layers, reshape=True, norm=True)
                z = torch.cat(z, dim=1)
                z = z.reshape([z.shape[0],z.shape[1],-1])
                z = torch.permute(z,(0,2,1))

            f_list.append(z.flatten(end_dim=-2).float().cpu().numpy())
        if c > args.num_batches:
            break

    img_size = args.image_size
    interm_layers_str = args.interm_layers
    f = np.concatenate(f_list)
    # Standardize the data
    mean = np.mean(f, axis=0)
    std = np.std(f, axis=0)
    f = (f - mean)/std

    checkpoint = {
        'mean': mean,
        'std': std
    }


    os.makedirs(args.norm_stats_savedir, exist_ok=True)
    savepath = os.path.join(args.norm_stats_savedir, f"dino_mean_std_{interm_layers_str.replace(",","_")}_{img_size}.pth")
    torch.save(checkpoint,savepath)

if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--num_batches", type=int, default=300)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--interm_layers", type = str, default = "8,9,10,11")
    parser.add_argument("--norm_stats_savedir", type = str, default  = "./norm_stats")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    args = parser.parse_args()
    main(args)