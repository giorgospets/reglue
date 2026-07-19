<h1 align="center">
   REGLUE Your Latents with
    Global and Local Semantics for Entangled Diffusion
</h1>

<h1 align="center">ECCV 2026</h1>

<div align="center">
  <a href="https://scholar.google.com/citations?user=0PI25YQAAAAJ&hl=en" target="_blank">Giorgos&nbsp;Petsangourakis</a><sup>1,2</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://scholar.google.com/citations?user=Cw7CeiAAAAAJ&hl=en" target="_blank">Christos &nbspSgouropoulos</a><sup>1</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://scholar.google.com/citations?user=qiDVfC4AAAAJ&hl=en" target="_blank">Bill &nbspPsomas</a><sup>3</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://scholar.google.com/citations?user=BeIoqhwAAAAJ&hl=en" target="_blank">Theodoros&nbsp;Giannakopoulos</a><sup>1</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://scholar.google.com/citations?user=X73G9lYAAAAJ&hl=en" target="_blank">Giorgos&nbsp;Sfikas</a><sup>2</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://scholar.google.com/citations?user=B_dKcz4AAAAJ&hl=en" target="_blank">Ioannis&nbsp;Kakogeorgiou</a><sup>1</sup>
  <br>
  <sup>1</sup> IIT, National Centre for Scientific Research 'Demokritos' &emsp; <sup>2</sup> University of West Attica &emsp; <sup>3</sup> VRG, FEE, Czech Technical University in Prague
  <br><br>

  <a href="https://reglueyourlatents.github.io/"><img src="https://img.shields.io/badge/-Project%20Page-blue.svg?colorA=333&logo=html5"></a>
  <a href="https://arxiv.org/abs/2512.16636"><img src="https://img.shields.io/badge/arXiv-2512.16636-brown.svg?logo=arxiv&logoColor=white"></a>
  <a href="https://huggingface.co/giorgospets/reglue/tree/main"><img src="https://img.shields.io/badge/🤗-Model-blue.svg"></a>

  <br><br>

![teaser.png](teaser.png)

</div>

## Contents

- [1. Data preparation](#1-data-preparation)
- [2. Environment setup](#2-environment-setup)
- [3. Semantic Compressor](#3-compression-model)
- [4. Training](#4-training)
- [5. Generate images and evaluation](#5-generate-images-and-evaluation)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)

## 1. Data preparation

Follow the [preprocessing guide](preprocessing/README.md) to prepare ImageNet images and VAE latents.

## 2. Environment Setup

```bash
conda env create -f environment.yml
conda activate reglue
```

## 3. Semantic Compressor

Compute the DINOv2 normalization stats and train the compressor that maps DINOv2 features to the 16 semantic-compressed channels:

```bash
torchrun --nnodes=1 --nproc_per_node=1 calc_norm_stats.py \
    --data-dir /path/to/preprocessed/imagenet \
    --norm_stats_savedir ./norm_stats

torchrun --nproc_per_node=8 train_compressor.py \
    --train_data_dir /path/to/preprocessed/imagenet \
    --val_data_dir /path/to/original/imagenet/val \
    --norm_stats ./norm_stats/dino_mean_std_8_9_10_11_256.pth \
    --results_path ./compression_model
```
**Note:**
- `--val_data_dir` points to the **original** ImageNet validation set (an `ImageFolder` of raw class sub-folders), not the preprocessed data.
- `--train_data_dir` uses the preprocessed ImageNet from step 1 (the dir with `images/` and `vae-sd/`).

We provide the normalization stats at [`norm_stats/dino_mean_std_8_9_10_11_256.pth`](norm_stats/dino_mean_std_8_9_10_11_256.pth) and the pretrained compressor on [Hugging Face](https://huggingface.co/giorgospets/reglue/tree/main).

## 4. Training

Train the REGLUE model:

```bash
bash train.sh
```

`train.sh` contains the following content.

```bash
WANDB_MODE=disabled accelerate launch --multi_gpu --num_processes 8 train.py \
    --report-to="wandb" \
    --allow-tf32 \
    --mixed-precision="fp16" \
    --seed=0 \
    --path-type="linear" \
    --prediction="v" \
    --weighting="uniform" \
    --model="SiT-XL/2" \
    --enc-type="dinov2-vit-b" \
    --proj-coeff=0.5 \
    --encoder-depth=8 \
    --output-dir="/path/to/reglue_models/" \
    --exp-name="linear-dinov2-b-enc8-sit-xl" \
    --batch-size=256 \
    --data-dir="/path/to/preprocessed/imagenet" \
    --cls=0.03 \
    --compression_model_path path/to/compression_model.pth \
    --norm_stats ./norm_stats/dino_mean_std_8_9_10_11_256.pth \
    --interm_layers 8,9,10,11 \
    --encoder_channel_list 3072,256,16 \
    --feat_channels 16 \
    --checkpointing-steps 10000
```

The `--data-dir` folder is the preprocessed ImageNet from step 1 and must contain two sub-folders: `images/` (RGB images) and `vae-sd/` (VAE latents). Set `WANDB_MODE=online` to enable Weights & Biases logging.

This script will automatically create the folder `output-dir/exp-name` to save logs and checkpoints. You can adjust the following options:

- `--model`: `[SiT-B/2, SiT-XL/2]`
- `--encoder-depth`: `4` for `SiT-B/2`, `8` for `SiT-XL/2`
- `--data-dir`: preprocessed ImageNet from step 1
- `--compression_model_path` / `--norm_stats`: outputs from step 3
- `--output-dir` / `--exp-name`: where checkpoints and logs are saved
- `--cls`: weight of the global-semantics entanglement loss

## 5. Generate images and evaluation

Sample images from a trained checkpoint and compute the metrics:

```bash
bash eval.sh
```

Samples 50k images with `generate.py` and scores them with [`evaluations/evaluator.py`](evaluations/evaluator.py) (FID, sFID, IS, Precision, Recall). Set `SAVE_PATH`, `STEP` and `MODEL_SIZE` at the top of the script, and download the [ImageNet 256x256 reference batch](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz) (see [`evaluations/README.md`](evaluations/README.md) for other resolutions/datasets).

Pretrained REGLUE model (SiT-XL/2, 1M steps) can also be found at [Hugging Face](https://huggingface.co/giorgospets/reglue/tree/main).

## Acknowledgement

Built upon [SiT](https://github.com/willisma/SiT), [REPA](https://github.com/sihyun-yu/REPA), [REG](https://github.com/Martinser/REG), [ReDi](https://github.com/zelaki/ReDi), and the [edm2](https://github.com/NVlabs/edm2) preprocessing pipeline.

## Citation

```bibtex
@InProceedings{petsangourakis2026reglue,
  author={Giorgos Petsangourakis and Christos Sgouropoulos and Bill Psomas and Theodoros Giannakopoulos and Giorgos Sfikas and Ioannis Kakogeorgiou},
  title={REGLUE Your Latents with Global and Local Semantics for Entangled Diffusion},
  booktitle={ECCV},
  year={2026}
}
```
