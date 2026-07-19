
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
    --feat_channels 16  \
    --checkpointing-steps 10000


    #Dataset Path
    #For example: /path/to/preprocessed/imagenet
    #This folder contains two folders
    #(1) The imagenet's RGB image: /path/to/preprocessed/imagenet/images/
    #(2) The imagenet's VAE latent: /path/to/preprocessed/imagenet/vae-sd/