NUM_GPUS=8
STEP="1000000"
SAVE_PATH="/your_path/linear-dinov2-b-enc8-sit-xl"
NUM_STEP=250
MODEL_SIZE='XL'
CFG_SCALE=2.3
CLS_CFG_SCALE=2.3
VAE_CFG='false' ## set to 'true' or 'false'
GH=0.9
GL=0.0
RESOLUTION=256

export NCCL_P2P_DISABLE=1

python -m torch.distributed.launch --master_port=13579 --nproc_per_node=$NUM_GPUS generate.py \
  --model SiT-${MODEL_SIZE}/2 \
  --num-fid-samples 50000 \
  --ckpt ${SAVE_PATH}/checkpoints/${STEP}.pt \
  --path-type=linear \
  --encoder-depth=8 \
  --projector-embed-dims=768 \
  --per-proc-batch-size=64 \
  --mode=sde \
  --num-steps=${NUM_STEP} \
  --cfg-scale=${CFG_SCALE} \
  --cls-cfg-scale=${CLS_CFG_SCALE} \
  --vae_cfg=${VAE_CFG} \
  --guidance-high=${GH} \
  --guidance-low=${GL} \
  --sample-dir ${SAVE_PATH}/samples \
  --cls=768 \
  --resolution=${RESOLUTION} \


python ./evaluations/evaluator.py \
    --ref_batch /path/to/VIRTUAL_imagenet${RESOLUTION}_labeled.npz \
    --sample_batch ${SAVE_PATH}/samples/SiT-${MODEL_SIZE}-2-${STEP}-size-${RESOLUTION}-vae-ema-cfg-${CFG_SCALE}-seed-0-sde-gh-${GH}-gl-${GL}-${CLS_CFG_SCALE}-vae_cfg-${VAE_CFG}.npz \
    --save_path ${SAVE_PATH}/samples \
    --cfg_cond 1 \
    --step ${STEP} \
    --num_steps ${NUM_STEP} \
    --cfg ${CFG_SCALE} \
    --cls_cfg ${CLS_CFG_SCALE} \
    --vae_cfg=${VAE_CFG} \
    --gh ${GH} \
    --gl ${GL}

