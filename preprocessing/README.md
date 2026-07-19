<h1 align="center"> Preprocessing Guide
</h1>

## Environment setup

```bash
conda create -n reglue_preprocess python=3.10 -y
conda activate reglue_preprocess
pip install -r requirements.txt
```

## Dataset preparation

Edit `[YOUR_DOWNLOAD_PATH]` / `[TARGET_PATH]` in the two scripts, then run from `preprocessing/`:

```bash
cd preprocessing
bash dataset_prepare_convert.sh
bash dataset_prepare_encode.sh
```
