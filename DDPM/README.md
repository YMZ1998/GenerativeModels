# DDPM CBCT-to-CT Baseline

This folder contains a minimal 2D axial-slice conditional DDPM baseline for synthesizing CT-like images from paired CBCT/CT NIfTI volumes.

It reuses the MONAI Generative Models components already present in this repository:

- `DiffusionModelUNet`
- `DDPMScheduler`
- MONAI NIfTI loading, orientation, spacing, and tensor transforms

The baseline is intended for research prototyping only. It is not validated for clinical diagnosis, dose calculation, or treatment planning.

## Data Layout

The checked-in `config.json` is set up for this SynthRAD pelvis folder:

```text
E:/CBCT2CT/SynthRAD/Task2/pelvis
```

The supported SynthRAD layout is one case per folder:

```text
E:/CBCT2CT/SynthRAD/Task2/pelvis/
  2PA001/
    cbct.nii.gz
    ct.nii.gz
    mask.nii.gz
  2PA002/
    cbct.nii.gz
    ct.nii.gz
    mask.nii.gz
```

Set this in `DDPM/config.json`:

```json
{
  "data_root": "E:/CBCT2CT/SynthRAD/Task2/pelvis",
  "data_layout": "synthrad_case_dirs",
  "val_fraction": 0.1
}
```

The baseline ignores `mask.nii.gz` for now and uses a deterministic seed-based train/validation split.

The original paired split-folder layout is also supported:

```text
data/cbct_ct/
  train/
    cbct/
      case001.nii.gz
      case002.nii.gz
    ct/
      case001.nii.gz
      case002.nii.gz
  val/
    cbct/
      case101.nii.gz
    ct/
      case101.nii.gz
```

The pairing key is the NIfTI file name without `.nii` or `.nii.gz`.

## Training

Edit `DDPM/config.json`, especially:

- `data_root`
- `output_dir`
- `spatial_size`
- `spacing`
- `hu_min` and `hu_max`
- `batch_size`
- `epochs`
- `train_steps_per_epoch`
- `val_steps_per_epoch`

Then run:

```powershell
python DDPM/preprocess.py --config DDPM/config.json
python DDPM/train.py --config DDPM/config.json
```

`preprocess.py` writes slice-level `.pt` files to `preprocessed_dir` and overwrites existing cached slices by default. Add `--skip-existing` only when you intentionally want to keep already cached cases. `train.py` uses the cache when `use_preprocessed` is `true`, which avoids repeatedly decompressing and resampling full NIfTI volumes during training.

To inspect the preprocessed slices:

```powershell
python DDPM/visualize_preprocessed.py --config DDPM/config.json --split train --num-samples 8
python DDPM/visualize_preprocessed.py --config DDPM/config.json --split train --case-id 2PA001 --num-samples 8
python DDPM/visualize_preprocessed.py --config DDPM/config.json --split train --num-samples 8 --seed 42
```

The script randomly shuffles cached slices by default, uses `matplotlib.pyplot` for plotting, and saves the figure to `DDPM/visualizations/`. The columns are CBCT, CT, and `CT - CBCT`.
It rotates slices by 90 degrees counterclockwise by default for easier axial viewing; adjust this with `--rotate-k`, `--flip-lr`, or `--flip-ud` if your local display convention differs.

Checkpoints are written to `output_dir`:

- `latest.pt`
- `best.pt`
- `epoch_XXXX.pt` every `save_interval` epochs

The default config has `auto_resume` enabled, so rerunning training will continue from `output_dir/latest.pt` when it exists:

```powershell
python DDPM/train.py --config DDPM/config.json
```

You can also choose a checkpoint explicitly:

```powershell
python DDPM/train.py --config DDPM/config.json --resume latest
python DDPM/train.py --config DDPM/config.json --resume best
python DDPM/train.py --config DDPM/config.json --resume DDPM/runs/pelvis_baseline/epoch_0010.pt
```

To force a fresh run without loading `latest.pt`:

```powershell
python DDPM/train.py --config DDPM/config.json --no-resume
```

Each epoch is iteration-based by default rather than a full pass over all cached slices. With the checked-in config, one training epoch is `train_steps_per_epoch=500` batches, and each validation run uses at most `val_steps_per_epoch=100` batches. Set either value to `0` or `null` to use the full DataLoader length.

When resuming, `epochs` is treated as the target final epoch. For example, if `latest.pt` was saved at epoch 20 and `epochs` is 100, training continues from epoch 21 to epoch 100. With the default config this means continuing from iteration-block 21 to iteration-block 100, where each block has 500 training steps.

The model learns to predict CT diffusion noise from `concat(CBCT, noisy_CT)`.

## Visdom

Training can stream losses and intermediate samples to Visdom. Install and start Visdom in the `monai` environment before training:

```powershell
conda activate monai
python -m pip install visdom
python -m visdom.server -p 8097
```

Then open `http://localhost:8097` in a browser and run training normally. With the default config, the environment is `ddpm_pelvis`; the sample panel shows validation groups, one group per row: CBCT, real CT, and denoised CT predictions at `visdom_preview_timesteps` such as `50`, `200`, and `500`. `visdom_num_images` controls how many groups are shown, while `visdom_inference_batch_size` controls how many preview samples are passed through the network at once. Keep `visdom_inference_batch_size=1` for 512x512 training to avoid CUDA out-of-memory during visualization. Low preview timesteps should preserve structure first; high timesteps are a harder generation diagnostic.

## Inference

Run inference on a single CBCT NIfTI:

```powershell
python DDPM/infer.py `
  --config DDPM/config.json `
  --input E:/CBCT2CT/SynthRAD/Task2/pelvis/2PA001/cbct.nii.gz `
  --output DDPM/runs/pelvis_baseline/2PA001_sct.nii.gz `
  --checkpoint DDPM/runs/pelvis_baseline/best.pt
```

The script preprocesses the CBCT with the configured orientation and spacing, samples each axial slice, converts the result back to HU, and saves a NIfTI volume.

The current baseline center-crops or pads each inference slice to `spatial_size`, so the output in-plane size follows the config rather than the original CBCT matrix size.

For a 3D `.mhd` CBCT volume, use:

```powershell
python DDPM/infer_mhd.py
```

By default this reads:

```text
D:/Data/cbct/denoise_output.mhd
```

and writes:

```text
D:/Data/cbct/denoise_output_ddpm_sct.mhd
```

You can override paths and checkpoint selection:

```powershell
python DDPM/infer_mhd.py `
  --input D:/Data/cbct/denoise_output.mhd `
  --output D:/Data/cbct/denoise_output_ddpm_sct.mhd `
  --checkpoint latest `
  --scheduler ddim `
  --num-inference-steps 25 `
  --mode img2img `
  --strength 0.1 `
  --batch-size 1
```

The MHD script writes an `.mhd/.raw` pair and preserves the input MHD size, spacing, origin, direction, and pixel type by default. Internally it resizes each slice to `spatial_size` for the model, then resizes the prediction back to the input size before saving. Add `--keep-working-grid` only if you intentionally want to save the internal `512x512xz` working grid. Add `--use-spacing-resample` only if you intentionally want the older config-spacing behavior. Add `--output-pixel-type float32` only if you want a floating-point output. MHD inference defaults to DDIM with `mhd_num_inference_steps=25` and `--mode img2img`, which starts from a lightly noised CBCT slice so the output keeps input anatomy. Use `--mode sample` for pure DDPM sampling from noise, but that usually needs a well-trained in-domain model. Keep `--batch-size 1` for 512x512 inference if GPU memory is tight.

To debug MHD resizing without running the model:

```powershell
python DDPM/infer_mhd.py `
  --output D:/Data/cbct/denoise_output_resize_debug.mhd `
  --debug-resize-only
```

## Quick Smoke Test

For a fast local check, create a tiny paired NIfTI dataset, then reduce the config:

```json
{
  "spatial_size": [64, 64],
  "batch_size": 1,
  "epochs": 1,
  "train_steps_per_epoch": 2,
  "val_steps_per_epoch": 1,
  "num_train_timesteps": 10,
  "num_inference_steps": 5,
  "model_channels": [8, 8, 8],
  "attention_levels": [false, false, false],
  "num_head_channels": 8,
  "norm_num_groups": 8,
  "samples_per_volume": 2,
  "val_slices_per_volume": 1
}
```

Then run:

```powershell
python -m py_compile DDPM/train.py DDPM/infer.py DDPM/utils.py
python DDPM/preprocess.py --config path/to/smoke_config.json --output-dir path/to/cache --max-cases 1
python DDPM/train.py --config path/to/smoke_config.json --device cpu
python DDPM/infer.py --config path/to/smoke_config.json --device cpu --input path/to/cbct.nii.gz --output path/to/sct.nii.gz --checkpoint path/to/runs/best.pt
```

## Notes

- CBCT and CT should be rigidly or deformably registered before training.
- HU clipping defaults to `[-1000, 2000]`; adjust this for your anatomy and scanner protocol.
- A 2D DDPM is a simple baseline. For better 3D consistency and high-resolution work, use this as a stepping stone toward a 3D or latent diffusion model.
