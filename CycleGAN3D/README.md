# 3D CycleGAN CBCT-to-CT Baseline

This folder contains a standalone 3D CycleGAN baseline for CBCT-to-CT synthesis.
It does not modify `DDPM/` or the `generative/` source tree.

## Data

Default dataset root:

```text
E:/CBCT2CT/SynthRAD/Task2/pelvis
```

Default layout:

```text
pelvis/
  2PA001/
    cbct.nii.gz
    ct.nii.gz
    mask.nii.gz        optional
  2PA002/
    cbct.nii.gz
    ct.nii.gz
```

The training code uses CycleGAN losses only. The paired CT is used for validation monitoring and visualization, not as a paired training loss.

## Preprocess

Preprocessing resizes each volume to `512x512xZ`, keeps the original number of axial slices, clips HU to `[-1000, 2000]`, and scales values to `[-1, 1]`.

By default, preprocessed volumes are saved as `npy` with `float16`, which is the fastest option for this training pipeline. Set `"preprocess_format": "nii.gz"`, `"mha"`, or `"mhd"` only when you need to inspect the preprocessed volumes directly in medical image viewers.

```powershell
python CycleGAN3D/preprocess.py --config CycleGAN3D/config.json
```

By default preprocessing overwrites existing files under `CycleGAN3D/preprocessed/`.

## Train

Start Visdom first if visualization is enabled:

```powershell
python -m visdom.server -port 8097
```

Then train:

```powershell
python CycleGAN3D/train.py --config CycleGAN3D/config.json
```

Resume:

```powershell
python CycleGAN3D/train.py --config CycleGAN3D/config.json --resume latest
```

Checkpoints are written to `CycleGAN3D/runs/pelvis_cyclegan3d/`:

```text
latest.pt
best.pt
epoch_0010.pt
```

Important defaults:

```json
{
  "device": "cuda:0",
  "preprocess_format": "npy",
  "preprocess_dtype": "float16",
  "patch_size_hwd": [128, 128, 64],
  "batch_size": 1,
  "train_steps_per_epoch": 500,
  "amp": true
}
```

If CUDA memory is not enough, first try:

```json
{
  "patch_size_hwd": [128, 128, 32],
  "generator_channels": 8,
  "discriminator_channels": 8
}
```

## Inference

NIfTI and MHD/MHA are supported. Output shape, spacing, origin, direction, and dtype are kept the same as the input.

```powershell
python CycleGAN3D/infer.py `
  --config CycleGAN3D/config.json `
  --input D:/Data/cbct/denoise_output.mhd `
  --output D:/Data/cbct/denoise_output_cyclegan3d_sct.mhd `
  --checkpoint CycleGAN3D/runs/pelvis_cyclegan3d/best.pt
```

If `--output` is omitted, the script writes `*_cyclegan3d_sct` next to the input file.
If `--checkpoint` is omitted, it uses `best.pt` from `output_dir`.

## Smoke Checks

```powershell
python -m py_compile CycleGAN3D/models.py CycleGAN3D/utils.py CycleGAN3D/preprocess.py CycleGAN3D/train.py CycleGAN3D/infer.py
```

For a quick functional test, create two tiny synthetic CBCT/CT case folders, set:

```json
{
  "xy_size": [64, 64],
  "patch_size_hwd": [64, 64, 16],
  "epochs": 1,
  "train_steps_per_epoch": 2,
  "val_steps_per_epoch": 1,
  "generator_channels": 8,
  "discriminator_channels": 8,
  "num_workers": 0,
  "visdom_enabled": false
}
```

Then run preprocess, train, and infer.

## Notes

This is a research baseline. Do not use generated CT images for clinical diagnosis or dose calculation without proper validation.
