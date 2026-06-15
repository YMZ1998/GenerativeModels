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
  "generator_type": "resnet",
  "patch_size_hwd": [128, 128, 64],
  "batch_size": 1,
  "train_steps_per_epoch": 500,
  "amp": true
}
```

For better fine detail on paired SynthRAD data, the training config enables paired detail supervision:

```json
{
  "train_paired_sampling": true,
  "lambda_paired_l1": 5.0,
  "lambda_paired_gradient": 2.0
}
```

`lambda_paired_l1` makes generated CT match the registered CT intensity patch. `lambda_paired_gradient` matches 3D finite-difference edges, which usually helps bone cortex and organ boundaries. Set both weights to `0.0` and `train_paired_sampling` to `false` if you need pure unpaired CycleGAN behavior.

Reference-project-inspired options are available for new experiments:

```json
{
  "generator_type": "resunet",
  "discriminator_type": "spectral",
  "real_label": 0.9,
  "fake_label": 0.1,
  "generator_update_steps": 2
}
```

Use `generator_type: "resnet"` when resuming checkpoints trained with the original `CycleGAN3D` baseline.

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
The default inference ROI is larger than the training patch to reduce sliding-window seams:

```json
{
  "infer_roi_size_hwd": [512, 512, 32],
  "infer_strategy": "z_sliding",
  "infer_overlap": 0.75,
  "infer_blend_mode": "gaussian",
  "infer_padding_mode": "replicate",
  "infer_tta": false
}
```

`z_sliding` processes the full `512x512` plane and only chunks along Z. This avoids XY checkerboard blocks caused by patch-wise InstanceNorm.

```powershell
python CycleGAN3D/infer.py `
  --config CycleGAN3D/config.json `
  --input D:/Data/cbct/denoise_output.mhd `
  --output D:/Data/cbct/denoise_output_cyclegan3d_sct.mhd `
  --checkpoint CycleGAN3D/runs/pelvis_cyclegan3d/best.pt
```

For severe grid artifacts, try TTA:

```powershell
python CycleGAN3D/infer.py `
  --config CycleGAN3D/config.json `
  --input D:/Data/cbct/denoise_output.mhd `
  --output D:/Data/cbct/denoise_output_cyclegan3d_sct_tta.mhd `
  --checkpoint CycleGAN3D/runs/pelvis_cyclegan3d/best.pt `
  --tta
```

If CUDA memory is not enough during inference, reduce ROI while keeping overlap high:

```powershell
python CycleGAN3D/infer.py `
  --config CycleGAN3D/config.json `
  --input D:/Data/cbct/denoise_output.mhd `
  --output D:/Data/cbct/denoise_output_cyclegan3d_sct.mhd `
  --checkpoint CycleGAN3D/runs/pelvis_cyclegan3d/best.pt `
  --roi-size 256 256 32 `
  --overlap 0.75
```

If memory allows, compare with whole-volume inference:

```powershell
python CycleGAN3D/infer.py `
  --config CycleGAN3D/config.json `
  --input D:/Data/cbct/denoise_output.mhd `
  --output D:/Data/cbct/denoise_output_cyclegan3d_sct_whole.mhd `
  --checkpoint CycleGAN3D/runs/pelvis_cyclegan3d/best.pt `
  --strategy whole
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
