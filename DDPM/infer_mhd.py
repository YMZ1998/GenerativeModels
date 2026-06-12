from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from torch.amp import autocast
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from generative.networks.schedulers import DDIMScheduler, DDPMScheduler, PNDMScheduler  # noqa: E402
from utils import (  # noqa: E402
    create_model,
    crop_or_pad_slice_2d,
    denormalize_hu,
    load_checkpoint,
    normalize_hu,
    print_config_summary,
    read_config,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 3D MHD CBCT-to-CT inference with a trained 2D DDPM.")
    parser.add_argument("--config", type=str, default=str(SCRIPT_DIR / "config.json"), help="Path to config.json.")
    parser.add_argument(
        "--input",
        type=str,
        default="D:/Data/cbct/denoise_output.mhd",
        help="Input 3D CBCT .mhd path.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output synthetic CT .mhd path. Defaults to '<input_stem>_ddpm_sct.mhd' beside the input.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best",
        help="Checkpoint path, or alias 'best'/'latest' under config output_dir.",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Override device, for example cuda:1 or cpu.")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of axial slices sampled at once.")
    parser.add_argument(
        "--scheduler",
        choices=("ddim", "ddpm", "pndm"),
        default="ddim",
        help="Sampling scheduler. DDIM is much faster for MHD volume inference.",
    )
    parser.add_argument(
        "--mode",
        choices=("img2img", "sample"),
        default="img2img",
        help="img2img starts from a noised CBCT slice and preserves structure; sample starts from pure noise.",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.1,
        help="Noise strength for img2img mode. Lower is closer to CBCT; higher changes more but can become noisy.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help="Override sampling steps. Defaults to 25 for MHD inference.",
    )
    parser.add_argument(
        "--empty-cache-interval",
        type=int,
        default=0,
        help="Call torch.cuda.empty_cache every N slice batches. 0 disables it for better speed.",
    )
    parser.add_argument(
        "--use-spacing-resample",
        action="store_true",
        help="Use config spacing to resample the whole volume. Disabled by default for MHD inference.",
    )
    parser.add_argument(
        "--no-resize",
        action="store_true",
        help="Do not resize the input volume to config spatial_size before inference.",
    )
    parser.add_argument(
        "--resize-spacing",
        choices=("same", "keep-fov"),
        default="same",
        help="MHD spacing after in-plane resize. 'same' preserves input spacing; 'keep-fov' changes spacing to preserve physical FOV.",
    )
    parser.add_argument(
        "--debug-resize-only",
        action="store_true",
        help="Only write the resized input MHD, then exit without running the model.",
    )
    parser.add_argument(
        "--no-resample",
        action="store_true",
        help="Deprecated alias kept for compatibility; spacing resampling is already disabled by default.",
    )
    parser.add_argument(
        "--restore-original-grid",
        dest="restore_original_grid",
        action="store_true",
        default=True,
        help="Save the prediction with the input MHD size/spacing/origin/direction. Enabled by default.",
    )
    parser.add_argument(
        "--keep-working-grid",
        dest="restore_original_grid",
        action="store_false",
        help="Save the prediction on the internal model working grid instead of the original input grid.",
    )
    parser.add_argument(
        "--outside-fill",
        choices=("cbct", "hu-min", "zero"),
        default="cbct",
        help="Fill value outside the center model patch when the input slice is larger than spatial_size.",
    )
    parser.add_argument(
        "--output-pixel-type",
        choices=("same", "float32", "float64", "int16", "uint16", "int32", "uint32", "uint8", "int8"),
        default="same",
        help="Output MHD pixel type. 'same' preserves the input MHD pixel type.",
    )
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA autocast during inference.")
    return parser.parse_args()


def resolve_checkpoint_path(checkpoint: str, output_dir: str | Path) -> Path:
    text = str(checkpoint).strip()
    lowered = text.lower()
    output_dir = Path(output_dir)
    if lowered == "best":
        return output_dir / "best.pt"
    if lowered == "latest":
        return output_dir / "latest.pt"

    path = Path(text)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def resample_image_to_spacing(image: sitk.Image, spacing: Sequence[float]) -> sitk.Image:
    old_spacing = np.array(image.GetSpacing(), dtype=np.float64)
    new_spacing = np.array([float(v) for v in spacing], dtype=np.float64)
    old_size = np.array(image.GetSize(), dtype=np.int64)
    new_size = np.maximum(np.round(old_size * old_spacing / new_spacing).astype(np.int64), 1)

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputSpacing(tuple(float(v) for v in new_spacing))
    resampler.SetSize([int(v) for v in new_size])
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetDefaultPixelValue(-1000.0)
    return resampler.Execute(image)


def resize_image_inplane(image: sitk.Image, spatial_size_hw: Sequence[int], spacing_mode: str = "same") -> sitk.Image:
    old_size = image.GetSize()
    old_spacing = image.GetSpacing()
    target_h, target_w = int(spatial_size_hw[0]), int(spatial_size_hw[1])
    image_array = sitk.GetArrayFromImage(image).astype(np.float32)
    image_tensor = torch.from_numpy(image_array).unsqueeze(1)
    resized_array = (
        F.interpolate(image_tensor, size=(target_h, target_w), mode="bilinear", align_corners=False)
        .squeeze(1)
        .numpy()
        .astype(np.float32)
    )

    resized_image = sitk.GetImageFromArray(resized_array)
    if spacing_mode == "keep-fov":
        new_spacing = (
            float(old_spacing[0]) * float(old_size[0]) / float(target_w),
            float(old_spacing[1]) * float(old_size[1]) / float(target_h),
            float(old_spacing[2]),
        )
    else:
        new_spacing = old_spacing
    resized_image.SetSpacing(tuple(float(v) for v in new_spacing))
    resized_image.SetOrigin(image.GetOrigin())
    resized_image.SetDirection(image.GetDirection())
    return resized_image


def resize_volume_array_inplane(volume_zyx: np.ndarray, target_hw: Sequence[int]) -> np.ndarray:
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    volume_tensor = torch.from_numpy(volume_zyx.astype(np.float32, copy=False)).unsqueeze(1)
    return (
        F.interpolate(volume_tensor, size=(target_h, target_w), mode="bilinear", align_corners=False)
        .squeeze(1)
        .numpy()
        .astype(np.float32)
    )


def resample_to_reference(image: sitk.Image, reference: sitk.Image, default_value: float) -> sitk.Image:
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(float(default_value))
    return resampler.Execute(image)


def resolve_output_pixel_id(output_pixel_type: str, reference: sitk.Image) -> int:
    if output_pixel_type == "same":
        return int(reference.GetPixelID())

    pixel_ids = {
        "float32": sitk.sitkFloat32,
        "float64": sitk.sitkFloat64,
        "int16": sitk.sitkInt16,
        "uint16": sitk.sitkUInt16,
        "int32": sitk.sitkInt32,
        "uint32": sitk.sitkUInt32,
        "uint8": sitk.sitkUInt8,
        "int8": sitk.sitkInt8,
    }
    return int(pixel_ids[output_pixel_type])


def cast_image_pixel_type(image: sitk.Image, pixel_id: int) -> sitk.Image:
    if int(image.GetPixelID()) == int(pixel_id):
        return image
    return sitk.Cast(image, pixel_id)


def paste_center_patch(
    prediction_hu: torch.Tensor,
    cbct_slice_hu: np.ndarray,
    outside_fill: str,
    hu_min: float,
) -> np.ndarray:
    out_h, out_w = cbct_slice_hu.shape
    pred = prediction_hu.float().cpu().numpy().astype(np.float32)
    pred_h, pred_w = pred.shape

    if outside_fill == "cbct":
        output = cbct_slice_hu.astype(np.float32, copy=True)
    elif outside_fill == "zero":
        output = np.zeros((out_h, out_w), dtype=np.float32)
    else:
        output = np.full((out_h, out_w), float(hu_min), dtype=np.float32)

    copy_h = min(out_h, pred_h)
    copy_w = min(out_w, pred_w)
    dst_top = max((out_h - pred_h) // 2, 0)
    dst_left = max((out_w - pred_w) // 2, 0)
    src_top = max((pred_h - out_h) // 2, 0)
    src_left = max((pred_w - out_w) // 2, 0)

    output[dst_top : dst_top + copy_h, dst_left : dst_left + copy_w] = pred[
        src_top : src_top + copy_h, src_left : src_left + copy_w
    ]
    return output


def create_scheduler(name: str, num_train_timesteps: int):
    if name == "ddim":
        return DDIMScheduler(num_train_timesteps=num_train_timesteps)
    if name == "pndm":
        return PNDMScheduler(num_train_timesteps=num_train_timesteps, skip_prk_steps=True)
    return DDPMScheduler(num_train_timesteps=num_train_timesteps)


def add_noise_from_scheduler(scheduler, clean: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
    alphas_cumprod = scheduler.alphas_cumprod.to(device=clean.device, dtype=clean.dtype)
    alpha = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
    return alpha.sqrt() * clean + (1.0 - alpha).sqrt() * noise


def move_scheduler_tensors_to_device(scheduler, device: torch.device) -> None:
    for name in ("betas", "alphas", "alphas_cumprod", "one", "timesteps"):
        value = getattr(scheduler, name, None)
        if torch.is_tensor(value):
            setattr(scheduler, name, value.to(device))


@torch.no_grad()
def sample_batch(
    model: torch.nn.Module,
    scheduler,
    cbct_batch: torch.Tensor,
    device: torch.device,
    use_amp: bool,
    mode: str,
    strength: float,
) -> torch.Tensor:
    timesteps_to_run = scheduler.timesteps
    if mode == "img2img":
        strength = max(0.0, min(float(strength), 1.0))
        start_index = int(round((1.0 - strength) * (len(scheduler.timesteps) - 1)))
        start_index = max(0, min(start_index, len(scheduler.timesteps) - 1))
        timesteps_to_run = scheduler.timesteps[start_index:]
        start_t = int(timesteps_to_run[0].item()) if torch.is_tensor(timesteps_to_run[0]) else int(timesteps_to_run[0])
        start_timesteps = torch.full((cbct_batch.shape[0],), start_t, device=device, dtype=torch.long)
        current = add_noise_from_scheduler(scheduler, cbct_batch, torch.randn_like(cbct_batch), start_timesteps)
    else:
        current = torch.randn_like(cbct_batch, device=device)

    for timestep in timesteps_to_run:
        t_int = int(timestep.item()) if torch.is_tensor(timestep) else int(timestep)
        timesteps = torch.full((cbct_batch.shape[0],), t_int, device=device, dtype=torch.long)
        model_input = torch.cat((cbct_batch, current), dim=1)
        with autocast(device_type=device.type, enabled=use_amp):
            model_output = model(x=model_input, timesteps=timesteps)
        current, _ = scheduler.step(model_output.float(), t_int, current)
    return current


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    set_seed(int(config["seed"]))

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_name(f"{input_path.stem}_ddpm_sct.mhd")
    checkpoint_path = resolve_checkpoint_path(args.checkpoint, config["output_dir"])
    if not input_path.is_file():
        raise FileNotFoundError(f"Input MHD not found: {input_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device_name = args.device or config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    use_amp = bool(config.get("amp", True)) and device.type == "cuda" and not args.no_amp
    num_inference_steps = int(args.num_inference_steps or config.get("mhd_num_inference_steps", 25))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print_config_summary(
        config,
        [
            "device",
            "spatial_size",
            "spacing",
            "hu_min",
            "hu_max",
            "num_train_timesteps",
            "num_inference_steps",
            "model_channels",
            "attention_levels",
            "num_res_blocks",
            "num_head_channels",
            "norm_num_groups",
            "amp",
        ],
        title="MHD inference configuration",
    )
    print(f"input: {input_path}")
    print(f"output: {output_path}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"device: {device}")
    print(f"batch_size: {args.batch_size}")
    print(f"scheduler: {args.scheduler}")
    print(f"mode: {args.mode}")
    print(f"strength: {args.strength}")
    print(f"num_inference_steps: {num_inference_steps}")
    print(f"use_spacing_resample: {args.use_spacing_resample}")
    print(f"resize_to_spatial_size: {not args.no_resize}")
    print(f"resize_spacing: {args.resize_spacing}")
    print(f"debug_resize_only: {args.debug_resize_only}")
    print(f"restore_original_grid: {args.restore_original_grid}")
    print(f"outside_fill: {args.outside_fill}")
    print(f"output_pixel_type: {args.output_pixel_type}")
    print(f"empty_cache_interval: {args.empty_cache_interval}")
    print("")

    input_image = sitk.ReadImage(str(input_path))
    output_pixel_id = resolve_output_pixel_id(args.output_pixel_type, input_image)
    working_image = input_image
    if args.use_spacing_resample and config.get("spacing") is not None:
        working_image = resample_image_to_spacing(input_image, config["spacing"])
    elif not args.no_resize:
        working_image = resize_image_inplane(input_image, config["spatial_size"], spacing_mode=args.resize_spacing)

    if args.debug_resize_only:
        if args.restore_original_grid:
            debug_array = sitk.GetArrayFromImage(working_image).astype(np.float32)
            input_h, input_w = int(input_image.GetSize()[1]), int(input_image.GetSize()[0])
            debug_array = resize_volume_array_inplane(debug_array, (input_h, input_w))
            debug_image = sitk.GetImageFromArray(debug_array)
            debug_image.CopyInformation(input_image)
        else:
            debug_image = working_image
        debug_image = cast_image_pixel_type(debug_image, output_pixel_id)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(debug_image, str(output_path))
        print(f"Input MHD size xyz: {input_image.GetSize()} | spacing xyz: {input_image.GetSpacing()}")
        print(f"Input pixel type: {input_image.GetPixelIDTypeAsString()}")
        print(f"Debug output MHD size xyz: {debug_image.GetSize()} | spacing xyz: {debug_image.GetSpacing()}")
        print(f"Debug output pixel type: {debug_image.GetPixelIDTypeAsString()}")
        print(f"Saved resized input MHD: {output_path}")
        return

    cbct_array = sitk.GetArrayFromImage(working_image).astype(np.float32)
    depth, height, width = cbct_array.shape
    finite_values = cbct_array[np.isfinite(cbct_array)]
    percentiles = np.percentile(finite_values, [0, 1, 50, 99, 100]) if finite_values.size else [np.nan] * 5
    print(f"Input MHD size xyz: {input_image.GetSize()} | spacing xyz: {input_image.GetSpacing()}")
    print(f"Input pixel type: {input_image.GetPixelIDTypeAsString()}")
    print(f"Working volume shape zyx: {cbct_array.shape} | spacing xyz: {working_image.GetSpacing()}")
    print(
        "Working intensity percentiles: "
        f"min={percentiles[0]:.2f}, p1={percentiles[1]:.2f}, p50={percentiles[2]:.2f}, "
        f"p99={percentiles[3]:.2f}, max={percentiles[4]:.2f}"
    )
    slice_batches = int(np.ceil(depth / max(1, int(args.batch_size))))
    if args.mode == "img2img":
        effective_steps = max(1, int(round(num_inference_steps * max(0.0, min(float(args.strength), 1.0)))))
    else:
        effective_steps = num_inference_steps
    total_network_calls = slice_batches * effective_steps
    print(
        f"Estimated UNet calls: about {total_network_calls} = "
        f"slice_batches({slice_batches}) x effective_steps({effective_steps})"
    )

    model = create_model(config).to(device)
    load_checkpoint(checkpoint_path, model, map_location=device)
    model.eval()

    scheduler = create_scheduler(args.scheduler, num_train_timesteps=int(config["num_train_timesteps"]))
    scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=device)
    move_scheduler_tensors_to_device(scheduler, device)

    batch_size = max(1, int(args.batch_size))
    output_array = np.empty((depth, height, width), dtype=np.float32)
    progress = tqdm(range(0, depth, batch_size), desc="MHD inference", unit="batch", dynamic_ncols=True)
    for batch_index, start in enumerate(progress, start=1):
        end = min(start + batch_size, depth)
        slices = []
        for z in range(start, end):
            cbct_slice = torch.from_numpy(cbct_array[z]).unsqueeze(0)
            cbct_slice = normalize_hu(cbct_slice, config["hu_min"], config["hu_max"])
            slices.append(crop_or_pad_slice_2d(cbct_slice, config["spatial_size"]))

        cbct_batch = torch.stack(slices, dim=0).to(device=device)
        sampled = sample_batch(
            model,
            scheduler,
            cbct_batch,
            device,
            use_amp=use_amp,
            mode=args.mode,
            strength=float(args.strength),
        )
        sampled_hu = denormalize_hu(sampled[:, 0].cpu(), config["hu_min"], config["hu_max"])

        for offset, z in enumerate(range(start, end)):
            output_array[z] = paste_center_patch(
                sampled_hu[offset], cbct_array[z], args.outside_fill, float(config["hu_min"])
            )

        if device.type == "cuda" and int(args.empty_cache_interval) > 0 and batch_index % int(args.empty_cache_interval) == 0:
            torch.cuda.empty_cache()

    if args.restore_original_grid:
        input_h, input_w = int(input_image.GetSize()[1]), int(input_image.GetSize()[0])
        output_array = resize_volume_array_inplane(output_array, (input_h, input_w))
        output_image = sitk.GetImageFromArray(output_array)
        output_image.CopyInformation(input_image)
    else:
        output_image = sitk.GetImageFromArray(output_array)
        output_image.CopyInformation(working_image)

    output_image = cast_image_pixel_type(output_image, output_pixel_id)
    print(f"Output MHD size xyz: {output_image.GetSize()} | spacing xyz: {output_image.GetSpacing()}")
    print(f"Output pixel type: {output_image.GetPixelIDTypeAsString()}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(output_image, str(output_path))
    print(f"Saved synthetic CT MHD: {output_path}")


if __name__ == "__main__":
    main()
