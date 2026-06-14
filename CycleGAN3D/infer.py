from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import torch
from monai.inferers import sliding_window_inference

from models import make_generator
from utils import (
    cast_array,
    denormalize_hu_np,
    hwd_to_dhw,
    normalize_hu_np,
    print_config_summary,
    read_config,
    read_medical_image,
    resize_zhw,
    safe_torch_load,
    set_seed,
    write_medical_like,
)


ARCHITECTURE_KEYS = (
    "generator_channels",
    "generator_res_blocks",
)


def get_device(config: Mapping[str, Any]) -> torch.device:
    requested = config.get("device")
    if requested is None:
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(str(requested))
    if device.type == "cuda":
        if not torch.cuda.is_available():
            print("CUDA was requested but is not available. Falling back to CPU.")
            return torch.device("cpu")
        torch.cuda.set_device(device)
    return device


def default_output_path(input_path: Path) -> Path:
    name = input_path.name
    lower = name.lower()
    if lower.endswith(".nii.gz"):
        return input_path.with_name(name[:-7] + "_cyclegan3d_sct.nii.gz")
    return input_path.with_name(input_path.stem + "_cyclegan3d_sct" + input_path.suffix)


def strip_module_prefix(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(key.startswith("module.") for key in state_dict):
        return dict(state_dict)
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def load_cbct2ct_generator(checkpoint_path: Path, config: Mapping[str, Any], device: torch.device) -> torch.nn.Module:
    checkpoint = safe_torch_load(checkpoint_path, map_location="cpu")
    model_config = dict(config)
    checkpoint_config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    if isinstance(checkpoint_config, dict):
        for key in ARCHITECTURE_KEYS:
            if key in checkpoint_config:
                model_config[key] = checkpoint_config[key]

    model = make_generator(model_config)
    if isinstance(checkpoint, dict):
        if "G_CBCT2CT" in checkpoint:
            state_dict = checkpoint["G_CBCT2CT"]
        elif "G_A2B" in checkpoint:
            state_dict = checkpoint["G_A2B"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    model.load_state_dict(strip_module_prefix(state_dict), strict=True)
    model.to(device)
    model.eval()
    return model


def prepare_input(array: np.ndarray, config: Mapping[str, Any]) -> tuple[np.ndarray, tuple[int, int]]:
    original_hw = (int(array.shape[1]), int(array.shape[2]))
    volume = normalize_hu_np(array, float(config["hu_min"]), float(config["hu_max"]))
    if bool(config.get("infer_resize_xy", True)):
        volume = resize_zhw(volume, config["xy_size"])
    return volume, original_hw


def restore_output(
    normalized_volume: np.ndarray,
    original_hw: tuple[int, int],
    reference_dtype: np.dtype,
    config: Mapping[str, Any],
) -> np.ndarray:
    volume = normalized_volume
    if bool(config.get("infer_resize_xy", True)) and volume.shape[-2:] != original_hw:
        volume = resize_zhw(volume, original_hw)
    volume = denormalize_hu_np(volume, float(config["hu_min"]), float(config["hu_max"]))
    return cast_array(volume, reference_dtype, str(config.get("output_dtype", "same")))


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    volume: np.ndarray,
    config: Mapping[str, Any],
    device: torch.device,
) -> np.ndarray:
    tensor = torch.from_numpy(np.ascontiguousarray(volume[None, None]).astype(np.float32)).to(device)
    roi_size = hwd_to_dhw(config["patch_size_hwd"])
    kwargs = {
        "inputs": tensor,
        "roi_size": roi_size,
        "sw_batch_size": int(config.get("infer_sw_batch_size", 1)),
        "predictor": model,
        "overlap": float(config.get("infer_overlap", 0.25)),
        "mode": "gaussian",
        "progress": True,
    }
    warnings.filterwarnings(
        "ignore",
        message="Using a non-tuple sequence for multidimensional indexing.*",
        category=UserWarning,
        module="monai.inferers.utils",
    )
    try:
        output = sliding_window_inference(**kwargs)
    except TypeError:
        kwargs.pop("progress")
        output = sliding_window_inference(**kwargs)
    return output[0, 0].detach().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 3D CycleGAN CBCT-to-CT inference.")
    parser.add_argument("--config", default="CycleGAN3D/config.json", help="Path to config.json.")
    parser.add_argument("--input", default="D:/Data/cbct/denoise_output.mhd", help="Input .nii/.nii.gz/.mhd/.mha file.")
    parser.add_argument("--output", default=None, help="Output path. Defaults to *_cyclegan3d_sct.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path. Defaults to output_dir/best.pt.")
    args = parser.parse_args()

    config = read_config(args.config)
    set_seed(int(config["seed"]))
    device = get_device(config)

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else default_output_path(input_path)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else Path(str(config["output_dir"])) / "best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print_config_summary(
        config,
        [
            "device",
            "xy_size",
            "patch_size_hwd",
            "hu_min",
            "hu_max",
            "infer_resize_xy",
            "infer_sw_batch_size",
            "infer_overlap",
            "output_dtype",
        ],
        title="3D CycleGAN Inference",
    )
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Resolved device: {device}")

    array, image = read_medical_image(input_path)
    reference_dtype = array.dtype
    original_shape = tuple(int(v) for v in array.shape)
    print(f"Input shape ZHW: {original_shape} | dtype: {reference_dtype}")

    model = load_cbct2ct_generator(checkpoint_path, config, device)
    network_volume, original_hw = prepare_input(array, config)
    print(f"Network shape ZHW: {tuple(int(v) for v in network_volume.shape)}")

    prediction_norm = run_inference(model, network_volume, config, device)

    output_array = restore_output(prediction_norm, original_hw, reference_dtype, config)
    if tuple(output_array.shape) != original_shape:
        raise RuntimeError(f"Output shape {output_array.shape} does not match input shape {original_shape}.")
    if output_array.dtype != reference_dtype and str(config.get("output_dtype", "same")).lower() == "same":
        raise RuntimeError(f"Output dtype {output_array.dtype} does not match input dtype {reference_dtype}.")

    write_medical_like(output_array, image, output_path)
    print(f"Done. Wrote {output_path}")
    print(f"Output shape ZHW: {tuple(int(v) for v in output_array.shape)} | dtype: {output_array.dtype}")


if __name__ == "__main__":
    main()
