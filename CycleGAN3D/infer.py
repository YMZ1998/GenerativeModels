from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
from tqdm import tqdm

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
    "generator_type",
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


def get_infer_roi_size(config: Mapping[str, Any]) -> tuple[int, int, int]:
    roi_size = config.get("infer_roi_size_hwd") or config["patch_size_hwd"]
    return hwd_to_dhw(roi_size)


def z_window_starts(depth: int, window: int, overlap: float) -> list[int]:
    if depth <= window:
        return [0]
    stride = max(1, int(round(window * (1.0 - overlap))))
    starts = list(range(0, depth - window + 1, stride))
    last_start = depth - window
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def z_blend_weight(window: int, config: Mapping[str, Any], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    mode = str(config.get("infer_blend_mode", "gaussian")).lower()
    if mode == "constant" or window <= 1:
        weight = torch.ones(window, device=device, dtype=dtype)
    else:
        coords = torch.arange(window, device=device, dtype=dtype) - (window - 1) * 0.5
        sigma = max(float(window) * float(config.get("infer_sigma_scale", 0.125)), 1.0)
        weight = torch.exp(-0.5 * (coords / sigma) ** 2)
        weight = torch.clamp(weight, min=1.0e-3)
    return weight.view(1, 1, window, 1, 1)


def z_sliding_predict(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    roi_size: tuple[int, int, int],
    config: Mapping[str, Any],
) -> torch.Tensor:
    _, _, depth, _, _ = tensor.shape
    roi_d = max(4, int(roi_size[0]))
    if roi_d % 4 != 0:
        roi_d = int(np.ceil(roi_d / 4.0) * 4)

    starts = z_window_starts(depth, roi_d, float(config.get("infer_overlap", 0.75)))
    output_sum = torch.zeros_like(tensor)
    weight_sum = torch.zeros_like(tensor)
    full_weight = z_blend_weight(roi_d, config, tensor.device, tensor.dtype)
    padding_mode = str(config.get("infer_padding_mode", "replicate"))

    for start in tqdm(starts, desc="Z-only inference", unit="chunk", dynamic_ncols=True):
        end = min(start + roi_d, depth)
        chunk = tensor[:, :, start:end]
        valid_d = int(chunk.shape[2])
        if valid_d < roi_d:
            chunk = F.pad(chunk, (0, 0, 0, 0, 0, roi_d - valid_d), mode=padding_mode)
        pred = model(chunk)[:, :, :valid_d]
        weight = full_weight[:, :, :valid_d]
        output_sum[:, :, start:end] += pred * weight
        weight_sum[:, :, start:end] += weight

    return output_sum / torch.clamp(weight_sum, min=1.0e-6)


def sliding_window_predict(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    roi_size: tuple[int, int, int],
    config: Mapping[str, Any],
) -> torch.Tensor:
    kwargs = {
        "inputs": tensor,
        "roi_size": roi_size,
        "sw_batch_size": int(config.get("infer_sw_batch_size", 1)),
        "predictor": model,
        "overlap": float(config.get("infer_overlap", 0.75)),
        "mode": str(config.get("infer_blend_mode", "gaussian")),
        "sigma_scale": float(config.get("infer_sigma_scale", 0.125)),
        "padding_mode": str(config.get("infer_padding_mode", "replicate")),
        "progress": True,
    }
    removable_keys = ("progress", "padding_mode", "sigma_scale")
    while True:
        try:
            return sliding_window_inference(**kwargs)
        except TypeError:
            for key in removable_keys:
                if key in kwargs:
                    kwargs.pop(key)
                    break
            else:
                raise


def predict_once(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    roi_size: tuple[int, int, int],
    config: Mapping[str, Any],
) -> torch.Tensor:
    strategy = str(config.get("infer_strategy", "z_sliding")).lower()
    if strategy in {"z", "z_only", "z_sliding"}:
        return z_sliding_predict(model, tensor, roi_size, config)
    if strategy in {"sliding", "sliding_window", "monai"}:
        return sliding_window_predict(model, tensor, roi_size, config)
    if strategy in {"whole", "whole_volume"}:
        return model(tensor)
    raise ValueError("infer_strategy must be 'z_sliding', 'sliding_window', or 'whole'.")


def maybe_tta_predict(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    roi_size: tuple[int, int, int],
    config: Mapping[str, Any],
) -> torch.Tensor:
    if not bool(config.get("infer_tta", False)):
        return predict_once(model, tensor, roi_size, config)

    outputs = [predict_once(model, tensor, roi_size, config)]
    for dims in ((-1,), (-2,), (-2, -1)):
        flipped = torch.flip(tensor, dims=dims)
        pred = predict_once(model, flipped, roi_size, config)
        outputs.append(torch.flip(pred, dims=dims))
    return torch.stack(outputs, dim=0).mean(dim=0)


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    volume: np.ndarray,
    config: Mapping[str, Any],
    device: torch.device,
) -> np.ndarray:
    tensor = torch.from_numpy(np.ascontiguousarray(volume[None, None]).astype(np.float32)).to(device)
    roi_size = get_infer_roi_size(config)
    warnings.filterwarnings(
        "ignore",
        message="Using a non-tuple sequence for multidimensional indexing.*",
        category=UserWarning,
        module="monai.inferers.utils",
    )
    output = maybe_tta_predict(model, tensor, roi_size, config)
    return output[0, 0].detach().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 3D CycleGAN CBCT-to-CT inference.")
    parser.add_argument("--config", default="./config.json", help="Path to config.json.")
    parser.add_argument("--input", default="D:/Data/cbct/denoise_output.mhd", help="Input .nii/.nii.gz/.mhd/.mha file.")
    # parser.add_argument('--input', type=str, default=r"E:\Data\synthRAD2025_Task2_Train\Task2\TH\2THA005\cbct.mha", help="Path to cbct file")
    parser.add_argument("--output", default=None, help="Output path. Defaults to *_cyclegan3d_sct.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path. Defaults to output_dir/best.pt.")
    parser.add_argument(
        "--roi-size",
        type=int,
        nargs=3,
        metavar=("H", "W", "D"),
        default=None,
        help="Override infer_roi_size_hwd.",
    )
    parser.add_argument("--overlap", type=float, default=None, help="Override infer_overlap.")
    parser.add_argument(
        "--strategy",
        choices=["z_sliding", "sliding_window", "whole"],
        default=None,
        help="Override infer_strategy.",
    )
    parser.add_argument("--tta", action="store_true", help="Enable flip test-time augmentation.")
    args = parser.parse_args()

    config = read_config(args.config)
    if args.roi_size is not None:
        config["infer_roi_size_hwd"] = list(args.roi_size)
    if args.overlap is not None:
        config["infer_overlap"] = float(args.overlap)
    if args.strategy is not None:
        config["infer_strategy"] = args.strategy
    if args.tta:
        config["infer_tta"] = True
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
            "infer_roi_size_hwd",
            "infer_strategy",
            "hu_min",
            "hu_max",
            "infer_resize_xy",
            "infer_sw_batch_size",
            "infer_overlap",
            "infer_blend_mode",
            "infer_sigma_scale",
            "infer_padding_mode",
            "infer_tta",
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
