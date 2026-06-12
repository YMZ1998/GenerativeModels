from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from generative.networks.schedulers import DDPMScheduler  # noqa: E402
from utils import (  # noqa: E402
    build_volume_transform,
    create_model,
    crop_or_pad_slice_2d,
    denormalize_hu,
    load_cbct_volume,
    load_checkpoint,
    print_config_summary,
    read_config,
    save_nifti,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CBCT-to-CT inference with a trained 2D conditional DDPM.")
    parser.add_argument("--config", type=str, default=str(SCRIPT_DIR / "config.json"), help="Path to config.json.")
    parser.add_argument("--input", type=str, required=True, help="Input CBCT NIfTI path.")
    parser.add_argument("--output", type=str, required=True, help="Output synthetic CT NIfTI path.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path.")
    parser.add_argument("--device", type=str, default=None, help="Override device, for example cuda or cpu.")
    parser.add_argument("--batch-size", type=int, default=4, help="Number of axial slices sampled at once.")
    return parser.parse_args()


@torch.no_grad()
def sample_batch(
    model: torch.nn.Module,
    scheduler: DDPMScheduler,
    cbct_batch: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    current = torch.randn_like(cbct_batch, device=device)
    for timestep in scheduler.timesteps:
        t_int = int(timestep.item()) if torch.is_tensor(timestep) else int(timestep)
        timesteps = torch.full((cbct_batch.shape[0],), t_int, device=device, dtype=torch.long)
        model_input = torch.cat((cbct_batch, current), dim=1)
        model_output = model(x=model_input, timesteps=timesteps)
        current, _ = scheduler.step(model_output, t_int, current)
    return current


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    set_seed(int(config["seed"]))

    device_name = args.device or config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    print_config_summary(
        config,
        [
            "data_root",
            "output_dir",
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
        ],
        title="Inference configuration",
    )
    print(f"input: {args.input}")
    print(f"output: {args.output}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"inference_batch_size: {args.batch_size}")
    print("")
    model = create_model(config).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    scheduler = DDPMScheduler(num_train_timesteps=int(config["num_train_timesteps"]))
    scheduler.set_timesteps(num_inference_steps=int(config["num_inference_steps"]), device=device)

    transform = build_volume_transform(["cbct"], config.get("spacing"))
    cbct_volume, affine = load_cbct_volume(args.input, config, transform=transform)
    depth = cbct_volume.shape[-1]
    target_h, target_w = int(config["spatial_size"][0]), int(config["spatial_size"][1])
    output_volume = torch.empty((target_h, target_w, depth), dtype=torch.float32)

    print(f"Device: {device}")
    print(f"Input volume shape after preprocessing: {tuple(cbct_volume.shape)}")
    print(f"Sampling {depth} axial slices with {len(scheduler.timesteps)} denoising steps")

    batch_size = int(args.batch_size)
    for start in range(0, depth, batch_size):
        end = min(start + batch_size, depth)
        slices = [crop_or_pad_slice_2d(cbct_volume[:, :, :, z], config["spatial_size"]) for z in range(start, end)]
        cbct_batch = torch.stack(slices, dim=0).to(device=device)
        sampled = sample_batch(model, scheduler, cbct_batch, device)
        sampled_hu = denormalize_hu(sampled[:, 0].cpu(), config["hu_min"], config["hu_max"])
        for offset, z in enumerate(range(start, end)):
            output_volume[:, :, z] = sampled_hu[offset]
        print(f"Finished slices {start + 1}-{end}/{depth}")

    save_nifti(output_volume.numpy().astype(np.float32), affine, args.output)
    print(f"Saved synthetic CT: {args.output}")


if __name__ == "__main__":
    main()
