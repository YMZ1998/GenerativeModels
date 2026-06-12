from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import torch
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning, module="monai")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from utils import (  # noqa: E402
    build_pair_list,
    crop_or_pad_pair_2d,
    load_mask_volume,
    load_pair_volume,
    print_config_summary,
    read_config,
    set_seed,
    to_plain_tensor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess paired CBCT/CT NIfTI volumes into 2D .pt slice cache.")
    parser.add_argument("--config", type=str, default=str(SCRIPT_DIR / "config.json"), help="Path to config.json.")
    parser.add_argument(
        "--split",
        choices=["all", "train", "val"],
        default="all",
        help="Which deterministic split to preprocess.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override config['preprocessed_dir']; useful for smoke tests.",
    )
    parser.add_argument("--max-cases", type=int, default=None, help="Optional limit per split for quick checks.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip cases that already have cached slices. By default, existing case caches are overwritten.",
    )
    return parser.parse_args()


def select_slice_indices(mask: torch.Tensor | None, depth: int, config: Mapping[str, Any]) -> List[int]:
    stride = max(1, int(config.get("slice_stride", 1)))
    indices = list(range(0, depth, stride))
    min_mask_fraction = float(config.get("min_mask_fraction", 0.0))

    if mask is None or min_mask_fraction <= 0:
        return indices

    mask_depth = mask.shape[-1]
    usable_depth = min(depth, mask_depth)
    selected = []
    for z in range(0, usable_depth, stride):
        mask_fraction = float(mask[:, :, :, z].float().mean().item())
        if mask_fraction >= min_mask_fraction:
            selected.append(z)

    return selected or indices


def remove_case_cache(split_dir: Path, case_id: str) -> None:
    for path in split_dir.glob(f"{case_id}_z*.pt"):
        path.unlink()


def preprocess_pair(pair: Mapping[str, str], split_dir: Path, config: Mapping[str, Any], overwrite: bool) -> int:
    if overwrite:
        remove_case_cache(split_dir, pair["case_id"])
    elif any(split_dir.glob(f"{pair['case_id']}_z*.pt")):
        return 0

    cbct, ct = load_pair_volume(pair, config)
    mask = load_mask_volume(pair, config)
    if mask is not None:
        target_shape = [min(mask.shape[i], cbct.shape[i]) for i in range(1, 4)]
        mask = mask[:, : target_shape[0], : target_shape[1], : target_shape[2]]
        cbct = cbct[:, : target_shape[0], : target_shape[1], : target_shape[2]]
        ct = ct[:, : target_shape[0], : target_shape[1], : target_shape[2]]

    depth = cbct.shape[-1]
    slice_indices = select_slice_indices(mask, depth, config)
    written = 0

    for z in slice_indices:
        cbct_slice = cbct[:, :, :, z]
        ct_slice = ct[:, :, :, z]
        cbct_slice, ct_slice = crop_or_pad_pair_2d(cbct_slice, ct_slice, config["spatial_size"], random_crop=False)
        output_path = split_dir / f"{pair['case_id']}_z{z:04d}.pt"
        torch.save(
            {
                "cbct": to_plain_tensor(cbct_slice),
                "ct": to_plain_tensor(ct_slice),
                "case_id": pair["case_id"],
                "slice_index": int(z),
                "source_cbct": pair["cbct"],
                "source_ct": pair["ct"],
            },
            output_path,
        )
        written += 1

    return written


def preprocess_split(config: Mapping[str, Any], split: str, output_dir: Path, max_cases: int | None, overwrite: bool) -> Dict[str, Any]:
    pairs = build_pair_list(config, split)
    if max_cases is not None:
        pairs = pairs[: max(0, max_cases)]

    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    total_slices = 0
    processed_cases = 0
    progress = tqdm(
        pairs,
        desc=f"Preprocess {split}",
        unit="case",
        dynamic_ncols=True,
        file=sys.stdout,
        leave=False,
        mininterval=1.0,
        maxinterval=5.0,
    )
    for index, pair in enumerate(progress, start=1):
        written = preprocess_pair(pair, split_dir, config, overwrite=overwrite)
        total_slices += written
        processed_cases += 1
        progress.set_postfix(case=pair["case_id"], slices=written, refresh=False)

    print(f"[{split}] processed {processed_cases} cases, wrote {total_slices} new slices to {split_dir}")
    return {"split": split, "cases": processed_cases, "new_slices": total_slices, "directory": str(split_dir)}


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    set_seed(int(config["seed"]))

    output_dir = Path(args.output_dir or config["preprocessed_dir"])
    # os.remove(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits: Sequence[str] = ("train", "val") if args.split == "all" else (args.split,)
    print_config_summary(
        config,
        [
            "data_root",
            "data_layout",
            "preprocessed_dir",
            "spatial_size",
            "spacing",
            "hu_min",
            "hu_max",
            "val_fraction",
            "slice_stride",
            "min_mask_fraction",
        ],
        title="Preprocess configuration",
    )
    print(f"output_dir: {output_dir}")
    print(f"splits: {', '.join(splits)}")
    print(f"overwrite_existing: {not args.skip_existing}")
    if args.max_cases is not None:
        print(f"max_cases_per_split: {args.max_cases}")
    print("")

    summaries = [
        preprocess_split(config, split, output_dir, max_cases=args.max_cases, overwrite=not args.skip_existing)
        for split in splits
    ]

    manifest = {
        "data_root": config["data_root"],
        "data_layout": config["data_layout"],
        "spatial_size": config["spatial_size"],
        "spacing": config["spacing"],
        "hu_min": config["hu_min"],
        "hu_max": config["hu_max"],
        "slice_stride": config["slice_stride"],
        "min_mask_fraction": config["min_mask_fraction"],
        "summaries": summaries,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Preprocessing complete: {output_dir}")


if __name__ == "__main__":
    main()
