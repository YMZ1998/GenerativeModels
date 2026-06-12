from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from utils import build_cached_slice_list, read_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize preprocessed CBCT/CT .pt slice cache.")
    parser.add_argument("--config", type=str, default=str(SCRIPT_DIR / "config.json"), help="Path to config.json.")
    parser.add_argument("--split", choices=["train", "val"], default="train", help="Cached split to visualize.")
    parser.add_argument("--case-id", type=str, default=None, help="Only visualize slices from this case id.")
    parser.add_argument("--num-samples", type=int, default=4, help="Number of slices to draw.")
    parser.add_argument("--start-index", type=int, default=0, help="Start index used only with --sequential.")
    parser.add_argument("--sequential", action="store_true", help="Use sequential slices instead of random shuffled slices.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for reproducible random shuffling.")
    parser.add_argument("--output", type=str, default=None, help="Output PNG path.")
    parser.add_argument("--show", action="store_true", help="Also show the figure interactively.")
    parser.add_argument("--dpi", type=int, default=150, help="Saved figure DPI.")
    parser.add_argument("--diff-window", type=float, default=1.0, help="Symmetric display window for CT-CBCT difference.")
    parser.add_argument("--rotate-k", type=int, default=1, help="Display rotation in 90-degree CCW steps. Default: 1.")
    parser.add_argument("--flip-lr", action="store_true", help="Flip display left/right after rotation.")
    parser.add_argument("--flip-ud", action="store_true", help="Flip display up/down after rotation.")
    return parser.parse_args()


def resolve_output_path(config_path: str | Path, output: str | None, split: str, case_id: str | None) -> Path:
    if output:
        path = Path(output)
        if not path.is_absolute():
            path = Path(config_path).resolve().parent / path
        return path

    suffix = f"_{case_id}" if case_id else ""
    return Path(config_path).resolve().parent / "visualizations" / f"preprocessed_{split}{suffix}.png"


def select_files(
    files: Sequence[Path], num_samples: int, start_index: int, sequential: bool, seed: int | None
) -> List[Path]:
    files = list(files)
    if not files:
        raise FileNotFoundError("No cached slices matched the requested filters.")

    num_samples = max(1, min(int(num_samples), len(files)))
    if not sequential:
        rng = random.Random(seed)
        shuffled = list(files)
        rng.shuffle(shuffled)
        return shuffled[:num_samples]

    start_index = max(0, min(int(start_index), max(len(files) - 1, 0)))
    selected = files[start_index : start_index + num_samples]
    if len(selected) < num_samples:
        selected.extend(files[: num_samples - len(selected)])
    return selected


def load_sample(path: Path) -> Dict[str, Any]:
    sample = torch.load(path, map_location="cpu", weights_only=False)
    cbct = sample["cbct"].float().squeeze().numpy()
    print(cbct.shape)
    ct = sample["ct"].float().squeeze().numpy()
    return {
        "path": path,
        "case_id": sample.get("case_id", path.stem.split("_z", 1)[0]),
        "slice_index": int(sample.get("slice_index", -1)),
        "cbct": cbct,
        "ct": ct,
        "diff": ct - cbct,
    }


def orient_image(image: np.ndarray, rotate_k: int, flip_lr: bool, flip_ud: bool) -> np.ndarray:
    oriented = np.rot90(image, k=int(rotate_k) % 4)
    if flip_lr:
        oriented = np.fliplr(oriented)
    if flip_ud:
        oriented = np.flipud(oriented)
    return oriented


def describe(samples: Sequence[Dict[str, Any]]) -> None:
    cbct_values = np.concatenate([sample["cbct"].reshape(-1) for sample in samples])
    ct_values = np.concatenate([sample["ct"].reshape(-1) for sample in samples])
    print(f"Samples: {len(samples)}")
    print(f"CBCT range: {cbct_values.min():.3f} to {cbct_values.max():.3f}; mean={cbct_values.mean():.3f}")
    print(f"CT range:   {ct_values.min():.3f} to {ct_values.max():.3f}; mean={ct_values.mean():.3f}")


def plot_samples(
    samples: Sequence[Dict[str, Any]],
    output_path: Path,
    show: bool,
    dpi: int,
    diff_window: float,
    rotate_k: int,
    flip_lr: bool,
    flip_ud: bool,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required: python -m pip install matplotlib") from exc

    rows = len(samples)
    fig, axes = plt.subplots(rows, 3, figsize=(10, max(2.8, rows * 2.6)), squeeze=False)
    column_titles = ["CBCT normalized", "CT normalized", "CT - CBCT"]

    for col, title in enumerate(column_titles):
        axes[0, col].set_title(title, fontsize=10)

    for row, sample in enumerate(samples):
        title = f"{sample['case_id']} z={sample['slice_index']}"
        cbct = orient_image(sample["cbct"], rotate_k, flip_lr, flip_ud)
        ct = orient_image(sample["ct"], rotate_k, flip_lr, flip_ud)
        diff = orient_image(sample["diff"], rotate_k, flip_lr, flip_ud)

        axes[row, 0].imshow(cbct, cmap="gray", vmin=-1.0, vmax=1.0)
        axes[row, 0].set_ylabel(title, fontsize=8)
        axes[row, 1].imshow(ct, cmap="gray", vmin=-1.0, vmax=1.0)
        axes[row, 2].imshow(diff, cmap="coolwarm", vmin=-float(diff_window), vmax=float(diff_window))
        for col in range(3):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    fig.suptitle("Preprocessed CBCT/CT Slice Cache", fontsize=12)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=int(dpi), bbox_inches="tight")
    print(f"Saved visualization: {output_path}")

    plt.show()


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    files = build_cached_slice_list(config["preprocessed_dir"], args.split)

    if args.case_id:
        files = [path for path in files if path.stem.startswith(f"{args.case_id}_z")]

    selected = select_files(files, args.num_samples, args.start_index, args.sequential, args.seed)
    samples = [load_sample(path) for path in selected]
    describe(samples)

    output_path = resolve_output_path(args.config, args.output, args.split, args.case_id)
    plot_samples(
        samples,
        output_path,
        args.show,
        args.dpi,
        args.diff_window,
        args.rotate_k,
        args.flip_lr,
        args.flip_ud,
    )


if __name__ == "__main__":
    main()
