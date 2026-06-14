from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from utils import build_pair_list, preprocess_volume, print_config_summary, read_config, read_medical_image, set_seed


MEDICAL_PREPROCESS_FORMATS = {"nii.gz", "nii", "mhd", "mha"}


def normalize_preprocess_format(config: Mapping[str, Any]) -> str:
    fmt = str(config.get("preprocess_format", "npy")).lower().lstrip(".")
    if fmt not in {"npy", *MEDICAL_PREPROCESS_FORMATS}:
        raise ValueError("preprocess_format must be one of: npy, nii.gz, nii, mhd, mha.")
    return fmt


def preprocessed_output_path(case_dir: Path, stem: str, config: Mapping[str, Any]) -> Path:
    fmt = normalize_preprocess_format(config)
    if fmt == "npy":
        return case_dir / f"{stem}.npy"
    return case_dir / f"{stem}.{fmt}"


def write_preprocessed_medical(
    volume: np.ndarray,
    reference_image: sitk.Image,
    original_shape_zhw: tuple[int, int, int],
    output_path: Path,
) -> None:
    output_image = sitk.GetImageFromArray(volume.astype(np.float32, copy=False))
    old_spacing = reference_image.GetSpacing()
    old_z, old_h, old_w = original_shape_zhw
    new_z, new_h, new_w = volume.shape
    new_spacing = (
        old_spacing[0] * (float(old_w) / float(new_w)),
        old_spacing[1] * (float(old_h) / float(new_h)),
        old_spacing[2] * (float(old_z) / float(new_z)),
    )
    output_image.SetSpacing(new_spacing)
    output_image.SetOrigin(reference_image.GetOrigin())
    output_image.SetDirection(reference_image.GetDirection())
    sitk.WriteImage(output_image, str(output_path))


def _save_preprocessed_volume(
    source_path: str,
    output_path: Path,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    array, image = read_medical_image(source_path)
    volume = preprocess_volume(array, config)
    fmt = normalize_preprocess_format(config)
    dtype = np.dtype(str(config.get("preprocess_dtype", "float32")))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "npy":
        np.save(output_path, volume.astype(dtype, copy=False))
        stored_dtype = str(dtype)
    else:
        write_preprocessed_medical(volume, image, tuple(int(v) for v in array.shape), output_path)
        stored_dtype = "float32"
    return {
        "source": source_path,
        "original_shape_zhw": list(array.shape),
        "preprocessed_shape_zhw": list(volume.shape),
        "original_spacing_xyz": list(image.GetSpacing()),
        "original_origin_xyz": list(image.GetOrigin()),
        "original_direction": list(image.GetDirection()),
        "stored_format": fmt,
        "stored_dtype": stored_dtype,
    }


def preprocess_split(config: Mapping[str, Any], split: str) -> int:
    pairs = build_pair_list(config, split)
    split_dir = Path(str(config["preprocessed_dir"])) / split
    overwrite = bool(config.get("overwrite_preprocessed", True))

    if overwrite and split_dir.exists():
        shutil.rmtree(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    manifest_cases = []
    progress = tqdm(pairs, desc=f"Preprocess {split}", unit="case", dynamic_ncols=True)
    for pair in progress:
        case_id = str(pair["case_id"])
        case_dir = split_dir / case_id
        cbct_out = preprocessed_output_path(case_dir, "cbct", config)
        ct_out = preprocessed_output_path(case_dir, "ct", config)
        progress.set_postfix(case=case_id)

        if not overwrite and cbct_out.is_file() and ct_out.is_file():
            cbct_meta = {"source": pair["cbct"], "skipped": True}
            ct_meta = {"source": pair["ct"], "skipped": True}
        else:
            cbct_meta = _save_preprocessed_volume(pair["cbct"], cbct_out, config)
            ct_meta = _save_preprocessed_volume(pair["ct"], ct_out, config)

        meta = {
            "case_id": case_id,
            "cbct": cbct_meta,
            "ct": ct_meta,
        }
        with (case_dir / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        manifest_cases.append({"case_id": case_id, "cbct": str(cbct_out), "ct": str(ct_out)})
        if "preprocessed_shape_zhw" in cbct_meta:
            progress.set_postfix(case=case_id, slices=cbct_meta["preprocessed_shape_zhw"][0])

    with (split_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump({"split": split, "cases": manifest_cases}, f, indent=2)
    return len(manifest_cases)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess paired CBCT/CT volumes for 3D CycleGAN.")
    parser.add_argument("--config", default="./config.json", help="Path to config.json.")
    args = parser.parse_args()

    config = read_config(args.config)
    set_seed(int(config["seed"]))
    print_config_summary(
        config,
        [
            "data_root",
            "data_layout",
            "preprocessed_dir",
            "xy_size",
            "hu_min",
            "hu_max",
            "preprocess_format",
            "preprocess_dtype",
            "overwrite_preprocessed",
        ],
        title="3D CycleGAN Preprocess",
    )

    train_count = preprocess_split(config, "train")
    val_count = preprocess_split(config, "val")
    manifest = {
        "data_root": config["data_root"],
        "data_layout": config["data_layout"],
        "xy_size": config["xy_size"],
        "hu_min": config["hu_min"],
        "hu_max": config["hu_max"],
        "train_cases": train_count,
        "val_cases": val_count,
    }
    manifest_path = Path(str(config["preprocessed_dir"])) / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Done. Wrote {train_count} train cases and {val_count} val cases to {config['preprocessed_dir']}")


if __name__ == "__main__":
    main()
