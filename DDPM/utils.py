from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from monai.transforms import Compose, EnsureChannelFirstd, EnsureTyped, LoadImaged, Orientationd, Spacingd
from torch.utils.data import Dataset

from generative.networks.nets import DiffusionModelUNet


DEFAULT_CONFIG: Dict[str, Any] = {
    "data_root": "data/cbct_ct",
    "data_layout": "paired_split_dirs",
    "output_dir": "DDPM/runs/baseline",
    "preprocessed_dir": "DDPM/preprocessed/pelvis",
    "use_preprocessed": False,
    "seed": 42,
    "device": None,
    "spatial_size": [256, 256],
    "spacing": [1.0, 1.0, 2.5],
    "hu_min": -1000.0,
    "hu_max": 2000.0,
    "batch_size": 4,
    "num_workers": 4,
    "epochs": 100,
    "lr": 1e-4,
    "val_interval": 1,
    "save_interval": 10,
    "num_train_timesteps": 1000,
    "num_inference_steps": 100,
    "model_channels": [64, 128, 256],
    "attention_levels": [False, False, True],
    "num_res_blocks": 1,
    "num_head_channels": 64,
    "norm_num_groups": 32,
    "samples_per_volume": 32,
    "val_slices_per_volume": 8,
    "val_fraction": 0.1,
    "slice_stride": 1,
    "min_mask_fraction": 0.001,
    "visdom_enabled": False,
    "visdom_server": "http://localhost",
    "visdom_port": 8097,
    "visdom_env": "ddpm_pelvis",
    "visdom_interval": 1,
    "visdom_num_images": 4,
    "visdom_num_inference_steps": 20,
    "show_batch_progress": True,
    "amp": True,
    "cache_data": False,
}


def _resolve_relative_artifact_paths(config: Dict[str, Any], config_path: Path) -> None:
    base_dir = config_path.resolve().parent
    for key in ("output_dir", "preprocessed_dir"):
        path = Path(config[key])
        if not path.is_absolute():
            config[key] = str((base_dir / path).resolve())


def read_config(path: str | Path) -> Dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8-sig") as f:
        user_config = json.load(f)

    config = dict(DEFAULT_CONFIG)
    config.update(user_config)
    validate_config(config)
    _resolve_relative_artifact_paths(config, config_path)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    spatial_size = config["spatial_size"]
    if len(spatial_size) != 2:
        raise ValueError("config['spatial_size'] must contain [height, width].")

    spacing = config.get("spacing")
    if spacing is not None and len(spacing) != 3:
        raise ValueError("config['spacing'] must contain [x, y, z] or be null.")

    if len(config["model_channels"]) != len(config["attention_levels"]):
        raise ValueError("model_channels and attention_levels must have the same length.")

    if float(config["hu_min"]) >= float(config["hu_max"]):
        raise ValueError("hu_min must be less than hu_max.")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_config_summary(config: Mapping[str, Any], keys: Sequence[str], title: str = "Config") -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for key in keys:
        print(f"{key}: {config.get(key)}")
    print("")


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed + worker_id)
    np.random.seed(worker_seed + worker_id)


def strip_nifti_suffix(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


def find_nifti_files(folder: Path) -> List[Path]:
    return sorted(list(folder.glob("*.nii")) + list(folder.glob("*.nii.gz")))


def _build_split_dir_pairs(root: Path, split: str) -> List[Dict[str, str]]:
    cbct_dir = root / split / "cbct"
    ct_dir = root / split / "ct"

    if not cbct_dir.is_dir():
        raise FileNotFoundError(f"CBCT directory not found: {cbct_dir}")
    if not ct_dir.is_dir():
        raise FileNotFoundError(f"CT directory not found: {ct_dir}")

    cbct_files = {strip_nifti_suffix(path): path for path in find_nifti_files(cbct_dir)}
    ct_files = {strip_nifti_suffix(path): path for path in find_nifti_files(ct_dir)}
    common_ids = sorted(set(cbct_files) & set(ct_files))

    if not common_ids:
        raise FileNotFoundError(
            f"No paired NIfTI files found for split '{split}'. Expected matching names under {cbct_dir} and {ct_dir}."
        )

    return [
        {"case_id": case_id, "cbct": str(cbct_files[case_id]), "ct": str(ct_files[case_id])}
        for case_id in common_ids
    ]


def _build_synthrad_case_pairs(root: Path) -> List[Dict[str, str]]:
    if not root.is_dir():
        raise FileNotFoundError(f"SynthRAD pelvis root not found: {root}")

    pairs = []
    for case_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if case_dir.name.lower() == "overview":
            continue
        cbct = case_dir / "cbct.nii.gz"
        ct = case_dir / "ct.nii.gz"
        if cbct.is_file() and ct.is_file():
            pair = {"case_id": case_dir.name, "cbct": str(cbct), "ct": str(ct)}
            mask = case_dir / "mask.nii.gz"
            if mask.is_file():
                pair["mask"] = str(mask)
            pairs.append(pair)

    if not pairs:
        raise FileNotFoundError(
            f"No SynthRAD case folders found under {root}. Expected case/cbct.nii.gz and case/ct.nii.gz."
        )
    return pairs


def _split_pairs(pairs: Sequence[Dict[str, str]], config: Mapping[str, Any], split: str) -> List[Dict[str, str]]:
    if split not in {"train", "val"}:
        raise ValueError("split must be 'train' or 'val'.")

    val_fraction = float(config.get("val_fraction", 0.1))
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1 for synthrad_case_dirs layout.")

    generator = random.Random(int(config["seed"]))
    shuffled = list(pairs)
    generator.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_fraction)))
    val_ids = {pair["case_id"] for pair in sorted(shuffled[:val_count], key=lambda item: item["case_id"])}

    if split == "val":
        return [pair for pair in pairs if pair["case_id"] in val_ids]
    return [pair for pair in pairs if pair["case_id"] not in val_ids]


def build_pair_list(data_root_or_config: str | Path | Mapping[str, Any], split: str) -> List[Dict[str, str]]:
    if isinstance(data_root_or_config, Mapping):
        config = data_root_or_config
        root = Path(config["data_root"])
        layout = str(config.get("data_layout", "paired_split_dirs"))
    else:
        config = DEFAULT_CONFIG
        root = Path(data_root_or_config)
        layout = "paired_split_dirs"

    if layout == "paired_split_dirs":
        return _build_split_dir_pairs(root, split)
    if layout == "synthrad_case_dirs":
        return _split_pairs(_build_synthrad_case_pairs(root), config, split)

    raise ValueError("data_layout must be 'paired_split_dirs' or 'synthrad_case_dirs'.")


def normalize_hu(volume: torch.Tensor, hu_min: float, hu_max: float) -> torch.Tensor:
    volume = torch.clamp(volume.float(), min=float(hu_min), max=float(hu_max))
    volume = (volume - float(hu_min)) / (float(hu_max) - float(hu_min))
    return volume * 2.0 - 1.0


def denormalize_hu(volume: torch.Tensor, hu_min: float, hu_max: float) -> torch.Tensor:
    volume = torch.clamp(volume.float(), min=-1.0, max=1.0)
    volume = (volume + 1.0) * 0.5
    return volume * (float(hu_max) - float(hu_min)) + float(hu_min)


def build_volume_transform(
    keys: Iterable[str], spacing: Optional[Sequence[float]], modes: Optional[Mapping[str, str]] = None
) -> Compose:
    key_list = list(keys)
    transforms = [
        LoadImaged(keys=key_list, image_only=False),
        EnsureChannelFirstd(keys=key_list),
    ]
    if spacing is not None:
        spacing_modes = tuple((modes or {}).get(key, "bilinear") for key in key_list)
        transforms.extend(
            [
                Orientationd(keys=key_list, axcodes="RAS"),
                Spacingd(keys=key_list, pixdim=tuple(float(v) for v in spacing), mode=spacing_modes),
            ]
        )
    transforms.append(EnsureTyped(keys=key_list, dtype=torch.float32))
    return Compose(transforms)


def _center_crop_3d(volume: torch.Tensor, target_shape: Sequence[int]) -> torch.Tensor:
    slices = [slice(None)]
    for axis, target in enumerate(target_shape, start=1):
        current = volume.shape[axis]
        start = max((current - int(target)) // 2, 0)
        slices.append(slice(start, start + int(target)))
    return volume[tuple(slices)]


def _match_volume_shapes(cbct: torch.Tensor, ct: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if cbct.shape == ct.shape:
        return cbct, ct
    target_shape = [min(cbct.shape[i], ct.shape[i]) for i in range(1, 4)]
    return _center_crop_3d(cbct, target_shape), _center_crop_3d(ct, target_shape)


def _extract_affine(image: torch.Tensor, data: Mapping[str, Any], key: str) -> np.ndarray:
    affine = getattr(image, "affine", None)
    if affine is None:
        meta = data.get(f"{key}_meta_dict", {})
        affine = meta.get("affine")
    if affine is None:
        return np.eye(4, dtype=np.float32)
    if torch.is_tensor(affine):
        affine = affine.detach().cpu().numpy()
    return np.asarray(affine, dtype=np.float32)


def load_pair_volume(
    pair: Mapping[str, str], config: Mapping[str, Any], transform: Optional[Compose] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    transform = transform or build_volume_transform(["cbct", "ct"], config.get("spacing"))
    data = transform({"cbct": pair["cbct"], "ct": pair["ct"]})
    cbct = normalize_hu(data["cbct"], config["hu_min"], config["hu_max"])
    ct = normalize_hu(data["ct"], config["hu_min"], config["hu_max"])
    return _match_volume_shapes(cbct, ct)


def load_cbct_volume(
    cbct_path: str | Path, config: Mapping[str, Any], transform: Optional[Compose] = None
) -> Tuple[torch.Tensor, np.ndarray]:
    transform = transform or build_volume_transform(["cbct"], config.get("spacing"))
    data = transform({"cbct": str(cbct_path)})
    cbct = normalize_hu(data["cbct"], config["hu_min"], config["hu_max"])
    affine = _extract_affine(data["cbct"], data, "cbct")
    return cbct, affine


def load_mask_volume(pair: Mapping[str, str], config: Mapping[str, Any]) -> Optional[torch.Tensor]:
    mask_path = pair.get("mask")
    if not mask_path:
        return None

    transform = build_volume_transform(["mask"], config.get("spacing"), modes={"mask": "nearest"})
    data = transform({"mask": mask_path})
    return (data["mask"].float() > 0).float()


def _pad_to_at_least(tensor: torch.Tensor, spatial_size: Sequence[int], pad_value: float = -1.0) -> torch.Tensor:
    target_h, target_w = int(spatial_size[0]), int(spatial_size[1])
    h, w = tensor.shape[-2:]
    pad_h = max(target_h - h, 0)
    pad_w = max(target_w - w, 0)
    if pad_h == 0 and pad_w == 0:
        return tensor

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    return F.pad(tensor, (left, right, top, bottom), value=pad_value)


def crop_or_pad_pair_2d(
    cbct: torch.Tensor, ct: torch.Tensor, spatial_size: Sequence[int], random_crop: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    cbct = _pad_to_at_least(cbct, spatial_size)
    ct = _pad_to_at_least(ct, spatial_size)

    target_h, target_w = int(spatial_size[0]), int(spatial_size[1])
    h, w = cbct.shape[-2:]
    max_top = max(h - target_h, 0)
    max_left = max(w - target_w, 0)
    if random_crop:
        top = random.randint(0, max_top) if max_top else 0
        left = random.randint(0, max_left) if max_left else 0
    else:
        top = max_top // 2
        left = max_left // 2

    return (
        cbct[..., top : top + target_h, left : left + target_w],
        ct[..., top : top + target_h, left : left + target_w],
    )


def crop_or_pad_slice_2d(slice_tensor: torch.Tensor, spatial_size: Sequence[int]) -> torch.Tensor:
    padded = _pad_to_at_least(slice_tensor, spatial_size)
    target_h, target_w = int(spatial_size[0]), int(spatial_size[1])
    h, w = padded.shape[-2:]
    top = max(h - target_h, 0) // 2
    left = max(w - target_w, 0) // 2
    return padded[..., top : top + target_h, left : left + target_w]


def to_plain_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu().clone().as_subclass(torch.Tensor).contiguous()


class NiftiSlicePairDataset(Dataset):
    def __init__(self, pairs: Sequence[Mapping[str, str]], config: Mapping[str, Any], training: bool) -> None:
        self.pairs = list(pairs)
        self.config = config
        self.training = training
        self.samples_per_volume = int(config["samples_per_volume"] if training else config["val_slices_per_volume"])
        self.transform = build_volume_transform(["cbct", "ct"], config.get("spacing"))
        self.cache_data = bool(config.get("cache_data", False))
        self._cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    def __len__(self) -> int:
        return len(self.pairs) * self.samples_per_volume

    def _load_pair(self, pair_index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cache_data and pair_index in self._cache:
            return self._cache[pair_index]

        cbct, ct = load_pair_volume(self.pairs[pair_index], self.config, transform=self.transform)
        if self.cache_data:
            self._cache[pair_index] = (cbct, ct)
        return cbct, ct

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if self.training:
            pair_index = index % len(self.pairs)
            slice_index = None
        else:
            pair_index = index // self.samples_per_volume
            slice_index = index % self.samples_per_volume

        cbct, ct = self._load_pair(pair_index)
        depth = cbct.shape[-1]
        if depth <= 0:
            raise ValueError(f"Empty z dimension for case: {self.pairs[pair_index]['case_id']}")

        if self.training:
            z = random.randint(0, depth - 1)
        else:
            z = int(round((slice_index + 1) * (depth - 1) / (self.samples_per_volume + 1)))

        cbct_slice = cbct[:, :, :, z]
        ct_slice = ct[:, :, :, z]
        cbct_slice, ct_slice = crop_or_pad_pair_2d(
            cbct_slice, ct_slice, self.config["spatial_size"], random_crop=self.training
        )

        return {
            "cbct": cbct_slice.contiguous(),
            "ct": ct_slice.contiguous(),
            "case_id": self.pairs[pair_index]["case_id"],
            "slice_index": z,
        }


def build_cached_slice_list(preprocessed_dir: str | Path, split: str) -> List[Path]:
    split_dir = Path(preprocessed_dir) / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Preprocessed split directory not found: {split_dir}")

    files = sorted(split_dir.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No cached .pt slices found in {split_dir}. Run DDPM/preprocess.py first.")
    return files


class CachedSlicePairDataset(Dataset):
    def __init__(self, preprocessed_dir: str | Path, split: str) -> None:
        self.files = build_cached_slice_list(preprocessed_dir, split)
        self.case_ids = sorted({path.stem.split("_z", 1)[0] for path in self.files})

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = torch.load(self.files[index], map_location="cpu", weights_only=False)
        return {
            "cbct": sample["cbct"].float(),
            "ct": sample["ct"].float(),
            "case_id": sample.get("case_id", self.files[index].stem.split("_z", 1)[0]),
            "slice_index": int(sample.get("slice_index", -1)),
        }


def create_model(config: Mapping[str, Any]) -> DiffusionModelUNet:
    num_head_channels = config["num_head_channels"]
    if isinstance(num_head_channels, list):
        num_head_channels = tuple(int(v) for v in num_head_channels)

    return DiffusionModelUNet(
        spatial_dims=2,
        in_channels=2,
        out_channels=1,
        num_channels=tuple(int(v) for v in config["model_channels"]),
        attention_levels=tuple(bool(v) for v in config["attention_levels"]),
        num_res_blocks=config["num_res_blocks"],
        num_head_channels=num_head_channels,
        norm_num_groups=int(config.get("norm_num_groups", 32)),
        with_conditioning=False,
    )


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    best_val_loss: float,
    config: Mapping[str, Any],
) -> None:
    checkpoint = {
        "model": model.state_dict(),
        "epoch": int(epoch),
        "best_val_loss": float(best_val_loss),
        "config": dict(config),
    }
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location=map_location)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    if optimizer is not None and isinstance(checkpoint, dict) and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint if isinstance(checkpoint, dict) else {"model": state_dict}


def save_nifti(volume_hwd: np.ndarray, affine: np.ndarray, output_path: str | Path) -> None:
    import nibabel as nib

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(volume_hwd.astype(np.float32), affine)
    nib.save(image, str(output_path))
