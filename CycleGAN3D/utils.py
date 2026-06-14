from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


DEFAULT_CONFIG: Dict[str, Any] = {
    "data_root": "E:/CBCT2CT/SynthRAD/Task2/pelvis",
    "data_layout": "synthrad_case_dirs",
    "output_dir": "CycleGAN3D/runs/pelvis_cyclegan3d",
    "preprocessed_dir": "CycleGAN3D/preprocessed/pelvis",
    "use_preprocessed": True,
    "overwrite_preprocessed": True,
    "auto_resume": True,
    "resume_checkpoint": None,
    "seed": 42,
    "device": "cuda:0",
    "xy_size": [512, 512],
    "patch_size_hwd": [128, 128, 64],
    "hu_min": -1000.0,
    "hu_max": 2000.0,
    "preprocess_format": "npy",
    "preprocess_dtype": "float16",
    "batch_size": 1,
    "num_workers": 4,
    "epochs": 100,
    "train_steps_per_epoch": 500,
    "val_steps_per_epoch": 50,
    "lr_g": 2e-4,
    "lr_d": 2e-4,
    "beta1": 0.5,
    "beta2": 0.999,
    "lambda_cycle": 10.0,
    "lambda_identity": 5.0,
    "pool_size": 25,
    "generator_channels": 16,
    "generator_res_blocks": 4,
    "discriminator_channels": 16,
    "discriminator_layers": 3,
    "val_fraction": 0.1,
    "val_interval": 1,
    "save_interval": 10,
    "log_interval": 10,
    "show_batch_progress": True,
    "amp": True,
    "cache_data": False,
    "visdom_enabled": False,
    "visdom_server": "http://localhost",
    "visdom_port": 8097,
    "visdom_env": "cyclegan3d_pelvis",
    "visdom_interval": 1,
    "visdom_num_images": 6,
    "visdom_rotate_k": 1,
    "visdom_flip_lr": False,
    "visdom_flip_ud": False,
    "infer_sw_batch_size": 1,
    "infer_overlap": 0.25,
    "infer_resize_xy": True,
    "output_dtype": "same",
}

MEDICAL_SUFFIXES = (".nii.gz", ".nii", ".mhd", ".mha")


def read_config(path: str | Path) -> Dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8-sig") as f:
        user_config = json.load(f)

    config = dict(DEFAULT_CONFIG)
    config.update(user_config)
    validate_config(config)
    _resolve_relative_paths(config, config_path)
    return config


def _resolve_relative_paths(config: Dict[str, Any], config_path: Path) -> None:
    base_dir = config_path.resolve().parent
    for key in ("output_dir", "preprocessed_dir"):
        path = Path(str(config[key]))
        if not path.is_absolute():
            config[key] = str((base_dir / path).resolve())

    data_root = Path(str(config["data_root"]))
    if not data_root.is_absolute():
        config["data_root"] = str((base_dir / data_root).resolve())

    resume_checkpoint = config.get("resume_checkpoint")
    if resume_checkpoint:
        text = str(resume_checkpoint).strip()
        if text.lower() not in ("latest", "best"):
            path = Path(text)
            if not path.is_absolute():
                config["resume_checkpoint"] = str((base_dir / path).resolve())


def validate_config(config: Mapping[str, Any]) -> None:
    xy_size = config["xy_size"]
    patch_size = config["patch_size_hwd"]
    if len(xy_size) != 2:
        raise ValueError("config['xy_size'] must contain [height, width].")
    if len(patch_size) != 3:
        raise ValueError("config['patch_size_hwd'] must contain [height, width, depth].")
    if any(int(v) <= 0 for v in xy_size + patch_size):
        raise ValueError("xy_size and patch_size_hwd values must be positive.")
    if any(int(v) % 4 != 0 for v in patch_size):
        raise ValueError("patch_size_hwd values must be divisible by 4 for the 3D generator.")
    if float(config["hu_min"]) >= float(config["hu_max"]):
        raise ValueError("hu_min must be less than hu_max.")
    if int(config["train_steps_per_epoch"]) <= 0:
        raise ValueError("train_steps_per_epoch must be positive.")
    if int(config["val_steps_per_epoch"]) <= 0:
        raise ValueError("val_steps_per_epoch must be positive.")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed + worker_id)
    np.random.seed(worker_seed + worker_id)


def print_config_summary(config: Mapping[str, Any], keys: Sequence[str], title: str = "Config") -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for key in keys:
        print(f"{key}: {config.get(key)}")
    print("")


def hwd_to_dhw(patch_size_hwd: Sequence[int]) -> Tuple[int, int, int]:
    height, width, depth = [int(v) for v in patch_size_hwd]
    return depth, height, width


def strip_medical_suffix(path: Path) -> str:
    name = path.name
    lower = name.lower()
    for suffix in MEDICAL_SUFFIXES:
        if lower.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def is_medical_file(path: Path) -> bool:
    lower = path.name.lower()
    return any(lower.endswith(suffix) for suffix in MEDICAL_SUFFIXES)


def find_medical_files(folder: Path) -> List[Path]:
    return sorted(path for path in folder.iterdir() if path.is_file() and is_medical_file(path))


def _first_existing(case_dir: Path, names: Iterable[str]) -> Optional[Path]:
    for name in names:
        path = case_dir / name
        if path.is_file():
            return path
    return None


def _build_split_dir_pairs(root: Path, split: str) -> List[Dict[str, str]]:
    cbct_dir = root / split / "cbct"
    ct_dir = root / split / "ct"
    if not cbct_dir.is_dir():
        raise FileNotFoundError(f"CBCT directory not found: {cbct_dir}")
    if not ct_dir.is_dir():
        raise FileNotFoundError(f"CT directory not found: {ct_dir}")

    cbct_files = {strip_medical_suffix(path): path for path in find_medical_files(cbct_dir)}
    ct_files = {strip_medical_suffix(path): path for path in find_medical_files(ct_dir)}
    common_ids = sorted(set(cbct_files) & set(ct_files))
    if not common_ids:
        raise FileNotFoundError(f"No paired CBCT/CT files found under {cbct_dir} and {ct_dir}.")

    return [{"case_id": case_id, "cbct": str(cbct_files[case_id]), "ct": str(ct_files[case_id])} for case_id in common_ids]


def _build_synthrad_case_pairs(root: Path) -> List[Dict[str, str]]:
    if not root.is_dir():
        raise FileNotFoundError(f"SynthRAD root not found: {root}")

    pairs: List[Dict[str, str]] = []
    for case_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if case_dir.name.lower() == "overview":
            continue
        cbct = _first_existing(case_dir, ("cbct.nii.gz", "cbct.nii", "cbct.mhd", "cbct.mha"))
        ct = _first_existing(case_dir, ("ct.nii.gz", "ct.nii", "ct.mhd", "ct.mha"))
        if cbct is not None and ct is not None:
            pair = {"case_id": case_dir.name, "cbct": str(cbct), "ct": str(ct)}
            mask = _first_existing(case_dir, ("mask.nii.gz", "mask.nii", "mask.mhd", "mask.mha"))
            if mask is not None:
                pair["mask"] = str(mask)
            pairs.append(pair)

    if not pairs:
        raise FileNotFoundError(f"No SynthRAD case folders found under {root}.")
    return pairs


def _split_pairs(pairs: Sequence[Dict[str, str]], config: Mapping[str, Any], split: str) -> List[Dict[str, str]]:
    if split not in {"train", "val"}:
        raise ValueError("split must be 'train' or 'val'.")
    if len(pairs) == 1:
        return list(pairs)

    val_fraction = float(config.get("val_fraction", 0.1))
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1.")

    shuffled = list(pairs)
    random.Random(int(config["seed"])).shuffle(shuffled)
    val_count = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * val_fraction))))
    val_ids = {pair["case_id"] for pair in shuffled[:val_count]}
    if split == "val":
        return [pair for pair in pairs if pair["case_id"] in val_ids]
    return [pair for pair in pairs if pair["case_id"] not in val_ids]


def build_pair_list(config: Mapping[str, Any], split: str) -> List[Dict[str, str]]:
    root = Path(str(config["data_root"]))
    layout = str(config.get("data_layout", "synthrad_case_dirs"))
    if layout == "paired_split_dirs":
        return _build_split_dir_pairs(root, split)
    if layout == "synthrad_case_dirs":
        return _split_pairs(_build_synthrad_case_pairs(root), config, split)
    raise ValueError("data_layout must be 'synthrad_case_dirs' or 'paired_split_dirs'.")


def build_preprocessed_pair_list(config: Mapping[str, Any], split: str) -> List[Dict[str, str]]:
    split_dir = Path(str(config["preprocessed_dir"])) / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Preprocessed split not found: {split_dir}. Run preprocess.py first.")

    pairs: List[Dict[str, str]] = []
    for case_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        cbct = find_preprocessed_volume(case_dir, "cbct")
        ct = find_preprocessed_volume(case_dir, "ct")
        if cbct.is_file() and ct.is_file():
            pairs.append({"case_id": case_dir.name, "cbct": str(cbct), "ct": str(ct), "preprocessed": "1"})
    if not pairs:
        raise FileNotFoundError(f"No preprocessed cases found under {split_dir}.")
    return pairs


def build_training_records(config: Mapping[str, Any], split: str) -> List[Dict[str, str]]:
    if bool(config.get("use_preprocessed", True)):
        return build_preprocessed_pair_list(config, split)
    return build_pair_list(config, split)


def find_preprocessed_volume(case_dir: Path, stem: str) -> Path:
    candidates = [case_dir / f"{stem}.npy"]
    candidates.extend(case_dir / f"{stem}{suffix}" for suffix in MEDICAL_SUFFIXES)
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


def read_medical_image(path: str | Path) -> Tuple[np.ndarray, sitk.Image]:
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image)
    return array, image


def normalize_hu_np(volume: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
    volume = np.asarray(volume, dtype=np.float32)
    volume = np.clip(volume, float(hu_min), float(hu_max))
    volume = (volume - float(hu_min)) / (float(hu_max) - float(hu_min))
    return volume * 2.0 - 1.0


def denormalize_hu_np(volume: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
    volume = np.asarray(volume, dtype=np.float32)
    volume = np.clip(volume, -1.0, 1.0)
    volume = (volume + 1.0) * 0.5
    return volume * (float(hu_max) - float(hu_min)) + float(hu_min)


def resize_zhw(volume: np.ndarray, target_hw: Sequence[int]) -> np.ndarray:
    target_h, target_w = [int(v) for v in target_hw]
    if volume.shape[-2:] == (target_h, target_w):
        return np.asarray(volume, dtype=np.float32)

    tensor = torch.from_numpy(np.ascontiguousarray(volume.astype(np.float32, copy=False)))[None, None]
    with torch.no_grad():
        resized = F.interpolate(
            tensor,
            size=(int(volume.shape[0]), target_h, target_w),
            mode="trilinear",
            align_corners=False,
        )
    return resized[0, 0].cpu().numpy()


def preprocess_volume(array: np.ndarray, config: Mapping[str, Any]) -> np.ndarray:
    volume = normalize_hu_np(array, float(config["hu_min"]), float(config["hu_max"]))
    return resize_zhw(volume, config["xy_size"])


def cast_array(array: np.ndarray, reference_dtype: np.dtype, output_dtype: str = "same") -> np.ndarray:
    dtype_name = str(output_dtype).lower()
    dtype = np.dtype(reference_dtype if dtype_name == "same" else output_dtype)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(np.rint(array), info.min, info.max).astype(dtype)
    return array.astype(dtype)


def write_medical_like(array: np.ndarray, reference_image: sitk.Image, output_path: str | Path) -> None:
    output_image = sitk.GetImageFromArray(array)
    output_image.CopyInformation(reference_image)
    sitk.WriteImage(output_image, str(output_path))


def pad_to_dhw(volume: np.ndarray, patch_dhw: Sequence[int], fill_value: float = -1.0) -> np.ndarray:
    pads = []
    for current, target in zip(volume.shape, patch_dhw):
        missing = max(0, int(target) - int(current))
        before = missing // 2
        after = missing - before
        pads.append((before, after))
    if any(before or after for before, after in pads):
        volume = np.pad(volume, pads, mode="constant", constant_values=float(fill_value))
    return volume


def random_crop_zhw(volume: np.ndarray, patch_dhw: Sequence[int], fill_value: float = -1.0) -> np.ndarray:
    volume = pad_to_dhw(volume, patch_dhw, fill_value)
    starts = []
    for current, target in zip(volume.shape, patch_dhw):
        max_start = max(0, int(current) - int(target))
        starts.append(0 if max_start == 0 else int(np.random.randint(0, max_start + 1)))
    d0, h0, w0 = starts
    patch_d, patch_h, patch_w = [int(v) for v in patch_dhw]
    return np.asarray(volume[d0 : d0 + patch_d, h0 : h0 + patch_h, w0 : w0 + patch_w], dtype=np.float32)


def random_crop_pair_zhw(
    volume_a: np.ndarray,
    volume_b: np.ndarray,
    patch_dhw: Sequence[int],
    fill_value: float = -1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    volume_a = pad_to_dhw(volume_a, patch_dhw, fill_value)
    volume_b = pad_to_dhw(volume_b, patch_dhw, fill_value)
    common_shape = tuple(min(a, b) for a, b in zip(volume_a.shape, volume_b.shape))
    volume_a = volume_a[: common_shape[0], : common_shape[1], : common_shape[2]]
    volume_b = volume_b[: common_shape[0], : common_shape[1], : common_shape[2]]

    starts = []
    for current, target in zip(common_shape, patch_dhw):
        max_start = max(0, int(current) - int(target))
        starts.append(0 if max_start == 0 else int(np.random.randint(0, max_start + 1)))
    d0, h0, w0 = starts
    patch_d, patch_h, patch_w = [int(v) for v in patch_dhw]
    patch_a = volume_a[d0 : d0 + patch_d, h0 : h0 + patch_h, w0 : w0 + patch_w]
    patch_b = volume_b[d0 : d0 + patch_d, h0 : h0 + patch_h, w0 : w0 + patch_w]
    return np.asarray(patch_a, dtype=np.float32), np.asarray(patch_b, dtype=np.float32)


class CycleGAN3DPatchDataset(Dataset):
    def __init__(
        self,
        records: Sequence[Mapping[str, str]],
        config: Mapping[str, Any],
        steps_per_epoch: Optional[int] = None,
        paired: bool = False,
    ) -> None:
        self.records = list(records)
        if not self.records:
            raise ValueError("records must not be empty.")
        self.config = config
        self.patch_dhw = hwd_to_dhw(config["patch_size_hwd"])
        self.paired = paired
        self.cache_data = bool(config.get("cache_data", False))
        self._cache: Dict[Tuple[str, str], np.ndarray] = {}
        if steps_per_epoch is None:
            self.length = len(self.records)
        else:
            self.length = int(steps_per_epoch) * int(config.get("batch_size", 1))

    def __len__(self) -> int:
        return self.length

    def _load_volume(self, record: Mapping[str, str], key: str) -> np.ndarray:
        path = str(record[key])
        cache_key = (key, path)
        if self.cache_data and cache_key in self._cache:
            return self._cache[cache_key]

        if path.lower().endswith(".npy"):
            volume = np.load(path, mmap_mode=None if self.cache_data else "r")
        elif str(record.get("preprocessed", "0")) == "1":
            volume, _ = read_medical_image(path)
            volume = np.asarray(volume, dtype=np.float32)
        else:
            array, _ = read_medical_image(path)
            volume = preprocess_volume(array, self.config)

        if self.cache_data:
            volume = np.asarray(volume, dtype=np.float32)
            self._cache[cache_key] = volume
        return volume

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        if self.paired:
            record_a = record_b = self.records[index % len(self.records)]
        else:
            record_a = self.records[index % len(self.records)]
            record_b = self.records[int(np.random.randint(0, len(self.records)))]

        volume_a = self._load_volume(record_a, "cbct")
        volume_b = self._load_volume(record_b, "ct")

        if self.paired:
            patch_a, patch_b = random_crop_pair_zhw(volume_a, volume_b, self.patch_dhw)
        else:
            patch_a = random_crop_zhw(volume_a, self.patch_dhw)
            patch_b = random_crop_zhw(volume_b, self.patch_dhw)

        return {
            "A": torch.from_numpy(np.array(patch_a[None], dtype=np.float32, copy=True)),
            "B": torch.from_numpy(np.array(patch_b[None], dtype=np.float32, copy=True)),
            "case_id_A": str(record_a["case_id"]),
            "case_id_B": str(record_b["case_id"]),
        }


class ImagePool:
    def __init__(self, pool_size: int) -> None:
        self.pool_size = int(pool_size)
        self.images: List[torch.Tensor] = []

    def query(self, images: torch.Tensor) -> torch.Tensor:
        if self.pool_size <= 0:
            return images.detach()

        device = images.device
        output = []
        for image in images.detach():
            image_cpu = image.unsqueeze(0).cpu()
            if len(self.images) < self.pool_size:
                self.images.append(image_cpu.clone())
                output.append(image.unsqueeze(0))
            elif random.random() > 0.5:
                index = random.randint(0, self.pool_size - 1)
                old = self.images[index].to(device=device)
                self.images[index] = image_cpu.clone()
                output.append(old)
            else:
                output.append(image.unsqueeze(0))
        return torch.cat(output, dim=0)


def safe_torch_load(path: str | Path, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def save_checkpoint(path: str | Path, payload: Mapping[str, Any]) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(payload), checkpoint_path)


def resolve_resume_checkpoint(config: Mapping[str, Any], resume_arg: Optional[str]) -> Optional[Path]:
    resume_value = resume_arg if resume_arg is not None else config.get("resume_checkpoint")
    output_dir = Path(str(config["output_dir"]))

    if resume_value:
        text = str(resume_value).strip()
        if text.lower() in {"latest", "best"}:
            return output_dir / f"{text.lower()}.pt"
        return Path(text)

    latest = output_dir / "latest.pt"
    if bool(config.get("auto_resume", False)) and latest.is_file():
        return latest
    return None
