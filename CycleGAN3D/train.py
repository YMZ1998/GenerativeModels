from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import count_parameters, get_discriminator_logits, make_discriminator, make_generator, set_requires_grad
from utils import (
    CycleGAN3DPatchDataset,
    ImagePool,
    build_training_records,
    print_config_summary,
    read_config,
    resolve_resume_checkpoint,
    safe_torch_load,
    save_checkpoint,
    seed_worker,
    set_seed,
)


class VisdomLogger:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.enabled = bool(config.get("visdom_enabled", False))
        self.viz = None
        self.loss_win = "cyclegan3d_losses"
        self.sample_win = "cyclegan3d_samples"
        self.rotate_k = int(config.get("visdom_rotate_k", 0))
        self.flip_lr = bool(config.get("visdom_flip_lr", False))
        self.flip_ud = bool(config.get("visdom_flip_ud", False))
        self.num_images = int(config.get("visdom_num_images", 3))
        if not self.enabled:
            return

        try:
            import visdom

            self.viz = visdom.Visdom(
                server=str(config.get("visdom_server", "http://localhost")),
                port=int(config.get("visdom_port", 8097)),
                env=str(config.get("visdom_env", "cyclegan3d")),
                use_incoming_socket=False,
            )
            if not self.viz.check_connection(timeout_seconds=3):
                print("Visdom is enabled but the server is not reachable. Training will continue without Visdom.")
                self.enabled = False
        except Exception as exc:
            print(f"Visdom setup failed: {exc}. Training will continue without Visdom.")
            self.enabled = False

    def plot_losses(self, epoch: int, metrics: Mapping[str, float]) -> None:
        if not self.enabled or self.viz is None:
            return
        labels = list(metrics.keys())
        y = np.array([[float(metrics[label]) for label in labels]], dtype=np.float32)
        x = np.array([[float(epoch)] * len(labels)], dtype=np.float32)
        update = "append" if self.viz.win_exists(self.loss_win) else None
        self.viz.line(
            X=x,
            Y=y,
            win=self.loss_win,
            update=update,
            opts={"title": "3D CycleGAN Loss", "xlabel": "epoch", "ylabel": "loss", "legend": labels},
        )

    def show_samples(self, epoch: int, real_a: torch.Tensor, fake_b: torch.Tensor, real_b: torch.Tensor) -> None:
        if not self.enabled or self.viz is None:
            return
        try:
            images = []
            count = min(self.num_images, int(real_a.shape[0]))
            for i in range(count):
                for volume in (real_a[i], fake_b[i], real_b[i]):
                    center = int(volume.shape[1] // 2)
                    image = volume[:, center, :, :].detach().cpu().clamp(-1, 1)
                    image = (image + 1.0) * 0.5
                    if self.rotate_k:
                        image = torch.rot90(image, k=self.rotate_k, dims=(-2, -1))
                    if self.flip_lr:
                        image = torch.flip(image, dims=(-1,))
                    if self.flip_ud:
                        image = torch.flip(image, dims=(-2,))
                    images.append(image)
            if images:
                grid = torch.stack(images, dim=0)
                self.viz.images(
                    grid,
                    nrow=3,
                    win=self.sample_win,
                    opts={"title": f"Epoch {epoch}: CBCT / generated CT / real CT"},
                )
        except Exception as exc:
            print(f"Visdom sample visualization failed: {exc}")


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
        torch.backends.cudnn.benchmark = True
    return device


def make_scaler(device: torch.device, enabled: bool) -> GradScaler:
    enabled = bool(enabled and device.type == "cuda")
    try:
        return GradScaler(device.type, enabled=enabled)
    except TypeError:
        return GradScaler(enabled=enabled)


def autocast_context(device: torch.device, enabled: bool):
    return autocast(device.type, enabled=bool(enabled and device.type == "cuda"))


def move_batch(batch: Mapping[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return batch["A"].to(device=device, non_blocking=True), batch["B"].to(device=device, non_blocking=True)


def mse_gan_loss(prediction: torch.Tensor, target_is_real: bool, criterion: nn.Module) -> torch.Tensor:
    target = torch.ones_like(prediction) if target_is_real else torch.zeros_like(prediction)
    return criterion(prediction, target)


def discriminator_loss(
    discriminator: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    pred_real = get_discriminator_logits(discriminator(real))
    pred_fake = get_discriminator_logits(discriminator(fake.detach()))
    return 0.5 * (mse_gan_loss(pred_real, True, criterion) + mse_gan_loss(pred_fake, False, criterion))


def make_dataloader(
    records,
    config: Mapping[str, Any],
    steps_per_epoch: int,
    paired: bool,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    dataset = CycleGAN3DPatchDataset(records, config, steps_per_epoch=steps_per_epoch, paired=paired)
    num_workers = int(config.get("num_workers", 0))
    kwargs: Dict[str, Any] = {
        "batch_size": int(config["batch_size"]),
        "shuffle": shuffle,
        "drop_last": True,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def train_one_epoch(
    epoch: int,
    config: Mapping[str, Any],
    device: torch.device,
    loader: DataLoader,
    models: Mapping[str, nn.Module],
    optimizers: Mapping[str, torch.optim.Optimizer],
    scaler: GradScaler,
    fake_pool_a: ImagePool,
    fake_pool_b: ImagePool,
) -> Dict[str, float]:
    G_A2B = models["G_A2B"]
    G_B2A = models["G_B2A"]
    D_A = models["D_A"]
    D_B = models["D_B"]
    optimizer_G = optimizers["G"]
    optimizer_D = optimizers["D"]

    for model in models.values():
        model.train()

    gan_criterion = nn.MSELoss()
    l1_criterion = nn.L1Loss()
    use_amp = bool(config.get("amp", True))
    lambda_cycle = float(config["lambda_cycle"])
    lambda_identity = float(config["lambda_identity"])
    train_steps = int(config["train_steps_per_epoch"])
    log_interval = max(1, int(config.get("log_interval", 10)))
    show_progress = bool(config.get("show_batch_progress", True))

    totals = {"G": 0.0, "D": 0.0, "cycle": 0.0, "identity": 0.0, "gan": 0.0}
    progress = tqdm(
        loader,
        total=train_steps,
        desc=f"Epoch {epoch:04d}/{int(config['epochs']):04d} train",
        unit="batch",
        dynamic_ncols=True,
        mininterval=1.0,
        disable=not show_progress,
    )
    for step, batch in enumerate(progress, start=1):
        if step > train_steps:
            break
        real_A, real_B = move_batch(batch, device)

        set_requires_grad([D_A, D_B], False)
        optimizer_G.zero_grad(set_to_none=True)
        with autocast_context(device, use_amp):
            fake_B = G_A2B(real_A)
            rec_A = G_B2A(fake_B)
            fake_A = G_B2A(real_B)
            rec_B = G_A2B(fake_A)
            idt_A = G_B2A(real_A)
            idt_B = G_A2B(real_B)

            pred_fake_B = get_discriminator_logits(D_B(fake_B))
            pred_fake_A = get_discriminator_logits(D_A(fake_A))
            loss_gan = mse_gan_loss(pred_fake_B, True, gan_criterion) + mse_gan_loss(pred_fake_A, True, gan_criterion)
            loss_cycle = (l1_criterion(rec_A, real_A) + l1_criterion(rec_B, real_B)) * lambda_cycle
            loss_identity = (l1_criterion(idt_A, real_A) + l1_criterion(idt_B, real_B)) * lambda_identity
            loss_G = loss_gan + loss_cycle + loss_identity

        scaler.scale(loss_G).backward()
        scaler.step(optimizer_G)

        set_requires_grad([D_A, D_B], True)
        optimizer_D.zero_grad(set_to_none=True)
        fake_A_for_D = fake_pool_a.query(fake_A.detach()).to(device=device, non_blocking=True)
        fake_B_for_D = fake_pool_b.query(fake_B.detach()).to(device=device, non_blocking=True)
        with autocast_context(device, use_amp):
            loss_D_A = discriminator_loss(D_A, real_A, fake_A_for_D, gan_criterion)
            loss_D_B = discriminator_loss(D_B, real_B, fake_B_for_D, gan_criterion)
            loss_D = loss_D_A + loss_D_B

        scaler.scale(loss_D).backward()
        scaler.step(optimizer_D)
        scaler.update()

        values = {
            "G": float(loss_G.detach().cpu()),
            "D": float(loss_D.detach().cpu()),
            "cycle": float(loss_cycle.detach().cpu()),
            "identity": float(loss_identity.detach().cpu()),
            "gan": float(loss_gan.detach().cpu()),
        }
        for key, value in values.items():
            totals[key] += value
        if show_progress and (step == 1 or step % log_interval == 0 or step == train_steps):
            progress.set_postfix(G=f"{totals['G'] / step:.4f}", D=f"{totals['D'] / step:.4f}")

    count = max(1, min(train_steps, step))
    return {key: value / count for key, value in totals.items()}


@torch.no_grad()
def validate(
    epoch: int,
    config: Mapping[str, Any],
    device: torch.device,
    loader: DataLoader,
    models: Mapping[str, nn.Module],
) -> tuple[Dict[str, float], Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    G_A2B = models["G_A2B"]
    G_B2A = models["G_B2A"]
    for model in models.values():
        model.eval()

    l1_criterion = nn.L1Loss()
    use_amp = bool(config.get("amp", True))
    val_steps = int(config["val_steps_per_epoch"])
    show_progress = bool(config.get("show_batch_progress", True))
    sample_limit = max(1, int(config.get("visdom_num_images", 6)))
    totals = {"val_l1_cbct2ct": 0.0, "val_cycle": 0.0, "val_identity": 0.0}
    sample_real_a: list[torch.Tensor] = []
    sample_fake_b: list[torch.Tensor] = []
    sample_real_b: list[torch.Tensor] = []

    progress = tqdm(
        loader,
        total=val_steps,
        desc=f"Epoch {epoch:04d}/{int(config['epochs']):04d} val",
        unit="batch",
        dynamic_ncols=True,
        mininterval=1.0,
        leave=False,
        disable=not show_progress,
    )
    for step, batch in enumerate(progress, start=1):
        if step > val_steps:
            break
        real_A, real_B = move_batch(batch, device)
        with autocast_context(device, use_amp):
            fake_B = G_A2B(real_A)
            fake_A = G_B2A(real_B)
            rec_A = G_B2A(fake_B)
            rec_B = G_A2B(fake_A)
            idt_A = G_B2A(real_A)
            idt_B = G_A2B(real_B)
            val_l1 = l1_criterion(fake_B, real_B)
            val_cycle = l1_criterion(rec_A, real_A) + l1_criterion(rec_B, real_B)
            val_identity = l1_criterion(idt_A, real_A) + l1_criterion(idt_B, real_B)
        totals["val_l1_cbct2ct"] += float(val_l1.detach().cpu())
        totals["val_cycle"] += float(val_cycle.detach().cpu())
        totals["val_identity"] += float(val_identity.detach().cpu())
        collected = sum(int(tensor.shape[0]) for tensor in sample_real_a)
        if collected < sample_limit:
            take = min(sample_limit - collected, int(real_A.shape[0]))
            sample_real_a.append(real_A[:take].detach().cpu())
            sample_fake_b.append(fake_B[:take].detach().cpu())
            sample_real_b.append(real_B[:take].detach().cpu())
        if show_progress:
            progress.set_postfix(l1=f"{totals['val_l1_cbct2ct'] / step:.4f}")

    count = max(1, min(val_steps, step))
    sample = None
    if sample_real_a:
        sample = (
            torch.cat(sample_real_a, dim=0)[:sample_limit],
            torch.cat(sample_fake_b, dim=0)[:sample_limit],
            torch.cat(sample_real_b, dim=0)[:sample_limit],
        )
    return {key: value / count for key, value in totals.items()}, sample


def checkpoint_payload(
    epoch: int,
    best_metric: float,
    config: Mapping[str, Any],
    models: Mapping[str, nn.Module],
    optimizers: Mapping[str, torch.optim.Optimizer],
    scaler: GradScaler,
) -> Dict[str, Any]:
    return {
        "epoch": epoch,
        "best_metric": best_metric,
        "config": dict(config),
        "G_CBCT2CT": models["G_A2B"].state_dict(),
        "G_CT2CBCT": models["G_B2A"].state_dict(),
        "D_CBCT": models["D_A"].state_dict(),
        "D_CT": models["D_B"].state_dict(),
        "optimizer_G": optimizers["G"].state_dict(),
        "optimizer_D": optimizers["D"].state_dict(),
        "scaler": scaler.state_dict(),
    }


def load_resume(
    checkpoint_path: Path,
    models: Mapping[str, nn.Module],
    optimizers: Mapping[str, torch.optim.Optimizer],
    scaler: GradScaler,
    device: torch.device,
) -> tuple[int, float]:
    checkpoint = safe_torch_load(checkpoint_path, map_location=device)
    models["G_A2B"].load_state_dict(checkpoint["G_CBCT2CT"])
    models["G_B2A"].load_state_dict(checkpoint["G_CT2CBCT"])
    models["D_A"].load_state_dict(checkpoint["D_CBCT"])
    models["D_B"].load_state_dict(checkpoint["D_CT"])
    if "optimizer_G" in checkpoint:
        optimizers["G"].load_state_dict(checkpoint["optimizer_G"])
    if "optimizer_D" in checkpoint:
        optimizers["D"].load_state_dict(checkpoint["optimizer_D"])
    if "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    return int(checkpoint.get("epoch", 0)) + 1, float(checkpoint.get("best_metric", float("inf")))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a 3D CycleGAN baseline for CBCT-to-CT synthesis.")
    parser.add_argument("--config", default="./config.json", help="Path to config.json.")
    parser.add_argument("--resume", default=None, help="Resume checkpoint: latest, best, or path.")
    args = parser.parse_args()

    config = read_config(args.config)
    set_seed(int(config["seed"]))
    device = get_device(config)

    print_config_summary(
        config,
        [
            "data_root",
            "data_layout",
            "preprocessed_dir",
            "use_preprocessed",
            "output_dir",
            "device",
            "xy_size",
            "patch_size_hwd",
            "batch_size",
            "epochs",
            "train_steps_per_epoch",
            "val_steps_per_epoch",
            "lr_g",
            "lr_d",
            "lambda_cycle",
            "lambda_identity",
            "generator_channels",
            "generator_res_blocks",
            "discriminator_channels",
            "discriminator_layers",
            "amp",
            "visdom_enabled",
        ],
        title="3D CycleGAN Training",
    )
    print(f"Resolved device: {device}")

    train_records = build_training_records(config, "train")
    val_records = build_training_records(config, "val")
    print(f"Train cases: {len(train_records)} | Val cases: {len(val_records)}")

    train_loader = make_dataloader(
        train_records,
        config,
        int(config["train_steps_per_epoch"]),
        paired=False,
        shuffle=True,
        device=device,
    )
    val_loader = make_dataloader(
        val_records,
        config,
        int(config["val_steps_per_epoch"]),
        paired=True,
        shuffle=False,
        device=device,
    )

    models = {
        "G_A2B": make_generator(config).to(device),
        "G_B2A": make_generator(config).to(device),
        "D_A": make_discriminator(config).to(device),
        "D_B": make_discriminator(config).to(device),
    }
    print(
        "Parameters: "
        f"G_CBCT2CT={count_parameters(models['G_A2B']):,}, "
        f"G_CT2CBCT={count_parameters(models['G_B2A']):,}, "
        f"D_CBCT={count_parameters(models['D_A']):,}, "
        f"D_CT={count_parameters(models['D_B']):,}"
    )

    optimizers = {
        "G": torch.optim.Adam(
            itertools.chain(models["G_A2B"].parameters(), models["G_B2A"].parameters()),
            lr=float(config["lr_g"]),
            betas=(float(config["beta1"]), float(config["beta2"])),
        ),
        "D": torch.optim.Adam(
            itertools.chain(models["D_A"].parameters(), models["D_B"].parameters()),
            lr=float(config["lr_d"]),
            betas=(float(config["beta1"]), float(config["beta2"])),
        ),
    }
    scaler = make_scaler(device, bool(config.get("amp", True)))
    fake_pool_a = ImagePool(int(config.get("pool_size", 0)))
    fake_pool_b = ImagePool(int(config.get("pool_size", 0)))
    visualizer = VisdomLogger(config)

    output_dir = Path(str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 1
    best_metric = float("inf")
    resume_path = resolve_resume_checkpoint(config, args.resume)
    if resume_path is not None:
        if resume_path.is_file():
            start_epoch, best_metric = load_resume(resume_path, models, optimizers, scaler, device)
            print(f"Resumed from {resume_path} at epoch {start_epoch}. Best metric: {best_metric:.6f}")
        elif args.resume or config.get("resume_checkpoint"):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

    for epoch in range(start_epoch, int(config["epochs"]) + 1):
        start_time = time.time()
        train_metrics = train_one_epoch(
            epoch,
            config,
            device,
            train_loader,
            models,
            optimizers,
            scaler,
            fake_pool_a,
            fake_pool_b,
        )

        val_metrics: Dict[str, float] = {}
        sample = None
        if epoch % int(config["val_interval"]) == 0:
            val_metrics, sample = validate(epoch, config, device, val_loader, models)
            metric = val_metrics["val_l1_cbct2ct"]
            if metric < best_metric:
                best_metric = metric
                save_checkpoint(output_dir / "best.pt", checkpoint_payload(epoch, best_metric, config, models, optimizers, scaler))

        save_checkpoint(output_dir / "latest.pt", checkpoint_payload(epoch, best_metric, config, models, optimizers, scaler))
        if epoch % int(config["save_interval"]) == 0:
            save_checkpoint(
                output_dir / f"epoch_{epoch:04d}.pt",
                checkpoint_payload(epoch, best_metric, config, models, optimizers, scaler),
            )

        if val_metrics and epoch % int(config.get("visdom_interval", 1)) == 0:
            visualizer.plot_losses(
                epoch,
                {
                    "train_G": train_metrics["G"],
                    "train_D": train_metrics["D"],
                    "val_l1": val_metrics["val_l1_cbct2ct"],
                },
            )
            if sample is not None:
                visualizer.show_samples(epoch, sample[0], sample[1], sample[2])

        elapsed = time.time() - start_time
        val_text = " ".join(f"{key}={value:.6f}" for key, value in val_metrics.items()) if val_metrics else "val=skipped"
        print(
            f"Epoch {epoch:04d}/{int(config['epochs']):04d} "
            f"train_G={train_metrics['G']:.6f} train_D={train_metrics['D']:.6f} "
            f"{val_text} best_val_l1={best_metric:.6f} time={elapsed:.1f}s"
        )


if __name__ == "__main__":
    main()
